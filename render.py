import os
os.environ["MPLBACKEND"] = "Agg"

import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio
from matplotlib.patches import Ellipse, FancyBboxPatch, Rectangle, PathPatch
from matplotlib.path import Path

# ============================================================
# OUTPUT
# ============================================================
OUT = "Equacao_do_Movimento_Spontanea_3Fases_16x9_60s.mp4"

# ============================================================
# VIDEO SETTINGS
# ============================================================
FPS = 20
DURATION_S = 60.0
W_IN, H_IN = 12.8, 7.2
DPI = 120

# ============================================================
# MODEL SETTINGS (didáctico)
# ============================================================
PEEP = 0.0  # espontânea a ar ambiente (Patm ~ 0)

# Fases (0-20s normal; 20-40 R↑; 40-60 C↓ -> E↑)
PHASES = [
    ("Normal",  0.0, 20.0, 5.0, 0.10),   # R=5 cmH2O/(L/s), C=0.10 L/cmH2O -> E=10
    ("R ↑",    20.0, 40.0, 15.0, 0.10),  # resistência aumenta
    ("C ↓",    40.0, 60.0, 5.0, 0.05),   # complacência diminui (E aumenta)
]

# Esforço muscular (Pmus) por ciclo: pausa 1s, insp 2s, pausa 1s, exp 2s, pausa 1s -> 7s
T_HOLD_EE  = 1.0
T_INSP     = 2.0
T_HOLD_EI  = 1.0
T_EXP      = 2.0
T_HOLD_EE2 = 1.0
T_CYCLE    = T_HOLD_EE + T_INSP + T_HOLD_EI + T_EXP + T_HOLD_EE2  # 7.0 s

PMUS_PEAK = 8.0  # cmH2O (didáctico; dá VT visível)

# ============================================================
# Helpers
# ============================================================
def smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return 0.5 - 0.5*np.cos(np.pi*x)

def phase_in_cycle(tau):
    a = T_HOLD_EE
    b = a + T_INSP
    c = b + T_HOLD_EI
    d = c + T_EXP
    if tau < a:
        return "hold_ee", tau / max(T_HOLD_EE, 1e-6)
    if tau < b:
        return "insp", (tau - a) / max(T_INSP, 1e-6)
    if tau < c:
        return "hold_ei", (tau - b) / max(T_HOLD_EI, 1e-6)
    if tau < d:
        return "exp", (tau - c) / max(T_EXP, 1e-6)
    return "hold_ee2", (tau - d) / max(T_HOLD_EE2, 1e-6)

def pmus_of_tau(tau):
    """Pmus(t): 0 nas pausas; sobe suavemente na inspiração; mantém; desce na expiração (relaxamento)."""
    ph, x = phase_in_cycle(tau)
    if ph in ("hold_ee", "hold_ee2"):
        return 0.0
    if ph == "insp":
        return PMUS_PEAK * smoothstep(x)
    if ph == "hold_ei":
        return PMUS_PEAK
    # exp / relaxamento
    return PMUS_PEAK * (1.0 - smoothstep(x))

def canvas_to_rgb(fig):
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()

def history(t, window, fps):
    t0 = max(0.0, t - window)
    n = int(max(120, min(int(window*fps), int((t - t0)*fps + 1))))
    return np.linspace(t0, t, n)

def get_params_for_time(t):
    """Escolhe R, C conforme a fase temporal."""
    for name, t0, t1, R, C in PHASES:
        if t0 <= t < t1:
            return name, R, C
    # fallback
    name, _, _, R, C = PHASES[-1]
    return name, R, C

# ============================================================
# Precompute high-res timeline for the whole 60s (piecewise R,C)
# ============================================================
dt = 1.0 / FPS
N = int(DURATION_S * FPS)
t_grid = np.linspace(0.0, DURATION_S, N, endpoint=False)

Pmus = np.zeros(N)
R_t  = np.zeros(N)
C_t  = np.zeros(N)
E_t  = np.zeros(N)

for i, tt in enumerate(t_grid):
    _, Rv, Cv = get_params_for_time(tt)
    R_t[i] = Rv
    C_t[i] = Cv
    E_t[i] = 1.0 / Cv
    tau = tt % T_CYCLE
    Pmus[i] = pmus_of_tau(tau)

# Integrate: dV/dt = (Pmus - E*V - PEEP)/R
V = np.zeros(N)       # L above FRC
Flow = np.zeros(N)    # L/s
P_el = np.zeros(N)    # E*V
P_res = np.zeros(N)   # R*Flow

for i in range(1, N):
    Rv = R_t[i-1]
    Ev = E_t[i-1]
    pm = Pmus[i-1]
    # flow from equation rearranged
    flow = (pm - Ev*V[i-1] - PEEP) / max(Rv, 1e-6)
    Flow[i-1] = flow
    V[i] = V[i-1] + flow * dt

# last flow estimate
Flow[-1] = Flow[-2]

P_el = E_t * V
P_res = R_t * Flow

# "Check": Pmus ≈ Pel + Pres + PEEP (numeric minor mismatch ok)
P_sum = P_el + P_res + PEEP

# Scale volume to a didactic VT (visual only) while preserving shape for display
# (We keep physics terms as computed; the plotted V is scaled just for readability.)
V_plot = V.copy()
Vmin, Vmax = float(np.min(V_plot)), float(np.max(V_plot))
VT_TARGET = 0.5  # L visual
if (Vmax - Vmin) > 1e-9:
    V_plot = (V_plot - Vmin) * (VT_TARGET / (Vmax - Vmin))
else:
    V_plot = V_plot * 0.0

# For bars/components we use the real computed pressures (P_el, P_res, Pmus)
# ============================================================
# Drawing utilities
# ============================================================
def _mix(c1, c2, a):
    c1 = np.array(c1, dtype=float)
    c2 = np.array(c2, dtype=float)
    return tuple(np.clip((1-a)*c1 + a*c2, 0, 1))

def draw_lungs(ax, inflate=0.0):
    """Pulmão anatómico simples (sem notch branco), com traqueia e vasos, dentro de uma caixa torácica."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # caixa torácica
    thor = Rectangle((0.08, 0.10), 0.70, 0.80, fill=False, lw=3, edgecolor="#111827", alpha=0.70)
    ax.add_patch(thor)

    ax.text(
        0.43, 0.88, "Caixa torácica",
        ha="center", va="center",
        fontsize=10, weight="bold", color="#111827",
        bbox=dict(boxstyle="round,pad=0.14", facecolor="white", edgecolor="none", alpha=0.85)
    )

    cx, cy = 0.43, 0.62
    s = 1.0 + 0.12 * float(np.clip(inflate, 0, 1))

    pink = (0.93, 0.62, 0.68)
    pink_dark = (0.78, 0.36, 0.46)
    highlight = (0.98, 0.86, 0.88)
    fill = _mix(pink, highlight, 0.32 + 0.25*inflate)
    edge = _mix(pink_dark, (0.55, 0.20, 0.28), 0.20)

    lobe_w = 0.20 * s
    lobe_h = 0.32 * s

    left = Ellipse((cx - 0.12*s, cy), width=lobe_w, height=lobe_h, angle=10,
                   facecolor=fill, edgecolor=edge, linewidth=4.0, alpha=0.98)
    right = Ellipse((cx + 0.12*s, cy), width=lobe_w, height=lobe_h, angle=-10,
                    facecolor=fill, edgecolor=edge, linewidth=4.0, alpha=0.98)
    ax.add_patch(left); ax.add_patch(right)

    # fissuras
    for side in (-1, 1):
        x0 = cx + side*0.12*s
        y0 = cy + 0.06*s
        xs = np.linspace(x0 - 0.06*s*side, x0 + 0.02*s*side, 120)
        ys = y0 - 0.05*s*np.sin(np.linspace(0, np.pi, 120))
        ax.plot(xs, ys, color=_mix(edge, (0,0,0), 0.15), lw=1.1, alpha=0.30, clip_on=True)

    # traqueia
    tr_w = 0.040*s
    tr_h = 0.095*s
    tr_x = cx - tr_w/2
    tr_y = cy + 0.14*s
    tr = FancyBboxPatch((tr_x, tr_y), tr_w, tr_h,
                        boxstyle="round,pad=0.006,rounding_size=0.012",
                        facecolor="#111827", edgecolor="#111827", linewidth=0)
    ax.add_patch(tr)

    # brônquios
    bronchi_top = tr_y
    bronchi_mid = cy + 0.095*s
    bronchi_low = cy + 0.055*s
    verts = [
        (cx, bronchi_top),
        (cx, bronchi_mid),
        (cx - 0.06*s, bronchi_low),
        (cx, bronchi_mid),
        (cx + 0.06*s, bronchi_low),
    ]
    codes = [Path.MOVETO, Path.LINETO, Path.LINETO, Path.MOVETO, Path.LINETO]
    bronchi = PathPatch(Path(verts, codes), edgecolor="#111827", linewidth=3.6,
                        facecolor="none", capstyle="round")
    ax.add_patch(bronchi)

    # vasos (simplificados)
    red = (0.86, 0.18, 0.18)
    blue = (0.12, 0.45, 0.78)
    vs = 1.0 + 0.10*inflate

    def branch(points, color, lw, a):
        pv = Path(points, [Path.MOVETO] + [Path.LINETO]*(len(points)-1))
        ax.add_patch(PathPatch(pv, edgecolor=color, linewidth=lw, facecolor="none",
                               alpha=a, capstyle="round"))

    rv = [(cx + 0.02*s, cy + 0.10*s),
          (cx + 0.10*s*vs, cy + 0.07*s),
          (cx + 0.14*s*vs, cy + 0.00*s),
          (cx + 0.14*s*vs, cy - 0.06*s)]
    lv = [(cx - 0.02*s, cy + 0.10*s),
          (cx - 0.10*s*vs, cy + 0.07*s),
          (cx - 0.14*s*vs, cy + 0.00*s),
          (cx - 0.14*s*vs, cy - 0.06*s)]
    branch(rv, red, 2.6, 0.85); branch(lv, red, 2.6, 0.85)
    branch([(x, y - 0.015*s) for (x,y) in rv], blue, 2.2, 0.75)
    branch([(x, y - 0.015*s) for (x,y) in lv], blue, 2.2, 0.75)

    # diafragma (curva)
    # inflate ~ esforço -> desce
    dia_y = 0.22 - 0.10 * float(np.clip(inflate, 0, 1))
    xs = np.linspace(0.12, 0.76, 240)
    arch = dia_y + 0.06 * np.sin(np.pi * (xs - 0.12) / (0.76 - 0.12))
    ax.plot(xs, arch, lw=7, color="#111827")
    ax.text(0.80, dia_y + 0.02, "Diafragma",
            fontsize=9.5, color="#111827",
            bbox=dict(boxstyle="round,pad=0.12", facecolor="white", edgecolor="none", alpha=0.85),
            va="center")

def draw_bar(ax, x, y, w, h, value, vmin, vmax, color, label):
    """Barra vertical com etiqueta."""
    value = float(np.clip(value, vmin, vmax))
    frac = 0.0 if vmax == vmin else (value - vmin) / (vmax - vmin)

    ax.add_patch(Rectangle((x, y), w, h, fill=False, lw=2.0, edgecolor="#111827", alpha=0.85))
    ax.add_patch(Rectangle((x, y), w, h*frac, facecolor=color, edgecolor="none", alpha=0.60))
    ax.plot([x-0.01, x+w+0.01], [y+h*frac, y+h*frac], color="#111827", lw=2.0)

    ax.text(x + w/2, y + h + 0.02, label, ha="center", fontsize=10, weight="bold", color="#111827")
    ax.text(x + w/2, y - 0.045, f"{value:.1f}", ha="center", fontsize=10, weight="bold", color=color)

# ============================================================
# Render
# ============================================================
fig = plt.figure(figsize=(W_IN, H_IN), dpi=DPI)
writer = imageio.get_writer(
    OUT,
    fps=FPS,
    codec="libx264",
    macro_block_size=1,
    ffmpeg_params=["-preset", "ultrafast", "-crf", "24"]
)

WINDOW = 12.0  # janela de histórico dos gráficos
total_frames = N

for i in range(total_frames):
    t = t_grid[i]

    phase_name, R_now, C_now = get_params_for_time(t)
    E_now = 1.0 / C_now

    pm = Pmus[i]
    v = V_plot[i]
    flow = Flow[i]

    pel = P_el[i]
    pres = P_res[i]
    psum = P_sum[i]

    # history window
    th = history(t, WINDOW, FPS)
    idx0 = int(max(0, i - len(th) + 1))
    # map th to available indices robustly
    # create index array covering the history uniformly
    j = np.linspace(max(0, i - int(WINDOW*FPS)), i, len(th)).astype(int)

    pm_h = Pmus[j]
    v_h = V_plot[j]
    flow_h = Flow[j] * 60.0
    pel_h = P_el[j]
    pres_h = P_res[j]

    # Layout: 2 rows x 3 cols
    fig.clf()
    gs = fig.add_gridspec(
        2, 3,
        left=0.04, right=0.985, top=0.93, bottom=0.08,
        wspace=0.28, hspace=0.28,
        width_ratios=[1.15, 1.60, 1.05],
        height_ratios=[1.25, 1.00]
    )

    ax_lung = fig.add_subplot(gs[:, 0])
    ax_mid1 = fig.add_subplot(gs[0, 1])
    ax_mid2 = fig.add_subplot(gs[1, 1])
    ax_right = fig.add_subplot(gs[:, 2])

    # ---------------------------
    # Left: lung + bars
    # ---------------------------
    # inflate driven by Pmus (0..PMUS_PEAK)
    inflate = float(np.clip(pm / max(PMUS_PEAK, 1e-6), 0, 1))
    draw_lungs(ax_lung, inflate=inflate)

    # flow arrow (acelerante/desacelerante aparece no gráfico; aqui só direcção)
    arrow_mag = float(np.clip(abs(flow) / 0.8, 0.0, 1.0))
    if flow > 1e-4:
        ax_lung.annotate("", xy=(0.43, 0.62), xytext=(0.90, 0.62),
                         arrowprops=dict(arrowstyle="->", lw=3 + 4*arrow_mag, color="#dc2626"))
        ax_lung.text(0.90, 0.67, "Ar entra", color="#dc2626", fontsize=10, weight="bold",
                     ha="center", bbox=dict(boxstyle="round,pad=0.12", facecolor="white", edgecolor="none", alpha=0.85))
    elif flow < -1e-4:
        ax_lung.annotate("", xy=(0.90, 0.62), xytext=(0.43, 0.62),
                         arrowprops=dict(arrowstyle="->", lw=3 + 4*arrow_mag, color="#16a34a"))
        ax_lung.text(0.90, 0.67, "Ar sai", color="#16a34a", fontsize=10, weight="bold",
                     ha="center", bbox=dict(boxstyle="round,pad=0.12", facecolor="white", edgecolor="none", alpha=0.85))
    else:
        ax_lung.text(0.90, 0.67, "Fluxo=0", color="#6b7280", fontsize=10, weight="bold",
                     ha="center", bbox=dict(boxstyle="round,pad=0.12", facecolor="white", edgecolor="none", alpha=0.85))

    # Bars area
    # Choose sensible range for bars (cmH2O)
    ax_lung.text(0.10, 0.02, f"Fase: {phase_name}   |   R={R_now:.0f}   C={C_now:.2f} (E={E_now:.0f})",
                 fontsize=9.5, weight="bold",
                 bbox=dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor="#e5e7eb", alpha=0.95))

    # Put bars inside the thorax margin zone right side of left panel (axes coords)
    # Values shown: Pmus, Pres, Pel
    ax_lung.add_patch(Rectangle((0.80, 0.10), 0.18, 0.40, facecolor="white", edgecolor="#e5e7eb", alpha=0.90))
    draw_bar(ax_lung, 0.82, 0.12, 0.04, 0.34, pm,   0.0, PMUS_PEAK, "#0ea5e9", "Pmus")
    draw_bar(ax_lung, 0.88, 0.12, 0.04, 0.34, pres, -10.0, 20.0,    "#f97316", "R·Flow")
    draw_bar(ax_lung, 0.94, 0.12, 0.04, 0.34, pel,  0.0,  20.0,     "#a855f7", "E·V")

    # ---------------------------
    # Middle top: equation terms vs time
    # ---------------------------
    ax_mid1.set_title("Equação do movimento — termos (cmH₂O)", fontsize=12, weight="bold")
    ax_mid1.plot(th, pm_h,   lw=3.0, color="#0ea5e9", label="Pmus")
    ax_mid1.plot(th, pres_h, lw=2.6, color="#f97316", label="R·Flow")
    ax_mid1.plot(th, pel_h,  lw=2.6, color="#a855f7", label="E·V")
    ax_mid1.axhline(0, color="#9ca3af", lw=1.0)
    ax_mid1.grid(True, alpha=0.20)
    ax_mid1.legend(loc="upper left", fontsize=9, frameon=True)
    ax_mid1.set_xlabel("Tempo (s)")
    ax_mid1.set_ylabel("cmH₂O")

    # phase shading on mid plots
    for name, t0, t1, _, _ in PHASES:
        ax_mid1.axvspan(max(th[0], t0), min(th[-1], t1), alpha=0.10)

    ax_mid1.scatter([t], [pm], s=60, color="#0ea5e9", zorder=6)

    # ---------------------------
    # Middle bottom: volume and flow
    # ---------------------------
    ax_mid2.set_title("Resposta do sistema — Volume e Fluxo", fontsize=12, weight="bold")
    ax2 = ax_mid2.twinx()

    ax_mid2.plot(th, v_h, lw=3.0, color="#b91c1c", label="V (L) — escala didáctica")
    ax2.plot(th, flow_h, lw=2.8, color="#16a34a", label="Flow (L/min)")

    ax_mid2.set_xlabel("Tempo (s)")
    ax_mid2.set_ylabel("Volume (L)")
    ax2.set_ylabel("Fluxo (L/min)")

    ax_mid2.grid(True, alpha=0.20)
    ax_mid2.set_ylim(-0.05, 0.70)
    ax2.set_ylim(-80, 80)
    ax_mid2.axhline(0, color="#9ca3af", lw=1.0)

    for name, t0, t1, _, _ in PHASES:
        ax_mid2.axvspan(max(th[0], t0), min(th[-1], t1), alpha=0.10)

    # merge legends cleanly
    lines1, labels1 = ax_mid2.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax_mid2.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=9, frameon=True)

    ax_mid2.scatter([t], [v], s=45, color="#2563eb", zorder=6)
    ax2.scatter([t], [flow*60], s=45, color="#2563eb", zorder=6)

    # ---------------------------
    # Right: big equation + live numbers (no overlaps)
    # ---------------------------
    ax_right.set_xlim(0, 1)
    ax_right.set_ylim(0, 1)
    ax_right.axis("off")

    ax_right.text(0.06, 0.95, "Ventilação espontânea", fontsize=14, weight="bold", color="#111827")

    # Main equation (big)
    ax_right.text(
        0.06, 0.86,
        "Equação do movimento (via aérea aberta):",
        fontsize=11.5, weight="bold", color="#111827"
    )
    ax_right.text(
        0.06, 0.78,
        r"$P_{mus}(t) \;=\; E\cdot V(t) \;+\; R\cdot \dot{V}(t) \;+\; PEEP$",
        fontsize=16, weight="bold", color="#111827",
        bbox=dict(boxstyle="round,pad=0.28", facecolor="#eef2ff", edgecolor="#c7d2fe", alpha=0.98)
    )
    ax_right.text(0.06, 0.71, "Aqui: PEEP = 0", fontsize=11, color="#374151")

    # Live numeric decomposition (stacked boxes)
    ax_right.text(0.06, 0.62, "Valores (agora):", fontsize=12.5, weight="bold", color="#111827")

    # Boxes positions (fixed)
    def box(y, text, face, edge, col):
        ax_right.text(
            0.06, y, text,
            fontsize=11.2, weight="bold", color=col,
            bbox=dict(boxstyle="round,pad=0.26", facecolor=face, edgecolor=edge, alpha=0.98),
            va="center"
        )

    box(0.56, f"Pmus = {pm:5.1f} cmH₂O",   "#e0f2fe", "#7dd3fc", "#075985")
    box(0.49, f"E·V  = {pel:5.1f} cmH₂O   (E={E_now:0.0f})", "#f3e8ff", "#d8b4fe", "#6b21a8")
    box(0.42, f"R·Flow = {pres:5.1f} cmH₂O   (R={R_now:0.0f})", "#ffedd5", "#fdba74", "#9a3412")
    box(0.35, f"Soma = {psum:5.1f} cmH₂O", "#ecfccb", "#bef264", "#365314")

    # Small notes (kept short)
    ax_right.text(
        0.06, 0.24,
        "Leitura fisiológica:\n"
        "• Início da inspiração: Pmus sobe → ΔP maior → fluxo acelera.\n"
        "• À medida que V sobe: E·V aumenta → ΔP reduz → fluxo desacelera.\n"
        "• Na expiração: Pmus relaxa → recoil elástico domina → fluxo sai.",
        fontsize=9.6, color="#111827",
        bbox=dict(boxstyle="round,pad=0.30", facecolor="#fff7ed", edgecolor="#fed7aa", alpha=0.98),
        va="top"
    )

    # Phase banner at bottom
    ax_right.text(
        0.06, 0.08,
        "3 fases didácticas: 0–20 Normal | 20–40 R↑ (asma/obstrução) | 40–60 C↓ (pulmão rígido)",
        fontsize=9.6, weight="bold", color="#111827",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#e5e7eb", alpha=0.95)
    )

    # Global title
    fig.suptitle(
        "Equação do Movimento — Ventilação Espontânea (valores em tempo real)",
        fontsize=16, weight="bold", y=0.985
    )

    fig.tight_layout()
    writer.append_data(canvas_to_rgb(fig))

writer.close()
print("OK ->", OUT)
