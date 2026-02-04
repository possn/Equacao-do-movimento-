import os
os.environ["MPLBACKEND"] = "Agg"

import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = "Equacao_Movimento_Spontanea_para_VNI.mp4"

FPS = 20
DURATION = 60
FRAMES = FPS * DURATION
figsize = (12.8, 7.2)

# --------------------------
# Helpers
# --------------------------
def smoothstep(x):
    x = float(np.clip(x, 0.0, 1.0))
    return 0.5 - 0.5 * np.cos(np.pi * x)

def lerp(a, b, x):
    x = float(np.clip(x, 0.0, 1.0))
    return a + (b - a) * x

def draw_bar(ax, x, y, w, h, frac, label, color):
    frac = float(np.clip(frac, 0.0, 1.0))
    ax.add_patch(plt.Rectangle((x, y), w, h, fill=False, lw=2, ec="#111827"))
    ax.add_patch(plt.Rectangle((x, y), w * frac, h, color=color, alpha=0.85))
    ax.text(x + w / 2, y + h + 0.03, label, ha="center",
            fontsize=11, weight="bold", color="#111827")

def draw_overlay_spont_to_vni(ax, t, t_start=37.0, t_fade_end=40.0):
    """
    Overlay 'Respiração espontânea → VNI'
    - Fade-in 37–40s
    - Mantém 40–60s (alpha=1)
    """
    if t < t_start:
        return

    if t < t_fade_end:
        a = smoothstep((t - t_start) / max(t_fade_end - t_start, 1e-6))
    else:
        a = 1.0

    # Lower-third centered
    x0, y0 = 0.16, 0.075
    w, h = 0.68, 0.11

    box = FancyBboxPatch(
        (x0, y0), w, h,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        linewidth=2.0,
        edgecolor="#111827",
        facecolor="#ffffff",
        alpha=0.93 * a
    )
    ax.add_patch(box)

    # Main line
    ax.text(
        x0 + 0.03, y0 + h / 2,
        "Respiração espontânea",
        va="center", ha="left",
        fontsize=13.5, weight="bold",
        color="#111827",
        alpha=a
    )

    arr = FancyArrowPatch(
        (x0 + 0.41, y0 + h / 2),
        (x0 + 0.54, y0 + h / 2),
        arrowstyle="-|>",
        mutation_scale=18,
        linewidth=2.8,
        color="#16a34a",
        alpha=a
    )
    ax.add_patch(arr)

    ax.text(
        x0 + 0.57, y0 + h / 2,
        "VNI",
        va="center", ha="left",
        fontsize=14.5, weight="bold",
        color="#16a34a",
        alpha=a
    )

    # Microtexto clínico (curto, legível, sem poluir)
    ax.text(
        x0 + 0.03, y0 - 0.028,
        "CPAP: ↑FRC/RECRUTAMENTO • PS: ↓WOB (↓Pmus) • Objectivo: estabilizar mecânica e melhorar troca gasosa.",
        va="center", ha="left",
        fontsize=10.6,
        color="#374151",
        alpha=0.95 * a
    )

# --------------------------
# Render
# --------------------------
writer = imageio.get_writer(
    OUT,
    fps=FPS,
    codec="libx264",
    macro_block_size=1,
    ffmpeg_params=["-preset", "ultrafast", "-crf", "23"]
)

for i in range(FRAMES):
    t = i / FPS

    # 3 fases (60s): espontânea base -> carga (R/C) -> VNI
    if t < 20:
        phase = "base"       # espontânea: leitura global
    elif t < 40:
        phase = "load"       # aumenta carga (R ou 1/C) -> Pmus sobe
    else:
        phase = "vni"        # Paw ajuda -> Pmus desce

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Título
    ax.text(
        0.5, 0.94,
        "Equação do Movimento — do espontâneo à VNI",
        ha="center", fontsize=18, weight="bold", color="#111827"
    )

    # Equações (simples e legíveis)
    if phase in ("base", "load"):
        eq = "Pmus = R · Fluxo + V / C"
    else:
        eq = "Pmus + Paw = R · Fluxo + V / C"

    ax.text(0.5, 0.76, eq, ha="center", fontsize=28, weight="bold", color="#111827")

    # Texto curto por fase
    if phase == "base":
        txt = "Respiração espontânea:\nPmus vence a carga mecânica"
    elif phase == "load":
        txt = "↑Carga (↑R e/ou ↓C):\npara o mesmo VT → Pmus sobe"
    else:
        txt = "VNI adiciona Paw:\nparte da carga é “paga” pela pressão\n→ Pmus desce (menos fadiga)"

    ax.text(0.08, 0.56, txt, fontsize=13.5, color="#111827")

    # Valores dinâmicos (didácticos, não “fisiologia perfeita” ao milímetro)
    if phase == "base":
        x = smoothstep(t / 20)
        R_term = lerp(0.30, 0.45, x)
        E_term = lerp(0.30, 0.45, x)   # V/C (elástico)
        Paw = 0.0
    elif phase == "load":
        x = smoothstep((t - 20) / 20)
        R_term = lerp(0.40, 0.80, x)
        E_term = lerp(0.40, 0.85, x)
        Paw = 0.0
    else:
        x = smoothstep((t - 40) / 20)
        # com VNI, mantemos carga alta mas Paw cresce e Pmus baixa
        R_term = 0.75
        E_term = 0.85
        Paw = lerp(0.15, 0.60, x)

    # Pmus = soma dos termos - Paw (clamp 0..1)
    Pmus = np.clip((R_term + E_term) / 2.0 - 0.55 * Paw, 0.05, 0.95)

    # Barras (sempre nos mesmos sítios)
    draw_bar(ax, 0.56, 0.52, 0.34, 0.06, R_term, "R·Fluxo (resistivo)", "#f59e0b")
    draw_bar(ax, 0.56, 0.41, 0.34, 0.06, E_term, "V/C (elástico)", "#8b5cf6")
    draw_bar(ax, 0.56, 0.30, 0.34, 0.06, Pmus, "Pmus (esforço)", "#ef4444")

    if phase == "vni":
        draw_bar(ax, 0.56, 0.19, 0.34, 0.06, Paw, "Paw (VNI)", "#22c55e")

        # Overlay transição (entra 37–40, mantém 40–60)
        draw_overlay_spont_to_vni(ax, t, t_start=37.0, t_fade_end=40.0)

    # Rodapé (muito curto)
    ax.text(
        0.08, 0.12,
        "Ideia-chave: a VNI não “cura” R nem C; reduz o esforço necessário (↓Pmus) e estabiliza volumes.",
        fontsize=11.2, color="#374151"
    )

    fig.canvas.draw()
    frame = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
    writer.append_data(frame)
    plt.close(fig)

writer.close()
print("OK ->", OUT)
