#!/usr/bin/env python3
"""Figura comparativa: 8 bits del set contra 4 bits con n = 1, 2 y 3 niveles por sigma.

Izquierda: la escalera de cada conversión (valor reconstruido, en sigma, contra la señal de entrada),
con el ruido de entrada (abajo) y el ruido a la salida (a la izquierda). Derecha: S/N que mide pirate
contra el S/N real de un burst de banda completa de 10 ms (todos sus píxeles con la misma altura).
Misma teoría que quantization_theory.py: nivel de 8 bits L = floor(32,5 + 4x),
recorte 0-255; 4 bits = clip(floor((L - 32) * n/4), -7, 7), reconstruido en el centro de cada código
(como el emisor: con n = 2, cada código junta 2 cuentas, centro 32,5).

Uso: environments/fetch/bin/python fig_8_vs_4bits.py  ->  plots_21504/diapo_comparativa_bits.png
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

OUT = Path(__file__).resolve().parent / "plots_21504"
OUT.mkdir(exist_ok=True)
L = np.arange(256)
BORDES = (np.arange(257) - 32.5) / 4.0            # entrada (en sigma) donde empieza cada nivel L
BORDES[0], BORDES[-1] = -np.inf, np.inf
N_PIX = 21504 * 10                                 # burst de referencia: toda la banda durante 10 ms
UMBRAL = 6.5                                       # umbral de operación de pirate (10 FP de ruido por beam·hora)


def cuatro_bits(n):
    """Valor reconstruido (en sigma) para cada nivel L: código q = clip(floor((L - 32)·n/4), -7, 7)."""
    q = np.clip(np.floor((L - 32) * n / 4.0), -7, 7)
    return (q + 0.5) / n


OPCIONES = [                                       # (etiqueta, valor reconstruido por nivel L, tope en sigma)
    ("8 bits (el .fil)", (L - 32) / 4.0, "+55σ"),
    ("4 bits, n = 3", cuatro_bits(3), "±2,3σ"),
    ("4 bits, n = 2 (en uso)", cuatro_bits(2), "±3,5σ"),
    ("4 bits, n = 1", cuatro_bits(1), "±7σ"),
]
COLORES = plt.cm.viridis(np.linspace(0.0, 0.85, len(OPCIONES)))


def prob_niveles(s):
    """P(L) para ruido N(0,1) más una señal s (en sigma)."""
    return np.diff(norm.cdf(BORDES - s))


def fraccion_snr(g, s):
    """S/N medido / S/N verdadero para un píxel con señal s: (E[g|s] - E[g|0]) / (s * σ_g)."""
    p0 = prob_niveles(0.0)
    m0 = (p0 * g).sum()
    sd = np.sqrt((p0 * (g - m0) ** 2).sum())
    return np.array([((prob_niveles(si) * g).sum() - m0) / (si * sd) for si in s])


def main():
    plt.rcParams.update({"font.size": 15})
    fig = plt.figure(figsize=(17, 6.6), facecolor="white")
    gs = fig.add_gridspec(2, 4, width_ratios=[1, 1, 0.12, 1.9], wspace=0.25, hspace=0.42)

    x = np.linspace(-4.5, 8.5, 4001)
    nivel = np.clip(np.floor(32.5 + 4 * x), 0, 255).astype(int)
    ruido = norm.pdf(x)
    p0 = prob_niveles(0.0)
    for i, ((nom, g, tope), c) in enumerate(zip(OPCIONES, COLORES)):
        ax = fig.add_subplot(gs[i // 2, i % 2])
        # Ruido de entrada (igual en los cuatro paneles), sobre el eje horizontal.
        ax.fill_between(x, -8, -8 + ruido * 10, color="0.85", lw=0)
        # Ese mismo ruido después de la conversión (lo que ve pirate), sobre el eje vertical:
        # una barra por valor posible, de largo proporcional a la densidad de probabilidad.
        vals, inv = np.unique(g, return_inverse=True)
        prob = np.bincount(inv, weights=p0)
        paso = np.min(np.diff(vals))
        ax.barh(vals, prob / paso * 6, height=paso * 0.9, left=-4.5, color=c, alpha=0.45, lw=0)
        recorte = prob[0] + prob[-1]
        ax.plot(x, x, color="0.6", lw=1, ls=":")
        ax.plot(x, g[nivel], color=c, lw=2.2)
        ax.set_xlim(-4.5, 8.5)
        ax.set_ylim(-8, 8.5)
        txt_rec = f"{100 * recorte:.2f}".replace(".", ",") + " % del ruido en el tope" if recorte >= 1e-4 else "nada en el tope"
        ax.set_title(f"{nom}\ntope {tope} · {txt_rec}", fontsize=13)
        if i == 0:
            ax.text(1.2, -7.6, "ruido de entrada\n(igual en todos)", fontsize=11, color="0.4")
            ax.text(-4.3, 5.2, "ruido a la salida", fontsize=11, color=c, rotation=0)
        if i // 2 == 1:
            ax.set_xlabel("señal de entrada (σ)")
        if i % 2 == 0:
            ax.set_ylabel("valor que ve pirate (σ)")

    # Derecha: S/N medido contra S/N real del burst completo (la potencia integrada sobre sus píxeles).
    # Burst de referencia: toda la banda (21 504 canales) durante 10 ms, todos los píxeles con la misma
    # altura s; S/N real = s * sqrt(N_PIX).
    ax = fig.add_subplot(gs[:, 3])
    raiz_npix = np.sqrt(N_PIX)
    snr = np.logspace(0, 5, 400)
    s = snr / raiz_npix
    ax.axvspan(5, 100, color="0.92", lw=0)
    ax.text(22, 250, "bursts\ndel set", ha="center", fontsize=13, color="0.35")
    ax.axhline(UMBRAL, color="0.45", lw=1.2, ls="--")
    ax.text(1.15, UMBRAL * 1.18, f"umbral de detección ({UMBRAL:.1f})".replace(".", ","), fontsize=12, color="0.35")
    ax.plot(snr, snr, color="0.6", lw=1, ls=":")
    for (nom, g, _), c in zip(OPCIONES, COLORES):
        f = fraccion_snr(g, s)
        techo = f"{snr[-1] * f[-1]:,.0f}".replace(",", " ")
        ax.plot(snr, snr * f, color=c, lw=2.6, label=f"{nom}: techo {techo}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1, 1e5)
    ax.set_ylim(1, 1e5)
    ax.set_xlabel("S/N real del burst (potencia integrada)")
    ax.set_ylabel("S/N que mide pirate")
    ax.set_title("Burst de banda completa de 10 ms: S/N medido frente al real", fontsize=15)
    ax.grid(alpha=0.3, which="both")
    ax.legend(loc="upper left", fontsize=12.5, frameon=False, title="S/N máximo medible", title_fontsize=12.5)
    sup = ax.secondary_xaxis("top", functions=(lambda v: v / raiz_npix, lambda v: v * raiz_npix))
    sup.set_xlabel("señal en cada píxel (σ)", fontsize=12)

    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / "diapo_comparativa_bits.png", dpi=110, bbox_inches="tight")
    for nom, g, tope in OPCIONES:
        f = fraccion_snr(g, np.array([0.01, 1.0, 4.0, 8.0, 24.0]))
        print(f"{nom:24s} tope {tope:6s} S/N: " + "  ".join(f"{v:.3f}" for v in f) + "   (s = 0,01 / 1 / 4 / 8 / 24 σ)")


if __name__ == "__main__":
    main()
