#!/usr/bin/env python3
"""Cuantización 8 bits -> 4 bits expresada en CUENTAS (los enteros que se guardan).

  a) histograma del ruido de un canal en cuentas de 8 bits (piso 32, 1 sigma = 4 cuentas)
  b) conversión cuenta de 8 bits -> código de 4 bits (-7..+7) para n = 1, 2 y 3 niveles por sigma
  c) S/N conservado según la altura del pulso por píxel, en cuentas de 8 bits sobre el piso
  d) los píxeles de un pulso brillante (B_001 ×10) en cuentas de 8 bits y en código de 4 bits (n = 2)

Uso: .venv/bin/python fig_quantization_counts.py
"""
import os
import sys
import csv
import importlib.util
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

HERE = Path(__file__).resolve().parent
(HERE / "plots_21504").mkdir(exist_ok=True)
SIM = HERE.parent / "01_simulator"                   # generador y física del simulador
spec = importlib.util.spec_from_file_location("gen", SIM / "12_generate_comparison_set_21504.py")
GEN = importlib.util.module_from_spec(spec)
spec.loader.exec_module(GEN)
G = GEN.G
SET = os.environ.get("CHARTS_SET_DIR_TEST") or os.environ.get("CHARTS_SET_DIR")
if not SET:
    sys.exit("ERROR: export CHARTS_SET_DIR=/carpeta/del/grupo3 (o CHARTS_SET_DIR_TEST para un set chico)")
SET = Path(SET)
TXT, TXT2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
COL = {1: "#4a3aa7", 2: "#1baf7a", 3: "#eb6834", 8: "#2a78d6"}


def cod4(c8, n):
    """Código de 4 bits desde la cuenta de 8 bits. n = 2 con centro 32,5 (sin empates)."""
    centro = 32.5 if n == 2 else 32.0
    return np.clip(np.rint((c8 - centro) * n / 4.0), -7, 7)


def main():
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": GRID, "axes.labelcolor": TXT2, "xtick.color": TXT2, "ytick.color": TXT2})
    fig = plt.figure(figsize=(16, 12))
    fig.patch.set_facecolor("#fcfcfb")
    gs = fig.add_gridspec(2, 3, hspace=0.38, wspace=0.3, top=0.9)
    fig.suptitle("Cuantización en cuentas: 8 bits (0–255) del .fil  →  código de 4 bits (−7…+7) para pirate",
                 fontsize=15, fontweight="bold", color=TXT, x=0.05, ha="left")

    # a) ruido en cuentas
    ax = fig.add_subplot(gs[0, 0])
    c = np.arange(0, 70)
    p = norm.cdf(c + 1 - 32.5, scale=4) - norm.cdf(c - 32.5, scale=4)
    ax.bar(c, 100 * p, width=0.85, color=COL[8])
    for k, s in ((-2, "−2σ"), (-1, "−1σ"), (1, "+1σ"), (2, "+2σ"), (3, "+3σ")):
        ax.axvline(32 + 4 * k, color=TXT2, ls=":", lw=0.8)
        ax.text(32 + 4 * k, 10.6, s, ha="center", fontsize=8, color=TXT2)
    ax.axvline(32, color=TXT, lw=1)
    ax.text(32, 11.4, "piso = 32", ha="center", fontsize=9, color=TXT, fontweight="bold")
    ax.set_xlim(10, 56)
    ax.set_ylim(0, 12)
    ax.set_xlabel("cuenta de 8 bits")
    ax.set_ylabel("% de las muestras de ruido")
    ax.set_title("a  Ruido de un canal en el .fil\n    1σ = 4 cuentas", loc="left", fontsize=11, fontweight="bold",
                 color=TXT)

    # b) conversión 8 -> 4 bits
    ax = fig.add_subplot(gs[0, 1:])
    c8 = np.arange(0, 90)
    ax.axvspan(28, 36, color="#8a8984", alpha=0.12, lw=0)
    ax.axvspan(24, 40, color="#8a8984", alpha=0.08, lw=0)
    ax.text(32, -8.6, "68 % del ruido\nentre 28 y 36", ha="center", fontsize=8, color=TXT2)
    for n, etq in ((1, "n = 1: 1 nivel/σ (4 cuentas por código)"), (2, "n = 2: 2 niveles/σ (2 cuentas por código)"),
                   (3, "n = 3: 3 niveles/σ (~1,33 cuentas por código)")):
        q = cod4(c8, n)
        ax.step(c8, q, where="mid", color=COL[n], lw=2, label=etq)
        tope = int(c8[np.argmax(q >= 7)])
        ax.plot(tope, 7, "o", color=COL[n], ms=6)
        ax.annotate(f"desde {tope} cuentas\nse guarda 7", xy=(tope, 7), xytext=(tope + 3, 4.6 + n * 0.0 - (n - 2) * 1.6),
                    fontsize=8, color=COL[n], arrowprops=dict(arrowstyle="->", color=COL[n], lw=0.7))
    ax.axhline(7, color=TXT2, lw=0.6, ls="--")
    ax.axhline(-7, color=TXT2, lw=0.6, ls="--")
    ax.set_xlim(0, 90)
    ax.set_ylim(-9.5, 9)
    ax.set_yticks(range(-8, 9, 2))
    ax.set_xlabel("cuenta de 8 bits (lo que está en el .fil)")
    ax.set_ylabel("código de 4 bits (lo que recibe pirate)")
    ax.set_title("b  Conversión cuenta por cuenta: 15 códigos posibles (el −8 está reservado para muestras enmascaradas)",
                 loc="left", fontsize=11, fontweight="bold", color=TXT)
    ax.legend(loc="upper left", fontsize=9, frameon=False)
    ax.grid(color=GRID, lw=0.6)
    sec = ax.secondary_xaxis("top", functions=(lambda x: (x - 32) / 4, lambda s: 32 + 4 * s))
    sec.set_xlabel("en σ (referencia)", color=TXT2)

    # c) S/N conservado vs altura del pulso en cuentas (medido con fig_quantization_range.py, B_001)
    ax = fig.add_subplot(gs[1, 0])
    pico_sigma = np.array([0.80, 1.60, 2.41, 4.01, 8.02, 16.05, 24.07, 40.12, 80.24])
    snr = {"real": [153.3, 306.7, 460.1, 766.8, 1534.0, 3068.4, 4602.7, 7671.5, 15343.4],
           8: [153.2, 306.5, 459.7, 766.0, 1532.4, 3065.2, 4598.1, 7663.8, 14104.4],
           3: [150.9, 294.3, 421.9, 606.4, 816.5, 988.0, 1078.6, 1184.7, 1318.1],
           2: [152.0, 303.4, 451.7, 714.8, 1066.6, 1340.1, 1480.1, 1644.3, 1848.4],
           1: [146.9, 294.7, 442.2, 737.6, 1436.5, 2110.9, 2426.7, 2783.8, 3217.7]}
    real = np.array(snr["real"])
    cuentas = 4 * pico_sigma
    for k, etq in ((8, "8 bits"), (2, "4 bits n = 2"), (3, "4 bits n = 3"), (1, "4 bits n = 1")):
        ax.plot(cuentas, 100 * np.array(snr[k]) / real, "o-", color=COL[k], lw=1.8, ms=5, label=etq)
    for x, s in zip(cuentas[[0, 3, 4, 6]], real[[0, 3, 4, 6]]):
        ax.text(x, 103, f"S/N\n{s:.0f}", ha="center", fontsize=7.5, color=TXT2)
    ax.set_xscale("log")
    ax.set_ylim(0, 110)
    ax.axhline(100, color=TXT2, lw=0.8)
    ax.set_xlabel("altura del pulso por píxel\n(cuentas de 8 bits sobre el piso)")
    ax.set_ylabel("S/N conservado (%)")
    ax.set_title("c  B_001 cada vez más brillante", loc="left", fontsize=11, fontweight="bold", color=TXT)
    ax.legend(fontsize=8, frameon=False, loc="lower left")
    ax.grid(color=GRID, lw=0.6)

    # d) píxeles del pulso ×10 en cuentas
    r = next(x for x in csv.DictReader(open(SET / "B_amplitud_fija" / "manifest.csv")) if x["name"] == "B_001")
    p = {k: float(r[k]) for k in ("dm", "width_s", "scat_ref_s", "sp_idx", "sp_run")}
    canales, t_c = GEN.base(p, G.DM_INDEX, float(r["t_top_s"]), int(r["nsamp"]))
    esc = float(r["escala_jy"]) / GEN.SIGMA * 10
    ch = np.arange(len(canales) - 40, len(canales))                      # 40 canales finos cerca de 501 MHz
    K = 60
    sen = np.zeros((K, ch.size), np.float32)
    for j, cidx in enumerate(ch):
        i0, v = canales[cidx]
        ic = int(round(t_c[cidx] / 1e-3))
        a = i0 - ic + K // 2
        lo, hi = max(0, a), min(K, a + len(v))
        if hi > lo:
            sen[lo:hi, j] = esc * v[lo - a:hi - a]
    x = np.random.default_rng(3).standard_normal(sen.shape).astype(np.float32) + sen
    c8 = np.clip(np.floor(x * 4 + 32.5), 0, 255)
    q2 = cod4(c8, 2)
    for k, (dat, tit, vmin, vmax, cl) in enumerate(((c8, "cuenta de 8 bits", 16, 80, "cuentas (8 bits)"),
                                                    (q2, "código de 4 bits (n = 2)", -7, 7, "código (4 bits)"))):
        ax = fig.add_subplot(gs[1, 1 + k])
        im = ax.imshow(dat.T, aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax, interpolation="nearest",
                       extent=[-K // 2, K // 2, 0, ch.size])
        ax.set_xlabel("ms (alineado al DM)")
        ax.set_ylabel("canal fino (cerca de 501 MHz)")
        n_tope = int((dat >= (7 if k else 255)).sum())
        ax.set_title(f"d{k + 1}  B_001 ×10 (S/N ~1500): {tit}\n     máximo {dat.max():.0f}" +
                     (f"  ·  {n_tope} píxeles recortados en 7" if k else ""),
                     loc="left", fontsize=11, fontweight="bold", color=TXT)
        cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.02)
        cb.set_label(cl)
    fig.text(0.05, 0.02, "Con n = 2 cada código de 4 bits agrupa 2 cuentas de 8 bits: 32–33 → 0, 34–35 → +1, … y desde "
             "46 cuentas (+3,5σ) todo queda en 7. En d2 el pulso pierde su forma: el núcleo (≈46–65 cuentas en d1) "
             "queda plano en 7.", fontsize=10, color=TXT2)
    out = HERE / "plots_21504" / "cuantizacion_en_cuentas.png"
    fig.savefig(out, dpi=105, bbox_inches="tight")
    print(out)


if __name__ == "__main__":
    main()
