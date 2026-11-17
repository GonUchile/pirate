#!/usr/bin/env python3
"""Figura explicativa de la cuantización (8 bits del set -> 4 bits para pirate).

1) Qué le hace cada cuantización al ruido de un canal (histogramas de valores).
2) Un canal en el tiempo: el dato original y su versión cuantizada.
3) Cuánto S/N se conserva: teoría (barra) y medido en los bursts brillantes de B (puntos).

Usa B_001 del set de prueba (reconstruido en float con la misma semilla) y
plots_21504/quantization_21504.csv.
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
spec = importlib.util.spec_from_file_location("cq", HERE / "quantization_21504.py")
CQ = importlib.util.module_from_spec(spec)
spec.loader.exec_module(CQ)

SET = os.environ.get("CHARTS_SET_DIR_TEST") or os.environ.get("CHARTS_SET_DIR")
if not SET:
    sys.exit("ERROR: export CHARTS_SET_DIR=/carpeta/del/grupo3 (o CHARTS_SET_DIR_TEST para un set chico)")
SET = Path(SET)
TXT, TXT2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
GRIS = "#8a8984"
ESQ = [  # (clave en el csv, etiqueta, color de la paleta en orden fijo)
    ("8 bits", "8 bits (el .fil)\n4 niveles/σ", "#2a78d6"),
    ("4b 3niv fija", "4 bits, 3 niveles/σ\n(recomendado)", "#eb6834"),
    ("4b 2niv fija", "4 bits, 2 niveles/σ", "#1baf7a"),
    ("4b 1niv fija", "4 bits, 1 nivel/σ", "#4a3aa7"),
    ("4b pirate", "4 bits estilo pirate\n(offset cada 256 ms)", "#e87ba4"),
]
TEO = {"8 bits": 99.74, "4b 3niv fija": 99.03, "4b 2niv fija": 98.22, "4b 1niv fija": 95.35, "4b pirate": 98.22}

plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": TXT2, "xtick.color": TXT2,
                     "ytick.color": TXT2, "axes.titlecolor": TXT, "axes.titlesize": 11, "axes.titleweight": "bold",
                     "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": "#fcfcfb",
                     "axes.facecolor": "#fcfcfb"})


def cuantizar(d8, niv):
    return np.clip(np.rint(d8 / 4.0 * niv), -7, 7) / niv


def main():
    r = next(x for x in csv.DictReader(open(SET / "B_amplitud_fija" / "manifest.csv")) if x["name"] == "B_001")
    xf = CQ.reconstruir_float(r)[:, ::-1]                         # (tiempo, canal desc.), en sigma
    u8 = np.asarray(CQ.VAL.leer(r["fil_path"]))
    d8 = u8.astype(np.float32) - 32.0                              # niveles (1 sigma = 4)

    fig = plt.figure(figsize=(15, 10.5))
    gs = fig.add_gridspec(2, 4, height_ratios=[1, 1.15], hspace=0.62, wspace=0.32, top=0.86)
    fig.suptitle("¿Qué se pierde al pasar de 8 bits a 4 bits?  (set de prueba del grupo 3, 21 504 canales)",
                 fontsize=14, fontweight="bold", color=TXT, x=0.06, ha="left", y=0.995)
    fig.text(0.06, 0.945, "1  El ruido de cada canal, cuantizado (histograma de valores; línea = gaussiana ideal)",
             fontsize=12, fontweight="bold", color=TXT)
    fig.text(0.06, 0.475, "2  Lo que ve el buscador: el pulso dedispersado", fontsize=12, fontweight="bold", color=TXT)
    fig.text(0.53, 0.475, "3  Cuánto S/N se conserva", fontsize=12, fontweight="bold", color=TXT)

    # ---- 1) histogramas del ruido ----
    ruido_f = xf[:2000, ::64].ravel()
    ruido_8 = d8[:2000, ::64].ravel() / 4.0
    paneles = [("float", "sin cuantizar", GRIS, ruido_f, None),
               ("8 bits", "8 bits: 4 niveles/σ", ESQ[0][2], ruido_8, 0.25),
               ("4b 3niv fija", "4 bits: 3 niveles/σ", ESQ[1][2], cuantizar(d8[:2000, ::64].ravel(), 3), 1 / 3),
               ("4b 1niv fija", "4 bits: 1 nivel/σ", ESQ[3][2], cuantizar(d8[:2000, ::64].ravel(), 1), 1.0)]
    xx = np.linspace(-4.5, 4.5, 400)
    for k, (_, tit, col, v, paso) in enumerate(paneles):
        ax = fig.add_subplot(gs[0, k])
        if paso is None:
            ax.hist(v, bins=120, range=(-4.5, 4.5), density=True, color=col, alpha=0.8)
        else:
            vals, cnt = np.unique(np.round(v / paso).astype(int), return_counts=True)
            ax.bar(vals * paso, cnt / cnt.sum() / paso, width=paso * 0.8, color=col)
            if k >= 2:
                tope = 7 * paso
                ax.axvspan(tope, 4.5, color=col, alpha=0.08)
                ax.axvspan(-4.5, -tope, color=col, alpha=0.08)
                ax.annotate(f"todo lo que pasa de ±{tope:.1f}σ\nse junta en el último nivel",
                            xy=(tope, 0.05), xytext=(0.6, 0.93), textcoords="axes fraction", fontsize=8,
                            color=TXT2, arrowprops=dict(arrowstyle="->", color=TXT2, lw=0.6))
        ax.plot(xx, norm.pdf(xx), color=TXT, lw=1, ls="--")
        n_val = "valores continuos" if paso is None else f"{len(np.unique(np.round(v / paso)))} valores distintos"
        ax.set_title(f"{tit}\n{n_val}", fontsize=10)
        ax.set_xlim(-4.5, 4.5)
        ax.set_xlabel("ruido de un canal (σ)")
        ax.set_yticks([])
        if k == 2:
            ax.text(0.5, -0.36, "los \"dientes\": la grilla de 8 bits (σ/4) no calza con la de 4 bits (σ/3);\n"
                    "cada nivel recibe 1 o 2 niveles de 8 bits. Ya está incluido en la teoría.",
                    transform=ax.transAxes, ha="center", fontsize=8, color=TXT2)

    # ---- 2) serie dedispersada de toda la banda ----
    ax = fig.add_subplot(gs[1, :2])
    f = CQ.VAL.freqs_fil({"freq_min_mhz": 299.8545, "freq_max_mhz": 501.7455})
    dm = float(r["dm"])
    versiones = [("sin cuantizar", xf, GRIS, 2.6), ("8 bits", d8 / 4.0, ESQ[0][2], 1.3),
                 ("4 bits, 3 niveles/σ", cuantizar(d8, 3), ESQ[1][2], 1.3),
                 ("4 bits, 1 nivel/σ", cuantizar(d8, 1), ESQ[3][2], 1.3)]
    for nom, x, col, lw in versiones:
        serie, off = CQ.VAL.dedispersar(x, f, dm, CQ.G.DM_INDEX)
        i = int(round(float(r["t_top_s"]) / 1e-3)) + off
        ruido = serie[:i - 2000]
        z = (serie - np.median(ruido)) / ruido.std()
        k = np.arange(-80, 80)
        snr = CQ.snr(serie, i, x.shape[1])
        ax.plot(k, z[i + k], color=col, lw=lw, label=f"{nom}: S/N {snr:.1f}")
    ax.axhline(0, color=GRID, lw=1)
    ax.set_xlabel("ms respecto a la llegada esperada a 501,6 MHz")
    ax.set_ylabel("S/N por muestra de 1 ms")
    ax.set_title("B_001 (DM 727, 21 504 canales sumados al DM real)", fontsize=10)
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    ax.text(0.02, 0.92, "las curvas casi coinciden:\nla diferencia es ~1–5 % del pico", transform=ax.transAxes,
            fontsize=9, color=TXT2, va="top")

    # ---- 3) S/N conservado ----
    ax = fig.add_subplot(gs[1, 2:])
    ax.tick_params(axis="y", pad=4)
    filas = list(csv.DictReader(open(HERE / "plots_21504" / "quantization_21504.csv")))
    brillantes = [x for x in filas if float(x["float"]) > 50]
    y = np.arange(len(ESQ))[::-1]
    for yk, (clave, etq, col) in zip(y, ESQ):
        ax.barh(yk, TEO[clave] - 90, left=90, height=0.55, color=col, alpha=0.35)
        med = np.array([100 * float(x[clave]) / float(x["float"]) for x in brillantes])
        ax.plot(med, np.full(med.size, yk), "o", color=col, ms=8, mec="white", mew=1.2, zorder=3)
        ax.text(TEO[clave] + 0.12, yk + 0.31, f"{TEO[clave]:.1f} %", va="center", fontsize=9, color=TXT)
    ax.set_yticks(y, [e for _, e, _ in ESQ])
    ax.set_xlim(90, 101)
    ax.set_xlabel("S/N conservado respecto al dato sin cuantizar (%)")
    ax.axvline(100, color=TXT2, lw=0.8)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.set_title("barra = teoría  ·  puntos = medido en los 4 bursts brillantes de B (S/N 57–156)", fontsize=10)
    ax.annotate("además pierde ~ancho/256 en pulsos anchos:\nA_007 (scattering 26 ms) perdió 22 %",
                xy=(97.0, y[-1]), xytext=(90.3, y[-1] - 0.75), fontsize=8, color=TXT2,
                arrowprops=dict(arrowstyle="->", color=TXT2, lw=0.6))
    ax.set_ylim(-1.1, len(ESQ) - 0.4)

    out = HERE / "plots_21504" / "cuantizacion_explicada.png"
    fig.savefig(out, dpi=115, bbox_inches="tight")
    print(out)


if __name__ == "__main__":
    main()
