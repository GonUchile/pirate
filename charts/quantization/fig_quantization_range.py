#!/usr/bin/env python3
"""Pérdida de RANGO al re-cuantizar: el mismo pulso de B_001 escalado ×1 ... ×100.

Se trabaja en el marco alineado al DM real (cada canal desplazado por su retardo), con el
modelo sin ruido del generador (misma física) + la misma realización de ruido para todas las
escalas. La cuantización es por muestra, así que da lo mismo cuantizar antes o después de alinear.

  a) función de transferencia de cada cuantizador (valor de entrada -> valor guardado), con la
     distribución del ruido y el valor del pulso por píxel para cada escala
  b) S/N medido frente al S/N verdadero: dónde satura cada cuantización
  c) el pulso alineado con escala ×30 (S/N ~4600): sin cuantizar, 8 bits y 4 bits

Uso: .venv/bin/python fig_quantization_range.py
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

HERE = Path(__file__).resolve().parent
(HERE / "plots_21504").mkdir(exist_ok=True)
SIM = HERE.parent / "simulator"                   # generador y física del simulador
spec = importlib.util.spec_from_file_location("gen", SIM / "12_generate_comparison_set_21504.py")
GEN = importlib.util.module_from_spec(spec)
spec.loader.exec_module(GEN)
G = GEN.G
SET = os.environ.get("CHARTS_SET_DIR_TEST") or os.environ.get("CHARTS_SET_DIR")
if not SET:
    sys.exit("ERROR: export CHARTS_SET_DIR=/carpeta/del/grupo3 (o CHARTS_SET_DIR_TEST para un set chico)")
SET = Path(SET)
TXT, TXT2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
ESQ = [("sin cuantizar", "#8a8984"), ("8 bits (4 niv/σ, hasta +55σ)", "#2a78d6"),
       ("4 bits 3 niv/σ (±2,3σ)", "#eb6834"), ("4 bits 2 niv/σ (±3,5σ)", "#1baf7a"),
       ("4 bits 1 niv/σ (±7σ)", "#4a3aa7")]
ESCALAS = [1, 2, 3, 5, 10, 20, 30, 50, 100]


def q8(x):        # el .fil: floor(32,5 + 4x) recortado a 0-255, devuelto en sigma
    return (np.clip(np.floor(x * 4 + 32.5), 0, 255) - 32.0) / 4.0


def q4(x, niv):   # desde los 8 bits, como haría el emisor
    return np.clip(np.rint(q8(x) * niv), -7, 7) / niv


CUANT = [lambda x: x, q8, lambda x: q4(x, 3), lambda x: q4(x, 2), lambda x: q4(x, 1)]


def snr_alineado(al, nchan):
    """al: (tiempo, canal) alineado; S/N con el mejor boxcar, ruido medido lejos del pulso."""
    s = al.sum(axis=1)
    ruido = np.r_[s[:700], s[-700:]]
    s = s - ruido.mean()
    sd = ruido.std()
    c = np.concatenate([[0.0], np.cumsum(s)])
    mejor = 0.0
    for w in G.ANCHOS[G.ANCHOS <= 256]:
        v = (c[w:] - c[:-w]) / (sd * np.sqrt(w))
        mejor = max(mejor, float(v[850:1150].max()))
    return mejor


def main():
    r = next(x for x in csv.DictReader(open(SET / "B_amplitud_fija" / "manifest.csv")) if x["name"] == "B_001")
    p = {k: float(r[k]) for k in ("dm", "width_s", "scat_ref_s", "sp_idx", "sp_run")}
    canales, t_c = GEN.base(p, G.DM_INDEX, float(r["t_top_s"]), int(r["nsamp"]))
    esc = float(r["escala_jy"]) / GEN.SIGMA
    nch = len(canales)
    K = 2000
    # señal alineada: cada canal centrado en su llegada
    sen = np.zeros((K, nch), np.float32)
    for c, (i0, v) in enumerate(canales):
        ic = int(round(t_c[c] / 1e-3))
        a = i0 - ic + K // 2
        lo, hi = max(0, a), min(K, a + len(v))
        if hi > lo:
            sen[lo:hi, c] = esc * v[lo - a:hi - a]
    ruido = np.random.default_rng(7).standard_normal((K, nch), dtype=np.float32)
    pico = float(sen.max())
    print(f"B_001: pico del pulso por píxel = {pico:.2f} σ (escala ×1)")

    res = np.zeros((len(ESCALAS), len(CUANT)))
    for i, e in enumerate(ESCALAS):
        x = ruido + e * sen
        for j, fq in enumerate(CUANT):
            res[i, j] = snr_alineado(fq(x), nch)
        print(f"  ×{e:<4d} pico/píxel {e * pico:6.2f} σ  " + "  ".join(f"{v:9.1f}" for v in res[i]), flush=True)
    snr_real = res[:, 0]

    fig = plt.figure(figsize=(16, 11))
    fig.patch.set_facecolor("#fcfcfb")
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.05], hspace=0.38, wspace=0.28, top=0.9)
    fig.suptitle("La pérdida de rango de 4 bits: el mismo pulso (B_001) cada vez más brillante",
                 fontsize=15, fontweight="bold", color=TXT, x=0.05, ha="left")

    # a) transferencia
    ax = fig.add_subplot(gs[0, 0])
    xx = np.linspace(-6, 12, 4000)
    for (nom, col), fq in zip(ESQ, CUANT):
        ax.plot(xx, fq(xx), color=col, lw=2.2 if nom.startswith("sin") else 1.4, label=nom)
    dens = np.exp(-xx ** 2 / 2)
    ax.fill_between(xx, -8, -8 + 2.5 * dens, color="#8a8984", alpha=0.25, lw=0)
    ax.text(0, -7.6, "ruido (gaussiana)", ha="center", fontsize=8, color=TXT2)
    for e, m in ((1, "×1"), (10, "×10"), (30, "×30")):
        ax.axvline(e * pico, color=TXT2, ls=":", lw=0.8)
        ax.text(e * pico, 10.5, f"pico\n{m}", ha="center", fontsize=8, color=TXT2)
    ax.set_xlim(-6, 12)
    ax.set_ylim(-8, 12)
    ax.set_xlabel("valor real de un píxel (σ)")
    ax.set_ylabel("valor guardado (σ)")
    ax.set_title("a  Qué guarda cada formato: todo lo que pasa\n    del tope se recorta", loc="left",
                 fontsize=11, fontweight="bold", color=TXT)
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    ax.grid(color=GRID, lw=0.6)

    # b) S/N medido vs real
    ax = fig.add_subplot(gs[0, 1:])
    for j, (nom, col) in enumerate(ESQ):
        ax.plot(snr_real, res[:, j] / snr_real * 100, "o-", color=col, lw=1.8, ms=6, label=nom)
    for i, e in enumerate(ESCALAS):
        ax.text(snr_real[i], 101.2, f"×{e}\n{e * pico:.1f}σ", ha="center", fontsize=7.5, color=TXT2)
    ax.set_xscale("log")
    ax.set_ylim(55, 104)
    ax.axhline(100, color=TXT2, lw=0.8)
    ax.set_xlabel("S/N verdadero del pulso (sin cuantizar)")
    ax.set_ylabel("S/N medido / S/N verdadero (%)")
    ax.set_title("b  S/N conservado según el brillo (arriba: escala y pico por píxel)", loc="left",
                 fontsize=11, fontweight="bold", color=TXT)
    ax.legend(fontsize=9, frameon=False, loc="lower left")
    ax.grid(color=GRID, lw=0.6)

    # c) pulso alineado ×30
    e = 30
    x = ruido + e * sen
    i_snr = ESCALAS.index(e)
    perfiles = []
    for j, ((nom, col), fq) in enumerate(zip(ESQ[:3], CUANT[:3])):
        ax = fig.add_subplot(gs[1, j])
        al = fq(x)[K // 2 - 120:K // 2 + 120]
        sub = al.reshape(al.shape[0], 168, 128).mean(axis=2)
        ax.imshow(sub.T, aspect="auto", cmap="viridis", vmin=-0.3, vmax=1.6 * e * pico / 4,
                  extent=[-120, 120, 300, 501.7], origin="lower")
        ax.set_title(f"c{j + 1}  ×{e}: {nom.split(' (')[0]}\n     S/N medido {res[i_snr, j]:.0f}",
                     loc="left", fontsize=11, fontweight="bold", color=TXT)
        ax.set_xlabel("ms (alineado al DM)")
        if j == 0:
            ax.set_ylabel("MHz")
        perfiles.append(fq(x)[K // 2 - 120:K // 2 + 120, :2048].mean(axis=1))
    fig.text(0.05, 0.015,
             f"Pico del pulso por píxel con escala ×1: {pico:.2f}σ. Con 4 bits y 3 niveles/σ el pulso se recorta cuando "
             "su valor por píxel pasa de ~2σ: en c3 la parte brillante (arriba de 450 MHz) queda aplanada en el tope.",
             fontsize=10, color=TXT2)
    out = HERE / "plots_21504" / "rango_cuantizacion_B_001.png"
    fig.savefig(out, dpi=105, bbox_inches="tight")
    print(out)


if __name__ == "__main__":
    main()
