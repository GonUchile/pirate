#!/usr/bin/env python3
"""Cómo se ve un pulso con cada cuantización (mismo ruido y misma señal).

Filas: sin cuantizar · 8 bits (el .fil) · 4 bits 3 niveles/σ · 4 bits 1 nivel/σ.
Columnas:
  1) barrido: parte alta de la banda (donde emite B_001), promediado a 64 canales × 4 ms
  2) pulso alineado al DM real: cada canal desplazado por su retardo, 168 sub-bandas × 1 ms
  3) píxeles crudos: 48 canales finos (9,39 kHz) × 120 ms a resolución completa, sin promediar

Uso: .venv/bin/python fig_quantized_pulses.py [NOMBRE=B_001]
"""
import os
import csv
import importlib.util
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
(HERE / "plots_21504").mkdir(exist_ok=True)
spec = importlib.util.spec_from_file_location("cq", HERE / "quantization_21504.py")
CQ = importlib.util.module_from_spec(spec)
spec.loader.exec_module(CQ)
G, VAL = CQ.G, CQ.VAL
SET = os.environ.get("CHARTS_SET_DIR_TEST") or os.environ.get("CHARTS_SET_DIR")
if not SET:
    sys.exit("ERROR: export CHARTS_SET_DIR=/carpeta/del/grupo3 (o CHARTS_SET_DIR_TEST para un set chico)")
SET = Path(SET)
TXT, TXT2 = "#0b0b0b", "#52514e"


def cuantizar(d8, niv):
    return np.clip(np.rint(d8 / 4.0 * niv), -7, 7) / niv


def main():
    nombre = sys.argv[1] if len(sys.argv) > 1 else "B_001"
    carpeta = "B_amplitud_fija" if nombre.startswith("B") else "A_snr_fijo"
    r = next(x for x in csv.DictReader(open(SET / carpeta / "manifest.csv")) if x["name"] == nombre)
    xf = CQ.reconstruir_float(r)[:, ::-1]
    d8 = np.asarray(VAL.leer(r["fil_path"])).astype(np.float32) - 32.0
    f = VAL.freqs_fil({"freq_min_mhz": 299.8545, "freq_max_mhz": 501.7455})        # descendente
    dm, t_top = float(r["dm"]), float(r["t_top_s"])
    ret = G.delay_s(dm, G.DM_INDEX, f) - G.delay_s(dm, G.DM_INDEX, 501.6)
    sh = np.round(ret / 1e-3).astype(int)
    i_top = int(round(t_top / 1e-3))

    versiones = [("sin cuantizar", xf), ("8 bits (el .fil)\n4 niveles/σ", d8 / 4.0),
                 ("4 bits\n3 niveles/σ", cuantizar(d8, 3)), ("4 bits\n1 nivel/σ", cuantizar(d8, 1))]

    # ventanas
    c_hi = slice(0, 8192)                                            # 501,7 -> ~424 MHz (parte alta)
    t0b, t1b = i_top - 300, i_top + sh[8191] + 300                   # barrido de la parte alta
    k = np.arange(-150, 150)                                         # alrededor del pulso alineado
    c480 = int(np.argmin(np.abs(f - 480.0)))
    cz = np.arange(c480 - 24, c480 + 24)
    tz = i_top + sh[c480] + np.arange(-60, 60)

    fig, axs = plt.subplots(4, 3, figsize=(16, 15), gridspec_kw={"width_ratios": [1.5, 1, 1], "hspace": 0.28,
                                                                  "wspace": 0.18})
    fig.patch.set_facecolor("#fcfcfb")
    fig.suptitle(f"{nombre}: el mismo pulso (DM {dm:.0f}, ancho {float(r['width_s']) * 1e3:.1f} ms, "
                 f"S/N {float(r['snr_esperado']):.0f}) con cada cuantización", fontsize=15, fontweight="bold",
                 color=TXT, x=0.06, ha="left", y=0.995)
    for fila, (etq, x) in enumerate(versiones):
        # 1) barrido promediado
        ax = axs[fila, 0]
        w = x[t0b:t1b, c_hi]
        nt = (w.shape[0] // 4) * 4
        wf = w[:nt].reshape(nt // 4, 4, 128, 64).mean(axis=(1, 3))
        sig = 1 / np.sqrt(4 * 64)
        ax.imshow(wf.T / sig, aspect="auto", cmap="viridis", vmin=-2, vmax=6,
                  extent=[t0b * 1e-3, (t0b + nt) * 1e-3, f[8191], f[0]])
        ax.set_ylabel(etq, fontsize=12, fontweight="bold", color=TXT, rotation=0, ha="right", va="center",
                      labelpad=12)
        ax.yaxis.set_label_coords(-0.1, 0.5)
        ax.set_yticks([430, 450, 470, 490])
        # 2) alineado al DM
        ax = axs[fila, 1]
        idx = i_top + sh[None, :] + k[:, None]                       # (t, canal)
        al = x[idx, np.arange(x.shape[1])[None, :]]
        al = al.reshape(len(k), 168, 128).mean(axis=2)
        ax.imshow(al.T / (1 / np.sqrt(128)), aspect="auto", cmap="viridis", vmin=-2, vmax=6,
                  extent=[k[0], k[-1], f[-1], f[0]])
        ax.set_yticks([300, 350, 400, 450, 500])
        # 3) píxeles crudos
        ax = axs[fila, 2]
        z = x[np.ix_(tz, cz)]
        im = ax.imshow(z.T, aspect="auto", cmap="viridis", vmin=-2.5, vmax=2.5, interpolation="nearest",
                       extent=[-60, 60, f[cz[-1]], f[cz[0]]])
        vals = np.unique(np.round(z * 12) / 12).size
        ax.text(0.02, 0.04, f"{'continuo' if fila == 0 else f'{vals} valores distintos'}", transform=ax.transAxes,
                fontsize=9, color="white", fontweight="bold")
    titulos = ["1  Barrido (424–501 MHz; 64 canales × 4 ms)",
               "2  Alineado al DM real",
               "3  Píxeles crudos (48 canales finos)"]
    for j, t in enumerate(titulos):
        axs[0, j].set_title(t, fontsize=11, fontweight="bold", color=TXT, loc="left")
    axs[3, 0].set_xlabel("tiempo desde el inicio del archivo (s)")
    axs[3, 1].set_xlabel("ms respecto a la llegada (alineado)")
    axs[3, 2].set_xlabel("ms respecto a la llegada a 480 MHz")
    for a in axs[:, 1:].ravel():
        a.tick_params(labelsize=8)
    axs[0, 1].set_ylabel("MHz", fontsize=9)
    axs[0, 2].set_ylabel("MHz", fontsize=9)
    cb = fig.colorbar(im, ax=axs[:, 2], fraction=0.04, pad=0.02)
    cb.set_label("columna 3: valor de cada píxel (σ de un canal)")
    fig.text(0.06, 0.005, "Columnas 1 y 2 en σ del promedio (escala -2 a 6). En la columna 3 el pulso no se ve en un "
             "solo píxel: por canal y por muestra aporta ~0,4σ; aparece recién al sumar canales (columnas 1 y 2).",
             fontsize=10, color=TXT2)
    out = HERE / "plots_21504" / f"pulsos_cuantizados_{nombre}.png"
    fig.savefig(out, dpi=100, bbox_inches="tight")
    print(out)


if __name__ == "__main__":
    main()
