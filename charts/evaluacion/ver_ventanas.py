#!/usr/bin/env python3
"""Muestra ventanas del mapa DM-tiempo guardadas por el grouper v0: 6 frb, 3 nofrb y 3 ruido.
Cada ventana: 256 celdas de DM (centradas en el evento) × 384 celdas de tiempo (chunk anterior, del evento y siguiente).
Uso: python ver_ventanas.py CARPETA_CORRIDA SALIDA.png"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

run, out = Path(sys.argv[1]), sys.argv[2]
idx = pd.read_csv(run / "ventanas" / "indice.csv")
sel = pd.concat([idx[idx.etiqueta == "frb"].sort_values("snr", ascending=False).head(6),
                 idx[idx.etiqueta == "nofrb"].sort_values("snr", ascending=False).head(3),
                 idx[(idx.etiqueta == "ruido") & (idx.estable == 1)].sample(3, random_state=0)])
fig, axs = plt.subplots(3, 4, figsize=(16, 9))
for ax, (_, r) in zip(axs.flat, sel.iterrows()):
    z = np.load(run / "ventanas" / r.npz)
    v = z["ventana"][r.k].astype(np.float32)
    nt = v.shape[1] // 3
    im = ax.imshow(v, aspect="auto", cmap="viridis", vmin=0, vmax=max(6, np.percentile(v, 99.9)), origin="lower")
    ax.axvline(nt, color="white", lw=0.6, ls=":")
    ax.axvline(2 * nt, color="white", lw=0.6, ls=":")
    ax.set_title(f"{r.etiqueta} · {r['name']}\nDM {r.dm:.0f} · S/N {r.snr:.1f} · {r.width_ms:.0f} ms", fontsize=9)
    ax.set_xlabel("celda de tiempo (16 ms)", fontsize=8)
    ax.set_ylabel("celda de DM (0,54) − 128", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.04, label="S/N")
fig.suptitle("Ventanas del mapa DM-tiempo (out_max) guardadas por el grouper v0 · resolución nativa · prueba corta", fontsize=12)
fig.tight_layout()
fig.savefig(out, dpi=110, facecolor="white")
