#!/usr/bin/env python3
"""Baseband del sitio antes y después de la canalización ×32, una figura por antena (entradas 0–7), ambos a 1 ms.

Antes: 672 canales gruesos de 0,3 MHz. Su potencia a 1 ms se obtiene sumando los 32 canales finos de cada uno
(Parseval: con FFT rectangular, Σ_k |X_k|² = 32·Σ_n |x_n|², sin aproximación).
Después: 21 504 canales de 9,375 kHz (01_dynamic_spectrum.py).
Salida: figuras_upcanalizacion/antena{0..7}.png
Uso: python 05_fig_upchannelization.py [ENTRADA ...]   (sin argumentos: las 8)
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

AQUI = Path(__file__).resolve().parent
D = np.load(AQUI / "espectro_dinamico.npy", mmap_mode="r")
f = np.load(AQUI / "frequencies_mhz.npy")
fg = 300.0 + 0.3 * np.arange(672)
db = lambda x: 10 * np.log10(np.maximum(x, 1e-6))
OUT = AQUI / "figuras_upcanalizacion"; OUT.mkdir(exist_ok=True)


def figura(ant):
    P = np.asarray(D[ant])
    G = P.reshape(len(P), 672, 32).sum(-1) / 32            # canales gruesos, misma unidad por canal fino

    fig, ax = plt.subplots(3, 2, figsize=(16, 15))
    # (a)/(b) banda completa, 4 s (32 ms por píxel para graficar)
    for j, (X, ff, nom) in enumerate(((G, fg, "Antes: 672 canales de 0,3 MHz"), (P, f, "Después: 21 504 canales de 9,375 kHz"))):
        w = db(X.reshape(128, 32, -1).mean(1))
        im = ax[0, j].imshow(w.T, aspect="auto", origin="lower", cmap="viridis", extent=[0, 4.096, ff[0], ff[-1]],
                             vmin=np.percentile(w, 1), vmax=np.percentile(w, 99.5))
        fig.colorbar(im, ax=ax[0, j], label="dB"); ax[0, j].set(title=f"{nom} · banda completa (32 ms por píxel)", xlabel="Tiempo (s)", ylabel="Frecuencia (MHz)")
    # (c)/(d) zoom 448–458 MHz, primer 1,5 s, resolución nativa 1 ms
    lo, hi, t1 = 448.0, 458.0, 1500
    sg = (fg >= lo) & (fg <= hi); sf = (f >= lo - 0.15) & (f <= hi + 0.15)
    for j, (X, ff, s, nom) in enumerate(((G, fg, sg, f"Antes: {sg.sum()} canales"), (P, f, sf, f"Después: {sf.sum()} canales (máx. de 3 por fila)"))):
        w = db(X[:t1, s])
        if j == 1:                                     # 1099 filas en ~380 píxeles: máximo de cada 3 canales finos
            n3 = w.shape[1] // 3 * 3                   # para que no desaparezcan las líneas de un solo canal
            w = w[:, :n3].reshape(len(w), -1, 3).max(-1)
        im = ax[1, j].imshow(w.T, aspect="auto", origin="lower", cmap="viridis", extent=[0, t1 / 1000, ff[s][0], ff[s][-1]],
                             vmin=np.percentile(w, 1), vmax=np.percentile(w, 99.7), interpolation="nearest")
        fig.colorbar(im, ax=ax[1, j], label="dB"); ax[1, j].set(title=f"{nom} · zoom {lo:.0f}–{hi:.0f} MHz, 1 ms por píxel", xlabel="Tiempo (s)", ylabel="Frecuencia (MHz)")
    # (e) espectro promedio 405–432 MHz
    lo, hi = 405.0, 432.0
    s = (f >= lo) & (f <= hi); s2 = (fg >= lo) & (fg <= hi)
    ax[2, 0].plot(f[s], db(P[:, s].mean(0)), lw=0.6, color=plt.cm.viridis(0.35), label="después (9,375 kHz)")
    ax[2, 0].step(fg[s2], db(G[:, s2].mean(0)), where="mid", lw=1.8, color="k", label="antes (0,3 MHz)")
    ax[2, 0].set(title=f"Espectro promedio en 4,1 s, {lo:.0f}–{hi:.0f} MHz", xlabel="Frecuencia (MHz)", ylabel="dB"); ax[2, 0].legend()
    # (f) 6 canales gruesos alrededor de una línea angosta (400,6 MHz): líneas resueltas y scalloping
    lo, hi = 399.6, 401.4
    s = (f >= lo) & (f <= hi); s2 = (fg >= lo - 0.1) & (fg <= hi + 0.1)
    ax[2, 1].plot(f[s], db(P[:, s].mean(0)), ".-", ms=3, lw=0.8, color=plt.cm.viridis(0.35), label="después: 32 canales por canal grueso")
    ax[2, 1].step(fg[s2], db(G[:, s2].mean(0)), where="mid", lw=1.8, color="k", label="antes")
    for c in fg[s2]:
        ax[2, 1].axvline(c - 0.15, color="0.8", lw=0.8, zorder=0)
    ax[2, 1].set(title="Zoom a 6 canales gruesos (gris: bordes); se ve el scalloping de la PFB", xlabel="Frecuencia (MHz)", ylabel="dB"); ax[2, 1].legend(fontsize=10)
    fig.suptitle(f"Antena {ant} (entrada {ant}) · baseband del sitio 2026-09-02, antes y después de canalizar ×32 · ambos integrados a 1 ms", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.97)); fig.savefig(OUT / f"antena{ant}.png", dpi=100); plt.close(fig)
    print(OUT / f"antena{ant}.png", flush=True)


for ant in ([int(x) for x in sys.argv[1:]] or range(D.shape[0])):
    figura(ant)
