#!/usr/bin/env python3
"""Cómo se ve un FRB con y sin la RFI del sitio, centrado en el pulso (en S/N por píxel).

Fila 1: waterfall dispersado alrededor del barrido (la curva punteada es el retardo por dispersión).
Fila 2: dedispersado al DM del FRB, ±250 ms.
Fila 3: perfil dedispersado (suma de canales), en S/N.
Cada canal se normaliza con su mediana y σ robusta (MAD) en la ventana; los canales saturados (> 50 % en 255) van a 0.
Al promediar n píxeles se multiplica por √n: el ruido de cada píxel queda en σ = 1.
Uso: python 06_figura_frb_con_rfi.py ORIGINAL.fil CON_RFI.fil DM T_TOP_S salida.png [titulo]
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parent))
from sigproc import leer_header  # noqa: E402

orig, conr, dm, ttop, out = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4]), sys.argv[5]
titulo = sys.argv[6] if len(sys.argv) > 6 else Path(orig).stem
K = 4.148808e3                                            # s MHz² (retardo en s)


def carga(p):
    head, h = leer_header(p)
    nch = h["nchans"]; n = (Path(p).stat().st_size - len(head)) // nch
    return np.memmap(p, np.uint8, "r", offset=len(head), shape=(n, nch)), h


def norm(x):
    x = x.astype(np.float32)
    med = np.median(x, 0); mad = 1.4826 * np.median(np.abs(x - med), 0)
    z = (x - med) / np.maximum(mad, 1e-3)
    z[:, ((x == 255).mean(0) > 0.5) | (mad < 0.5)] = 0
    return z


def binea(z, bt, bf):
    nt, nf = z.shape[0] // bt * bt, z.shape[1] // bf * bf
    w = z[:nt, :nf].reshape(nt // bt, bt, nf // bf, bf).mean((1, 3)) * np.sqrt(bt * bf)
    return w - np.median(w, 0)            # quita el sesgo chico de la mediana por canal, que el binning amplifica


fig, ax = plt.subplots(3, 2, figsize=(16, 14), gridspec_kw=dict(height_ratios=[1.1, 1, 0.6]))
for j, (p, nom) in enumerate(((orig, "Sin RFI"), (conr, "Con RFI del sitio"))):
    d, h = carga(p); n, nch = d.shape
    f = h["fch1"] + h["foff"] * np.arange(nch)            # descendente
    ret = K * dm * (f ** -2 - f.max() ** -2)               # s
    # fila 1: barrido dispersado
    t0, t1 = max(0, ttop - 1.0), min(n / 1000, ttop + ret.max() + 1.0)
    z = norm(np.asarray(d[int(t0 * 1000):int(t1 * 1000)]))
    bt, bf = 20, 256
    w = binea(z, bt, bf)
    ax[0, j].imshow(w.T, aspect="auto", origin="upper", cmap="viridis", vmin=-2, vmax=4,
                    extent=[t0, t0 + w.shape[0] * bt / 1000, f.min(), f.max()])
    ax[0, j].plot(ttop + ret, f, "w:", lw=1.2, alpha=0.8)
    ax[0, j].set(title=f"{nom}: barrido dispersado (punteado: curva de DM {dm:.0f})",
                 xlabel="Tiempo (s)", ylabel="Frecuencia (MHz)")
    # fila 2: dedispersado ±250 ms
    r = np.rint(ret * 1000).astype(int); i0 = int(ttop * 1000) - 250
    win = np.stack([np.asarray(d[i0 + rr:i0 + rr + 500, c]) for c, rr in enumerate(r)], 1)
    z = norm(win)
    w = binea(z, 4, 448)
    ax[1, j].imshow(w.T, aspect="auto", origin="upper", cmap="viridis", vmin=-2, vmax=4,
                    extent=[-250, 250, f.min(), f.max()])
    ax[1, j].set(title=f"{nom}: dedispersado a DM {dm:.0f}",
                 xlabel="Tiempo respecto de la llegada (ms)", ylabel="Frecuencia (MHz)")
    # fila 3: perfil
    prof = z.sum(1) / np.sqrt(max(1, (np.abs(z).sum(0) > 0).sum()))
    prof = np.convolve(prof, np.ones(4) / 2, "same")       # boxcar 4 ms, σ = 1
    off = np.r_[prof[:150], prof[350:]]
    snr = (prof - np.median(off)) / (1.4826 * np.median(np.abs(off - np.median(off))))
    ax[2, j].plot(np.arange(-250, 250), snr, lw=1.2, color=plt.cm.viridis(0.3))
    ax[2, j].axhline(0, color="0.7", lw=0.8)
    ax[2, j].set(title=f"{nom}: perfil dedispersado (boxcar 4 ms) · pico {snr[230:280].max():.1f} σ",
                 xlabel="Tiempo respecto de la llegada (ms)", ylabel="S/N (σ)")
    print(nom, "pico", round(float(snr[230:280].max()), 1))
lim = max(a.get_ylim()[1] for a in ax[2]); [a.set_ylim(min(a.get_ylim()[0], -4), lim) for a in ax[2]]
fig.suptitle(titulo + "\nPíxeles en S/N (σ = 1 de ruido): fila 1 de 20 ms × 256 canales, fila 2 de 4 ms × 448 canales", fontsize=14)
fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(out, dpi=100)
print(out)
