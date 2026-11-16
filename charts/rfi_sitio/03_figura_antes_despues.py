#!/usr/bin/env python3
"""Figura sin RFI vs con RFI del sitio para un archivo del grupo 3 (mismo formato que fil_prueba_rfi/B_277_sin_vs_con_rfi.png).
Uso: python 03_figura_antes_despues.py ORIGINAL.fil CON_RFI.fil DM T_TOP_S salida.png [titulo]"""
import sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parent))
from sigproc import leer_header  # noqa: E402

orig, conr, dm, ttop, out = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4]), sys.argv[5]
titulo = sys.argv[6] if len(sys.argv) > 6 else Path(orig).stem


def carga(p):
    head, h = leer_header(p)
    nch = h["nchans"]; n = (Path(p).stat().st_size - len(head)) // nch
    return np.memmap(p, np.uint8, "r", offset=len(head), shape=(n, nch)), h


fig, ax = plt.subplots(3, 2, figsize=(16, 13), gridspec_kw=dict(height_ratios=[1, 1, 0.6]))
for j, (p, nom) in enumerate(((orig, "Sin RFI"), (conr, "Con RFI del sitio (líneas > 10 dB visibles, resto calibrado)"))):
    d, h = carga(p)
    n, nch = d.shape
    f = h["fch1"] + h["foff"] * np.arange(nch)
    # archivo completo: 100 ms × 64 canales
    nt = n // 100 * 100
    w = np.asarray(d[:nt]).astype(np.float32).reshape(nt // 100, 100, nch // 64, 64).mean((1, 3))
    a = ax[0, j]
    a.imshow(w.T, aspect="auto", origin="upper", cmap="viridis", extent=[0, nt / 1000, f.min(), f.max()], vmin=30, vmax=40)
    a.axvline(ttop, color="w", ls=":", lw=1)
    a.set(title=f"{nom}: archivo completo (100 ms × 64 canales)", xlabel="Tiempo (s)", ylabel="Frecuencia (MHz)")
    # dedispersado
    ret = np.rint(4.148808e6 * dm * (f ** -2 - f.max() ** -2)).astype(int)
    i0 = int(ttop * 1000) - 1500
    win = np.stack([np.asarray(d[i0 + r:i0 + r + 3000, c]) for c, r in enumerate(ret)], 1).astype(np.float32)  # (3000, nch)
    sat = (win == 255).mean(0) > 0.5
    med = np.median(win, 0); mad = 1.4826 * np.median(np.abs(win - med), 0) + 1e-3
    z = (win - med) / mad; z[:, sat] = 0
    zg = z.reshape(300, 10, nch // 64, 64).mean((1, 3))
    a = ax[1, j]
    a.imshow(zg.T, aspect="auto", origin="upper", cmap="viridis", extent=[-1500, 1500, f.min(), f.max()], vmin=-1, vmax=2)
    a.set(title=f"Dedispersado a DM {dm:.0f} (mediana restada, normalizado; canales saturados en 0)",
          xlabel="Tiempo respecto de la llegada (ms)", ylabel="Frecuencia (MHz)")
    prof = z.sum(1); prof = np.convolve(prof, np.ones(10) / 10, "same")
    off = np.r_[prof[:1200], prof[1800:]]
    snr = (prof - np.median(off)) / (1.4826 * np.median(np.abs(off - np.median(off))))
    a = ax[2, j]
    a.plot(np.arange(-1500, 1500), snr, lw=1)
    a.set(title=f"Suma de canales (boxcar 10 ms) · {sat.sum()} canales saturados excluidos", xlabel="Tiempo respecto de la llegada (ms)", ylabel="SNR (σ)")
    print(nom, "frac 255:", float((np.asarray(d[::50]) == 255).mean()), "pico SNR:", float(snr[1450:1600].max()))
fig.suptitle(titulo, fontsize=15)
fig.tight_layout(); fig.savefig(out, dpi=100)
print(out)
