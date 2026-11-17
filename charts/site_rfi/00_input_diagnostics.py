#!/usr/bin/env python3
"""Diagnóstico rápido del baseband del sitio (260902T161623Z): potencia por entrada y canal grueso.
Lee las primeras 30 000 muestras (0,1 s) del chunk 0 de cada grupo de 84 canales.
Salida: diag_entradas.npz y input_diagnostics.png."""
import os
import sys
from pathlib import Path
import h5py, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

if "CHARTS_BASEBAND_DIR" not in os.environ:
    sys.exit("ERROR: export CHARTS_BASEBAND_DIR=/carpeta/260902T161623Z_CHARTS_hdf5 (baseband del sitio)")
BB = Path(os.environ["CHARTS_BASEBAND_DIR"])
LUT = np.array([v - 16 if v > 7 else v for v in range(16)], np.int8)   # nibble → entero con signo

def decodifica(b):
    """uint8 (4 bits reales altos + 4 imaginarios bajos, complemento a 2) → complex64."""
    return (LUT[b >> 4] + 1j * LUT[b & 15]).astype(np.complex64)

P = np.zeros((32, 672)); Z = np.zeros((32, 672)); K = np.zeros((32, 672)); SAT = np.zeros((32, 672))
for g in range(8):
    with h5py.File(BB / f"bb_f{84*g:04d}_{84*g+84:04d}_chunk00000.h5", "r") as f:
        b = f["baseband"][:, :, :30000]
    x = decodifica(b); p = np.abs(x) ** 2
    s = slice(84 * g, 84 * g + 84)
    P[:, s] = p.mean(-1); Z[:, s] = (b == 0).mean(-1)
    K[:, s] = (p ** 2).mean(-1) / np.maximum(p.mean(-1) ** 2, 1e-9)          # 2 = gaussiano complejo
    SAT[:, s] = ((np.abs(LUT[b >> 4]) >= 7) | (np.abs(LUT[b & 15]) >= 7)).mean(-1)
np.savez("diag_entradas.npz", P=P, Z=Z, K=K, SAT=SAT)
f = 300.0 + 0.3 * np.arange(672)
fig, ax = plt.subplots(2, 2, figsize=(16, 9))
im = ax[0, 0].imshow(10 * np.log10(P + 1e-3), aspect="auto", cmap="viridis", extent=[f[0], f[-1], 31.5, -0.5])
fig.colorbar(im, ax=ax[0, 0], label="potencia (dB, unidades de nivel²)"); ax[0, 0].set(xlabel="MHz", ylabel="entrada", title="Potencia media por entrada y canal")
for i in range(9):
    ax[0, 1].plot(f, 10 * np.log10(P[i] + 1e-3), lw=0.8, label=f"entrada {i}")
ax[0, 1].set(xlabel="MHz", ylabel="dB", title="Espectro entradas 0–8"); ax[0, 1].legend(fontsize=8, ncol=3)
ax[1, 0].plot(np.median(K, 1), "o"); ax[1, 0].axhline(2, color="k", ls="--"); ax[1, 0].set(xlabel="entrada", ylabel="⟨p²⟩/⟨p⟩² (mediana en canales)", title="Curtosis de potencia (2 = ruido gaussiano)")
ax[1, 1].plot(np.median(Z, 1) * 100, "o", label="% bytes = 0"); ax[1, 1].plot(SAT.mean(1) * 100, "s", label="% saturado (|x| = 7, 8)")
ax[1, 1].set(xlabel="entrada", ylabel="%", title="Cuantización 4+4"); ax[1, 1].legend()
fig.tight_layout(); fig.savefig("input_diagnostics.png", dpi=110)
for i in range(32):
    print(i, f"P med {np.median(P[i]):.2f}  rms/comp {np.sqrt(np.median(P[i])/2):.2f}  K {np.median(K[i]):.2f}  ceros {np.median(Z[i])*100:.0f}%  sat {SAT[i].mean()*100:.3f}%")
