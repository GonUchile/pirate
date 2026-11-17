#!/usr/bin/env python3
"""Espectro dinámico del baseband del sitio a la resolución de las simulaciones del grupo 3.

Entrada: $CHARTS_BASEBAND_DIR = 260902T161623Z_CHARTS_hdf5 (32 entradas × 672 canales de 0,3 MHz ×
1 228 800 muestras de 3,333 µs = 4,096 s; uint8 = 4 bits reales altos + 4 imaginarios bajos, complemento a 2).
Proceso por entrada (0–7, las únicas con señal; ver 00_input_diagnostics.py):
  FFT de 32 puntos sin ventana por canal grueso → 21 504 canales de 9,375 kHz cada 106,67 µs,
  |x|², promedio en bins de 1 ms exactos (9,375 cuadros por bin, con reparto fraccional en los bordes).
Orden de canales: frecuencia ascendente; canal fino j = 32·c + (k + 16), con k = −16..15 el índice tras fftshift,
centro = 300,0 + 0,3·c + 0,009375·k MHz (se asume que freq_start_MHz = 300,0 es el centro del canal grueso 0).
Salida: espectro_dinamico.npy, float32 (8 entradas, 4096 ms, 21504 canales) = 2,8 GB, y frequencies_mhz.npy.
Uso: python 01_dynamic_spectrum.py
"""
from multiprocessing import Pool
import os
import sys
from pathlib import Path
import h5py, numpy as np

if "CHARTS_BASEBAND_DIR" not in os.environ:
    sys.exit("ERROR: export CHARTS_BASEBAND_DIR=/carpeta/260902T161623Z_CHARTS_hdf5 (baseband del sitio)")
BB = Path(os.environ["CHARTS_BASEBAND_DIR"])
AQUI = Path(__file__).resolve().parent
ENTRADAS = 8
NFFT, NCH_G, NT = 32, 84, 1228800
NFR = NT // NFFT                     # 38 400 cuadros de 106,67 µs
NMS = 4096                           # bins de 1 ms
FR_POR_MS = NFR / NMS                # 9,375
LUT = np.array([v - 16 if v > 7 else v for v in range(16)], np.float32)
SALIDA = AQUI / "espectro_dinamico.npy"


def grupo(g):
    resto = None
    pot = []
    for ch in range(7):
        with h5py.File(BB / f"bb_f{NCH_G*g:04d}_{NCH_G*g+NCH_G:04d}_chunk{ch:05d}.h5", "r") as f:
            b = f["baseband"][:ENTRADAS]                                  # (8, 84, n)
        x = (LUT[b >> 4] + 1j * LUT[b & 15]).astype(np.complex64)
        if resto is not None:
            x = np.concatenate([resto, x], axis=-1)
        n = x.shape[-1] // NFFT * NFFT
        resto = x[..., n:]
        y = np.fft.fftshift(np.fft.fft(x[..., :n].reshape(ENTRADAS, NCH_G, -1, NFFT), axis=-1), axes=-1)
        pot.append((np.abs(y) ** 2).astype(np.float32))                    # (8, 84, nfr, 32)
    p = np.concatenate(pot, axis=2)
    assert p.shape[2] == NFR, p.shape
    # promedio en bins de 1 ms con reparto fraccional: integral acumulada interpolada en los bordes
    c = np.concatenate([np.zeros_like(p[:, :, :1]), np.cumsum(p, axis=2, dtype=np.float64)], axis=2)
    bordes = np.arange(NMS + 1) * FR_POR_MS
    i0 = np.floor(bordes).astype(int); fr = bordes - i0
    i1 = np.minimum(i0 + 1, NFR)
    cb = c[:, :, i0] * (1 - fr)[None, None, :, None] + c[:, :, i1] * fr[None, None, :, None]
    ms = (np.diff(cb, axis=2) / FR_POR_MS).astype(np.float32)           # (8, 84, 4096, 32)
    out = np.lib.format.open_memmap(SALIDA, mode="r+")
    out[:, :, g * NCH_G * NFFT:(g + 1) * NCH_G * NFFT] = ms.transpose(0, 2, 1, 3).reshape(ENTRADAS, NMS, -1)
    out.flush()
    return g


if __name__ == "__main__":
    np.lib.format.open_memmap(SALIDA, mode="w+", dtype=np.float32, shape=(ENTRADAS, NMS, 672 * NFFT))
    c = np.arange(672); k = np.arange(-16, 16)
    np.save(AQUI / "frequencies_mhz.npy", (300.0 + 0.3 * c[:, None] + 0.3 / NFFT * k[None, :]).ravel())
    with Pool(8) as pool:
        for g in pool.imap_unordered(grupo, range(8)):
            print("grupo", g, "listo", flush=True)
    print(SALIDA)
