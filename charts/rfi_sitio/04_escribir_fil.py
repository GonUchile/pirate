#!/usr/bin/env python3
"""Escribe el baseband del sitio canalizado ×32 e integrado a 1 ms (espectro_dinamico.npy, de 01_) como .fil SIGPROC.

Por cada entrada 0–7 y para la suma de las 8 (suma incoherente de potencias):
  *_f32.fil  potencia |x|² promediada en 1 ms, float32, sin normalizar (unidades de nivel² del 4+4 bits)
  *_8bit.fil calibrado por canal como el 8 bits de CHARTS: y = 32 + 4·(P − mediana)/σ, con mediana y
             σ = 1,4826·MAD de cada canal en los 4,096 s; redondeado y recortado a 0–255
Canales en orden descendente (fch1 = canal más alto, foff < 0), como los .fil del grupo 3; tsamp 1 ms.
Frecuencias: centro = 300,0 + 0,3·c + 0,009375·k MHz (k = −16..15 tras fftshift); se asume que 300,0 MHz es el
centro del canal grueso 0. MJD de inicio tomado de start_time_utc_us del HDF5.
Salida: fil/  (9 archivos × 2 versiones; 352 MB los float32, 88 MB los de 8 bits)
"""
import struct
from pathlib import Path
import numpy as np

AQUI = Path(__file__).resolve().parent
OUT = AQUI / "fil"; OUT.mkdir(exist_ok=True)
D = np.load(AQUI / "espectro_dinamico.npy", mmap_mode="r")
f = np.load(AQUI / "frecuencias_mhz.npy")
START_US = 1788365783335973                         # start_time_utc_us del HDF5
MJD = 40587 + START_US / 1e6 / 86400


def header(nbits, nombre):
    def s(k): return struct.pack("<i", len(k)) + k.encode()
    def i(k, v): return s(k) + struct.pack("<i", v)
    def d(k, v): return s(k) + struct.pack("<d", v)
    return (s("HEADER_START") + s("source_name") + s(nombre) + i("machine_id", 0) + i("telescope_id", 0) +
            i("data_type", 1) + d("fch1", float(f[-1])) + d("foff", -float(f[1] - f[0])) + i("nchans", len(f)) +
            i("nbits", nbits) + i("nifs", 1) + d("tstart", MJD) + d("tsamp", 0.001) + s("HEADER_END"))


def escribe(P, nombre):
    P = P[:, ::-1]                                   # orden descendente
    with open(OUT / f"{nombre}_f32.fil", "wb") as fh:
        fh.write(header(32, nombre)); fh.write(np.ascontiguousarray(P, np.float32).tobytes())
    med = np.median(P, 0); sig = 1.4826 * np.median(np.abs(P - med), 0) + 1e-6
    y = np.clip(np.rint(32 + 4 * (P - med) / sig), 0, 255).astype(np.uint8)
    with open(OUT / f"{nombre}_8bit.fil", "wb") as fh:
        fh.write(header(8, nombre)); fh.write(y.tobytes())
    print(nombre, f"8 bits: {(y == 255).mean() * 100:.2f} % en 255, {(y == 0).mean() * 100:.3f} % en 0", flush=True)


suma = np.zeros(D.shape[1:], np.float64)
for a in range(D.shape[0]):
    P = np.asarray(D[a]); suma += P
    escribe(P, f"sitio_260902_entrada{a}")
escribe(suma.astype(np.float32), "sitio_260902_suma8")
