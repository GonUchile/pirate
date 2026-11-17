#!/usr/bin/env python3
"""Máscaras estáticas de canales a partir de la RFI conocida del sitio (site_rfi_model.npz).

  m1: líneas persistentes > 3 dB  o  RFI rápida > 1 % del tiempo      (17,5 % de la banda)
  m2: líneas persistentes > 1 dB  o  RFI rápida > 0,2 % del tiempo    (39 % de la banda)
Salida: mask_m1.npy, mask_m2.npy (bool, 21 504 canales, orden de frequencies_mhz.npy, ascendente).
Uso en tiempo real: emisor_fil.py ... --mascara site_rfi/mask_m1.npy
"""
from pathlib import Path
import numpy as np
AQUI = Path(__file__).resolve().parent
m = np.load(AQUI / "site_rfi_model.npz")
db = 10 * np.log10(1 + np.clip(m["r_p"], 0, None)); ocup = (m["e"] > 0).mean(0)
for nom, mk in (("m1", (db > 3) | (ocup > 0.01)), ("m2", (db > 1) | (ocup > 0.002))):
    np.save(AQUI / f"mascara_{nom}.npy", mk)
    print(nom, int(mk.sum()), f"{mk.mean() * 100:.1f} %")
