#!/usr/bin/env python3
"""Eficiencia teórica de S/N de cada cuantización (señal débil en ruido gaussiano).

Q(x) = valor reconstruido de la muestra x ~ N(0,1) (en sigma). Para una señal débil s:
  E[Q(x+s)] - E[Q(x)] ~ s * sum_k J_k phi(t_k)      (t_k umbrales, J_k salto de Q en t_k)
  eta = sum_k J_k phi(t_k) / sqrt(Var Q(x))          fracción de S/N que se conserva.
Se calcula exacto a partir de los umbrales del cuantizador de 8 bits del set
(nivel L = floor(32,5 + 4x), recorte 0-255) y de cada mapeo a 4 bits desde L.
"""
import numpy as np
from scipy.stats import norm

L = np.arange(256)
# 8 bits: L = floor(32.5 + 4x)  ->  umbral entre L-1 y L en x = (L - 32.5)/4
t8 = (L[1:] - 32.5) / 4.0
p8 = np.diff(np.r_[0.0, norm.cdf(t8), 1.0])              # P(L)


def eta_desde_8bits(g):
    """g: valor reconstruido (en sigma) para cada nivel L de 8 bits."""
    g = np.asarray(g, float)
    m = (p8 * g).sum()
    var = (p8 * (g - m) ** 2).sum()
    gan = (np.diff(g) * norm.pdf(t8)).sum()
    return gan / np.sqrt(var), var


def eta_directo(paso, nmax):
    """Cuantizador uniforme centrado aplicado al float: q = clip(rint(x/paso), -nmax, nmax)."""
    k = np.arange(-nmax, nmax + 1)
    t = (k[:-1] + 0.5) * paso
    p = np.diff(np.r_[0.0, norm.cdf(t), 1.0])
    v = k * paso
    var = (p * v ** 2).sum() - (p * v).sum() ** 2
    return (paso * norm.pdf(t)).sum() / np.sqrt(var)


filas = [("8 bits (set: 4 niveles/σ, piso 32)", eta_desde_8bits((L - 32) / 4.0)[0])]
for n, nom in ((2, "4b 2niv fija"), (1, "4b 1niv fija"), (3, "4b 3niv fija")):
    filas.append((f"{nom} (desde 8 bits)", eta_desde_8bits(np.clip(np.rint((L - 32) / 4.0 * n), -7, 7) / n)[0]))
filas.append(("4b 2niv desde float (sin pasar por 8 bits)", eta_directo(0.5, 7)))
print("Eficiencia teórica de S/N (señal débil), respecto al dato sin cuantizar:")
for nom, e in filas:
    print(f"  {nom:45s} {e:.4f}   pérdida {100 * (1 - e):.2f} %")

print("\nReferencia (cuantizador uniforme óptimo, señal débil):")
for bits in (1, 2, 3, 4):
    nmax = 2 ** (bits - 1) - 1 if bits > 1 else None
    if bits == 1:
        print(f"  1 bit: {np.sqrt(2 / np.pi):.4f}")
        continue
    pasos = np.linspace(0.05, 2.0, 2000)
    # con 2^bits - 1 niveles simétricos (como int4 con -8 reservado)
    e = [eta_directo(p, nmax) for p in pasos]
    i = int(np.argmax(e))
    print(f"  {bits} bits ({2 * nmax + 1} niveles): η máx {e[i]:.4f} con paso {pasos[i]:.3f} σ "
          f"({1 / pasos[i]:.2f} niveles/σ)")

print("\npirate (escala/offset por canal y bloque de 256 muestras):")
print("  - el offset se estima con la media del bloque: un pulso de ancho w (muestras) dentro de un")
print("    bloque pierde ~w/256 de su amplitud; a 21 504 canales el pulso por canal dura ~ancho intrínseco + scattering.")
for w in (1, 4, 16, 32, 64, 128):
    print(f"    w = {w:4d} ms -> S/N x {1 - w / 256:.3f}")
print("  - el error de la media (sigma/sqrt(256)) es un escalón de 256 ms por canal: agrega ~1/256 de varianza (despreciable).")
