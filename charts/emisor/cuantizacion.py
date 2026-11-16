"""Paso de 8 bits (.fil, piso 32, σ = 4 cuentas) a los 4 bits + escala/offset que recibe pirate.

Entrada: bloque [T, nchan] uint8 en el orden de canales del .fil (descendente en frecuencia).
Salida:  códigos [nchan, T] uint8 (0..15 = int4 en complemento a 2) con canal 0 = frecuencia más baja,
         y escalas/offsets [nchan, T/256, 2] float16 por minichunk.

Dos modos (iguales a los del emisor original, legado/emisor_fil.py):
  fija      q = floor((x − 32)/2) recortado a [−7, 7]; escala 0,5 σ, offset 0,125 σ.
  por_canal mediana y σ = (p84 − p16)/2 de cada canal en el bloque (con 1 de cada 4 muestras);
            q = rint((x − mediana)/(σ/2)) en [−7, 7]; escala 0,5, offset 0; canales con σ < 1 o mediana > 250
            y muestras en 255 van como faltantes (−8). Es lo que haría el conversor del nodo de búsqueda.
Máscara opcional: canales marcados = −8 y la escala de los útiles × √(N/N_útiles), porque pirate calcula el
S/N suponiendo varianza 1 en todos los canales.

Funciona con numpy o cupy (`xp`).
"""
import numpy as np

_LUT = np.clip(np.floor((np.arange(256) - 32) / 2.0), -7, 7).astype(np.int8)
LUT_FIJA = (_LUT & 0x0F).astype(np.uint8)
FALTANTE = 8                                      # código de −8


def escalas_base(nchan, T, por_canal, xp=np):
    so = xp.zeros((nchan, T // 256, 2), dtype=xp.float16)
    so[..., 0] = 0.5
    so[..., 1] = 0.0 if por_canal else 0.125
    return so


def aplicar_mascara(codigos, so, mascara_asc, xp=np):
    """mascara_asc: bool [nchan] en orden ascendente. Modifica codigos y so en el lugar."""
    if mascara_asc is None:
        return
    m = xp.asarray(mascara_asc)
    codigos[m, :] = FALTANTE
    corr = float(np.sqrt(m.size / max(1, m.size - int(m.sum()))))
    so[~m, :, 0] = so[~m, :, 0] * corr


def fija(bloque, xp=np):
    lut = xp.asarray(LUT_FIJA)
    q = xp.take(lut, xp.asarray(bloque))           # [T, nchan]
    return xp.ascontiguousarray(q[:, ::-1].T)       # [nchan asc, T]


def por_canal(bloque, xp=np, paso=2688):
    g = xp.asarray(bloque)
    T, nchan = g.shape
    q = xp.empty((T, nchan), dtype=xp.uint8)
    for j in range(0, nchan, paso):
        x = g[:, j:j + paso].astype(xp.float32)
        p16, med, p84 = xp.percentile(x[::4], xp.asarray([16.0, 50.0, 84.0], dtype=xp.float32), axis=0)
        sig = (p84 - p16) / 2
        v = xp.clip(xp.rint((x - med) / xp.maximum(sig / 2, 1e-3)), -7, 7).astype(xp.int8)
        malo = (sig < 1) | (med > 250)
        v = xp.where(malo[None, :] | (g[:, j:j + paso] == 255), xp.int8(-8), v)
        q[:, j:j + paso] = (v & 0x0F).astype(xp.uint8)
    return xp.ascontiguousarray(q[:, ::-1].T)
