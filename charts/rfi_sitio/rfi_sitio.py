"""RFI medida en el sitio (baseband 2026-09-02) para agregar a las simulaciones del grupo 3 (21 504 canales, 1 ms, 8 bits).

El modelo (modelo_rfi_sitio.npz, de 02_modelo_rfi.py) da la RFI en unidades de la potencia del sistema P_sys:
  r_p(f)    persistente (mediana en 4,096 s)
  e(t,f)    rápida (exceso > 5σ sobre r_p, con vecinos > 3σ), 4096 ms × 21 504 canales
En el .fil la potencia del sistema vale P_sys = σ·√(Δν τ) = 4·√9,375 = 12,25 cuentas (piso 32, σ = 4 cuentas).

Para cada píxel del .fil, con s = x − 32 (señal + ruido de σ0 = 4), g = 10^(−atenuacion_db/10),
rp = g·r_p(f) y re = g·e(t,f):
  canal "visible" (línea con 10 log10(1 + rp) > umbral_lineas_db; el 8 bits no la aplana):
      y = 32 + s + P_sys·(rp + re) + σ0·√((1 + rp + re)² − 1)·N(0,1)
  canal "calibrado" (el offset/escala por canal del 8 bits normaliza la potencia persistente):
      y = 32 + [s + P_sys·re + σ0·√((1 + rp + re)² − 1)·N(0,1)] / (1 + rp)
El término con N(0,1) agrega el ruido radiométrico de la potencia de RFI (σ crece con la potencia total).
En los canales calibrados el ruido queda en σ0 donde no hay RFI rápida y la señal del FRB se divide por (1 + rp).
Se redondea y se recorta a 0–255, como el 8 bits real.

La medición dura 4,096 s. Para archivos más largos se usa un bootstrap de bloques contiguos: cada bloque de
`bloque_ms` toma un desfase aleatorio dentro de la captura (circular). En tiempo real todos los beams usan el
mismo reloj (la RFI entra por los lóbulos laterales y llega a todos los beams a la vez); `ganancia_beam_db`
permite que cada beam la vea con distinta ganancia de lóbulo lateral.

Funciona con numpy o cupy (xp).
"""
import os
from pathlib import Path

import numpy as np

MODELO = Path(os.environ.get("CHARTS_RFI_MODELO", Path(__file__).resolve().parent / "modelo_rfi_sitio.npz"))
PISO, SIGMA0 = 32.0, 4.0


class RfiSitio:
    def __init__(self, frecuencias_fil_mhz, umbral_lineas_db=10.0, atenuacion_db=0.0, rapida=True, persistente=True,
                 bloque_ms=4096, semilla=0, modelo=MODELO, xp=np):
        m = np.load(modelo)
        self.xp = xp
        fm = m["frecuencias_mhz"]
        # canal del .fil -> canal medido más cercano en frecuencia (las grillas difieren en < 0,2 %)
        idx = np.clip(np.searchsorted(fm, frecuencias_fil_mhz), 1, len(fm) - 1)
        idx = np.where(np.abs(fm[idx - 1] - frecuencias_fil_mhz) < np.abs(fm[idx] - frecuencias_fil_mhz), idx - 1, idx)
        g = 10 ** (-atenuacion_db / 10)
        rp = np.clip(m["r_p"][idx], 0, None) * g if persistente else np.zeros(len(idx), np.float32)
        visible = 10 * np.log10(1 + rp) > umbral_lineas_db
        self.psys = float(m["psys_cuentas"])
        self.rp = xp.asarray(rp.astype(np.float32))
        self.visible = xp.asarray(visible)
        e = m["e"][:, idx] if rapida else np.zeros((len(m["e"]), len(idx)), np.float16)
        self.e = xp.asarray((e.astype(np.float32) * g).astype(np.float16))   # float16: 176 MB, se comparte entre hilos
        self.nt = self.e.shape[0]
        self.bloque = int(bloque_ms)
        self.semilla = int(semilla)
        self.n_visibles = int(visible.sum())
        self.perdida_snr_mediana = float(np.median(np.where(visible, 1.0, 1 / (1 + rp))))

    def _filas(self, t0, n):
        """Índices de tiempo en la captura para las muestras t0..t0+n−1 (bootstrap de bloques contiguos)."""
        t = np.arange(t0, t0 + n)
        b = t // self.bloque
        d = {k: int(np.random.default_rng([self.semilla, int(k)]).integers(self.nt)) for k in np.unique(b)}
        return (t % self.bloque + np.array([d[k] for k in b])) % self.nt     # determinista: seguro entre hilos

    def desfase_inicial(self):
        return int(np.random.default_rng([self.semilla, 0]).integers(self.nt))

    def aplicar(self, bloque_u8, t0, ganancia=1.0, rng=None):
        """bloque_u8: (n, nchan) uint8 en el orden de canales del .fil; t0: muestra (ms) de la primera fila.
        ganancia: factor extra de este beam (lóbulo lateral). rng: generador para el ruido de la RFI
        (numpy Generator o cupy RandomState; uno por hilo). Devuelve uint8 con la RFI agregada."""
        xp = self.xp
        if rng is None:
            rng = np.random.default_rng([self.semilla, t0, 1]) if xp is np else xp.random.RandomState(self.semilla + t0)
        b = xp.asarray(bloque_u8)
        out = xp.empty(b.shape, xp.uint8)
        filas = xp.asarray(self._filas(t0, b.shape[0]))
        rp = self.rp * ganancia
        div = xp.where(self.visible, 1.0, 1.0 / (1 + rp)).astype(xp.float32)
        paso = 4096 if xp is np else 256          # en GPU, temporales chicos: varios hilos comparten la memoria con pirate
        for i in range(0, b.shape[0], paso):
            x = b[i:i + paso].astype(xp.float32) - PISO
            re = self.e[filas[i:i + paso]].astype(xp.float32) * ganancia
            tot = 1 + rp[None, :] + re
            y = x + self.psys * re + SIGMA0 * xp.sqrt(xp.maximum(tot * tot - 1, 0)) * rng.standard_normal(x.shape, dtype=xp.float32)
            y = xp.where(self.visible[None, :], y + self.psys * rp[None, :], y * div[None, :])
            out[i:i + paso] = xp.clip(xp.rint(y + PISO), 0, 255).astype(xp.uint8)
        return out
