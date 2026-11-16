"""Fuentes de datos para el emisor: de dónde sale el bloque de 8 bits de cada beam.

- Set de .fil (grupo 3 del simulador): los archivos se barajan y se reparten uno tras otro en los streams de los
  beams con datos, con ruido antes (prepad) y después (cola). `asignacion.csv` guarda qué archivo va en qué beam y
  desde qué muestra: es la verdad para evaluar.
- Ruido: huecos entre archivos y relleno, ruido gaussiano con la escala del set (piso 32, σ = 4 cuentas).
- Relleno precalculado (beams sin datos): 8 bloques de ruido ya convertidos a 4 bits que se repiten rotando;
  cuesta solo copiar memoria. Sirve para probar muchos beams sin que el emisor sea el límite.

Para conectar otra fuente (por ejemplo datos en vivo), basta con una clase con el método
`bloque(t0, T, out) -> lista de tramos de ruido` que llene `out` [T, nchan] uint8 en el orden del .fil.
"""
import csv
import random
from pathlib import Path

import numpy as np

CARPETAS = {"A": "A_snr_fijo", "B": "B_amplitud_fija", "NEG": "negativos", "P": "parcial"}


def largo_header(path):
    with open(path, "rb") as f:
        raw = f.read(8192)
    return raw.index(b"HEADER_END") + len(b"HEADER_END")


def armar_streams(set_dir, sets, nbeams, prepad, cola, semilla, limite=None, T=2048):
    """Reparte los archivos de los sets en nbeams streams. Devuelve (tramos por beam, asignación, muestras totales)."""
    filas = []
    for s in sets:
        with open(Path(set_dir) / CARPETAS[s] / "manifest.csv", newline="") as f:
            m = list(csv.DictReader(f))
        filas += m[:limite] if limite else m
    random.Random(semilla).shuffle(filas)
    largo = [prepad] * nbeams
    tramos = [[("ruido", prepad)] if prepad else [] for _ in range(nbeams)]
    asign = []
    for r in filas:
        b = int(np.argmin(largo))
        n = int(r["nsamp"])
        asign.append({"beam": b, "inicio_muestra": largo[b], **r})
        tramos[b].append(("fil", r["fil_path"], n))
        largo[b] += n
    total = ((max(largo) + cola + T - 1) // T) * T
    for b in range(nbeams):
        if total > largo[b]:
            tramos[b].append(("ruido", total - largo[b]))
    return tramos, asign, total


def ruido_8bits(T, nchan, n=8, semilla=99):
    """n bloques [T, nchan] uint8 de ruido gaussiano con la escala del set (piso 32, σ = 4 cuentas)."""
    rng = np.random.default_rng(semilla)
    return np.stack([np.clip(np.floor(rng.standard_normal((T, nchan), dtype=np.float32) * 4 + 32.5), 0, 255).astype(np.uint8)
                     for _ in range(n)])


class LectorBeam:
    """Bloque [a, a+T) del stream de un beam como uint8 [T, nchan] en el orden del .fil.

    Los huecos de ruido se llenan copiando de `pool` (bloques de ruido precalculados, ver ruido_8bits) si se da;
    si no, se genera ruido nuevo en cada bloque (unas 5 veces más lento en la CPU)."""

    def __init__(self, tramos, nchan, semilla, pool=None, beam=0):
        self.nchan = nchan
        self.pool, self.beam = pool, beam
        self.rng = np.random.default_rng(semilla)
        self.tabla, t = [], 0
        for tr in tramos:
            n = tr[1] if tr[0] == "ruido" else tr[2]
            self.tabla.append((t, t + n, tr[0], tr[1] if tr[0] == "fil" else None))
            t += n
        self.mm = {}

    def _mm(self, path):
        if path not in self.mm:
            self.mm.clear()                         # un archivo abierto a la vez por beam
            d = np.memmap(path, dtype=np.uint8, mode="r", offset=largo_header(path))
            self.mm[path] = d.reshape(-1, self.nchan)
        return self.mm[path]

    def bloque(self, a, T, out):
        """Llena out con los tramos de archivo y genera el ruido de los huecos. Devuelve los tramos de ruido."""
        b, ruido = a + T, []
        for (i0, i1, tipo, path) in self.tabla:
            lo, hi = max(a, i0), min(b, i1)
            if hi <= lo:
                continue
            if tipo == "fil":
                out[lo - a:hi - a] = self._mm(path)[lo - i0:hi - i0]
            else:
                if self.pool is not None:
                    # canales rotados según el beam: sin esto, los beams b, b+8, ... llevan el mismo ruido al mismo
                    # tiempo y dan los mismos eventos de ruido (coincidencias falsas entre beams)
                    blq, s = self.pool[(a // T + self.beam) % len(self.pool)], (self.beam * 2687) % self.nchan
                    out[lo - a:hi - a, :self.nchan - s] = blq[lo - a:hi - a, s:]
                    out[lo - a:hi - a, self.nchan - s:] = blq[lo - a:hi - a, :s]
                else:
                    x = self.rng.standard_normal((hi - lo, self.nchan), dtype=np.float32)
                    out[lo - a:hi - a] = np.clip(np.floor(x * 4 + 32.5), 0, 255).astype(np.uint8)
                ruido.append((lo - a, hi - a))
        return ruido


def relleno_precalculado(nchan, T, n=8, ceros=False, semilla=4242):
    """n bloques empaquetados [nchan, T/2] uint8 (cuantización fija) para los beams sin datos."""
    from cuantizacion import fija
    from protocolo import empaquetar_int4
    out = np.zeros((n, nchan, T // 2), np.uint8)
    if not ceros:
        rng = np.random.default_rng(semilla)
        for k in range(n):
            x = rng.standard_normal((T, nchan), dtype=np.float32)
            out[k] = empaquetar_int4(fija(np.clip(np.floor(x * 4 + 32.5), 0, 255).astype(np.uint8)))
    return out
