#!/usr/bin/env python
"""Calibración del S/N de pirate: pulsos boxcar dispersados de S/N conocido en ruido gaussiano.

S/N verdadero = A * sqrt(W * nchan) para un boxcar de W muestras y amplitud A (en sigmas por
canal y muestra), sumando todos los canales con el retardo exacto del centro de cada canal.
También cuenta candidatos en ruido puro (S/N >= 5).

Uso: python calibrar_snr.py [--nchan 672] [--snr 20] [--dm 500]
"""
import argparse

import numpy as np
import cupy as cp

from pirate_fil_search import BuscadorPirate, CONFIG_DEFAULT, agrupar_nms

K_DM = 4148.808  # s MHz^2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nchan", type=int, default=672)
    ap.add_argument("--snr", type=float, default=20.0)
    ap.add_argument("--dms", default="200,500,1000,1500")
    ap.add_argument("--anchos", default="1,4,16")
    ap.add_argument("--int4", type=float, default=0.0)
    a = ap.parse_args()

    foff = -201.6 / (a.nchan - 1)
    hdr = dict(nchans=a.nchan, fch1=501.6, foff=foff, tsamp=1e-3, nbits=8)
    centros = 501.6 + foff * np.arange(a.nchan)        # orden .fil (descendente)
    rng = np.random.default_rng(1)

    print(f"nchan={a.nchan}  S/N inyectado={a.snr}")
    print(f"{'DM':>6} {'W':>3} {'S/N pirate':>10} {'cociente':>8} {'DM med':>8} {'W med':>6} {'dt (ms)':>8}")
    for dm in [float(x) for x in a.dms.split(",")]:
        for W in [int(x) for x in a.anchos.split(",")]:
            b = BuscadorPirate(CONFIG_DEFAULT, hdr, orden_freq="asc")
            T = b.nt_in
            nchunks = 2 + int(np.ceil(K_DM * dm * (300.0 ** -2 - 501.6 ** -2) / 1e-3 / T)) + 4
            n = nchunks * T
            x = rng.standard_normal((n, a.nchan), dtype=np.float32)
            t0 = 2 * T + 100                                 # llegada a 501,6 MHz (muestra)
            A = a.snr / np.sqrt(W * a.nchan)
            ret = np.rint(K_DM * dm * (centros ** -2 - 501.6 ** -2) / 1e-3).astype(int)
            for c in range(a.nchan):
                x[t0 + ret[c]: t0 + ret[c] + W, c] += A
            xg = cp.asarray(x[:, ::-1].T.copy())             # (nfreq asc, n)
            cands = []
            for i in range(nchunks):
                blk = xg[:, i * T:(i + 1) * T].astype(cp.float16)
                if a.int4:
                    from pirate_fil_search import cuantizar_int4
                    blk = cuantizar_int4(blk, a.int4)
                cands += b.chunk(blk, 5.0)
            c = np.array(cands).reshape(-1, 4)
            c[:, 1] = b.a_t_top(c[:, 0], c[:, 1])
            t_true = t0 * 1e-3 + (W - 1) / 2 * 1e-3
            cerca = c[np.abs(c[:, 1] - t_true) < 0.2]
            if not len(cerca):
                print(f"{dm:6.0f} {W:3d}   no detectado")
                continue
            m = cerca[np.argmax(cerca[:, 2])]
            print(f"{dm:6.0f} {W:3d} {m[2]:10.2f} {m[2] / a.snr:8.3f} {m[0]:8.1f} {m[3]:6.1f} {(m[1] - t_true) * 1e3:8.1f}")
            lejos = agrupar_nms(c[np.abs(c[:, 1] - t_true) > 1.0])
            dur = (n - 2 * T) * 1e-3
            print(f"{'':>10} ruido: {len(lejos)} candidatos S/N>=5 agrupados en {dur:.0f} s "
                  f"(max S/N ruido {lejos[:, 2].max() if len(lejos) else 0:.2f})")


if __name__ == "__main__":
    main()
