#!/usr/bin/env python3
"""Para cada archivo positivo de una prueba de 64 beams, los eventos de pirate más cercanos (para clasificar pérdidas).

Usa el mismo parseo de sifter.log y la misma conversión de tiempos que evaluar_prueba_64b.py. Por archivo guarda:
  snr_dm_ok, dm_dm_ok      mejor evento en la ventana de tiempo con DM ±10 % (cualquier S/N ≥ 5 que entregó el grouper)
  snr_otro_dm, dm_otro_dm  mejor evento en la ventana de tiempo con DM fuera de ±10 %
Ventana de tiempo: |Δt| ≤ max(0,1 s, duración del burst) respecto de la llegada esperada a 501,6 MHz.

Uso: environments/pirate/bin/python eventos_cercanos_64b.py tiempo_real/ETIQUETA
Salida: ETIQUETA/analisis_por_parametro/eventos_cercanos.csv (lo lee analisis_perdidas_64b.py)
"""
import csv
import sys
from pathlib import Path

import numpy as np

AQUI = Path(__file__).resolve().parent
D = Path(sys.argv[1]).resolve()
src = (AQUI / "evaluar_prueba_64b.py").read_text()
src = src[:src.index("# mejor S/N de un evento")]          # parseo de eventos y de la verdad, sin evaluar
sys.argv = [str(AQUI / "evaluar_prueba_64b.py"), str(D)]
g = {"__file__": str(AQUI / "evaluar_prueba_64b.py"), "__name__": "eventos_cercanos"}
exec(compile(src, "evaluar_prueba_64b.py", "exec"), g)
ev_beam, ev_dm, ev_snr, ev_t, filas, TOL_DM = (g[k] for k in ("ev_beam", "ev_dm", "ev_snr", "ev_t", "filas", "TOL_DM"))

OUT = D / "analisis_por_parametro"
OUT.mkdir(exist_ok=True)
with open(OUT / "eventos_cercanos.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["name", "snr_dm_ok", "dm_dm_ok", "snr_otro_dm", "dm_otro_dm"])
    for r in filas:
        if r["set"] == "NEG":
            continue
        ok_t = (ev_beam == r["b"]) & (np.abs(ev_t - r["t_abs"]) <= max(0.1, r["box"]))
        ok_dm = np.abs(ev_dm - r["dmv"]) <= max(TOL_DM * r["dmv"], 1.0)
        fila = [r["name"]]
        for m in (ok_t & ok_dm, ok_t & ~ok_dm):
            if m.any():
                k = np.argmax(np.where(m, ev_snr, -np.inf))
                fila += [f"{ev_snr[k]:.2f}", f"{ev_dm[k]:.1f}"]
            else:
                fila += ["", ""]
        w.writerow(fila)
print("listo", OUT / "eventos_cercanos.csv")
