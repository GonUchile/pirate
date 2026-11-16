#!/usr/bin/env python3
"""Completitud a tasas de falsas alarmas estándar, con los mismos criterios que evaluar_prueba_64b.py.

Reporta, para una corrida:
  - FA a S/N ≥ 6 (falsas alarmas de ruido por beam·hora);
  - umbral y completitud A / B / cortados y FP en negativos a 10 FA por beam·hora;
  - lo mismo a 10 FA por hora del SISTEMA (10 / nbeams por beam·hora). Si el tiempo sin señal no alcanza para
    resolver esa tasa, se usa el primer umbral sin ningún evento de ruido y se avisa.

Uso: python evaluacion/tasas_estandar.py CARPETA_CORRIDA [--log eventos_todos.log]
     (por defecto usa sifter.log; con --log se evalúa otro registro con el mismo formato, p. ej. el completo del grouper)
"""
import argparse
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

AQUI = Path(__file__).resolve().parent

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("corrida", type=Path)
ap.add_argument("--log", default="sifter.log")
a = ap.parse_args()

# se reutiliza la lógica del evaluador (lectura de eventos, ventanas de señal, aciertos) sobre una carpeta temporal
D = a.corrida.resolve()
tmp = Path(tempfile.mkdtemp())
for f in ("asignacion.csv", "emisor.json", "server.log", "gpu.csv", "vmstat.log", "config_dedispersion.yml"):
    if (D / f).exists():
        (tmp / f).symlink_to(D / f)
(tmp / "sifter.log").symlink_to(D / a.log)
src = (AQUI / "evaluar_prueba_64b.py").read_text()
src = src[:src.index("def tabla_md")]
sys.argv = [str(AQUI / "evaluar_prueba_64b.py"), str(tmp)]
g = {"__name__": "tasas", "__file__": str(AQUI / "evaluar_prueba_64b.py")}
exec(src, g)
shutil.rmtree(tmp)

um, fp, fila, horas, nbeams = g["umbrales"], g["fp_curve"], g["fila"], g["horas"], g["nbeams"]
print(f"# Tasas estándar · {D.name} ({a.log})\n")
print(f"Tiempo sin señal: {horas:.1f} beam·hora (resolución de la tasa: {1 / horas:.2f} FA por beam·hora)\n")
print(f"FA a S/N ≥ 6: {fila(6.0)['FP ruido/h']:.0f} por beam·hora\n")
print("| Objetivo | Umbral S/N | FA medidas (por beam·hora) | Completitud A | B | cortados | FP en negativos |")
print("|---|---|---|---|---|---|---|")
for obj, nom in ((10.0, "10 por beam·hora"), (10.0 / nbeams, f"10/h del sistema de {nbeams} beams")):
    k = np.flatnonzero(fp <= obj)
    if not len(k):
        print(f"| {nom} | no se alcanza | | | | | |"); continue
    u = float(um[k[0]]); f = fila(u)
    nota = " (sin eventos de ruido: cota)" if fp[k[0]] == 0 and obj < 1 / horas else ""
    print(f"| {nom} | {u:.2f} | {fp[k[0]]:.2f}{nota} | {f['Recall A DM≤1000 (%)']} % | {f['Recall B DM≤1000 (%)']} % | "
          f"{f['Recall cortados DM≤1000 (%)']} % | {f['FP no-FRB (% NEG DM≤1000)']} % |")
