#!/usr/bin/env python3
"""Tiempo de la detección por bloque a partir del perfil del grouper v1 (opción --perfil → tiempos_grouper.csv).

Separa, por bloque de 2,048 s, el tiempo que el grouper pasa ESPERANDO a pirate del que pasa TRABAJANDO, y desglosa
las etapas. Si la espera es grande, el cuello de botella está antes (emisor o recepción); el tiempo propio de la
detección es pirate (time_dedisperser) + grouper trabajando.

Uso: python evaluacion/perfil_grouper.py CARPETA_CORRIDA [--descartar 5]
"""
import argparse
import csv
import statistics as st
from pathlib import Path

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("corrida", type=Path)
ap.add_argument("--descartar", type=int, default=5, help="bloques del arranque que no se cuentan")
a = ap.parse_args()

filas = list(csv.DictReader(open(a.corrida / "tiempos_grouper.csv")))[a.descartar:-2]
tot = [float(x["total_ms"]) for x in filas]
esp = [float(x["espera_pirate_ms"]) for x in filas]
tra = sorted(t - e for t, e in zip(tot, esp))
print(f"# Perfil del grouper · {a.corrida.name} ({len(filas)} bloques)\n")
print("| | mediana (ms) |\n|---|---|")
print(f"| Período por bloque | {st.median(tot):.0f} |")
print(f"| Esperando a pirate | {st.median(esp):.0f} |")
print(f"| Trabajando | {st.median(tra):.0f} (p90 {tra[int(0.9 * len(tra))]:.0f}) |")
etapas = [k[:-3] for k in filas[0] if k.endswith("_ms") and k not in ("total_ms", "espera_pirate_ms")]
for e in etapas:
    v = [float(x[e + "_ms"]) for x in filas]
    if st.median(v) >= 0.5:
        print(f"| · {e} | {st.median(v):.1f} |")
print(f"\nCandidatos decodificados por bloque: mediana {st.median(float(x['n_cand']) for x in filas):.0f}; "
      f"eventos v1: {st.median(float(x['n_v1']) for x in filas):.0f}")
