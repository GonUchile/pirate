#!/usr/bin/env python3
"""Resumen de recursos por proceso (archivo de analisis_rfi/monitor_procesos.py): memoria y uso de GPU, RAM y CPU
del servidor de pirate, el grouper, el sifter y el emisor, mientras el emisor está enviando.

Uso: python evaluacion/resumen_recursos.py CARPETA_CORRIDA/recursos_por_proceso.csv
Notas: el uso de GPU de `nvidia-smi pmon` es una muestra de ~1 s cada 5 s y subestima procesos que trabajan en
ráfagas (pirate); la RAM del emisor incluye los .fil mapeados desde la caché de disco.
"""
import csv
import math
import statistics as st
import sys

filas = list(csv.DictReader(open(sys.argv[1])))
con_emisor = {r["t"] for r in filas if r["rol"] == "emisor"} or {r["t"] for r in filas}
print("| Proceso | GPU memoria máx (GiB) | GPU uso medio (%) | RAM máx (GiB) | CPU media (%) | CPU p90 (%) |")
print("|---|---|---|---|---|---|")
for rol in ("pirate_server", "grouper", "sifter", "emisor"):
    r = [x for x in filas if x["rol"] == rol and x["t"] in con_emisor]
    if not r:
        continue
    cpu = sorted(float(x["cpu_pct"]) for x in r if not math.isnan(float(x["cpu_pct"])))
    print(f"| {rol} | {max(float(x['gpu_mem_mib']) for x in r) / 1024:.2f} | {st.mean(float(x['gpu_sm_pct']) for x in r):.1f} | "
          f"{max(float(x['rss_gib']) for x in r):.1f} | {st.mean(cpu):.0f} | {cpu[int(0.9 * len(cpu))]:.0f} |")
