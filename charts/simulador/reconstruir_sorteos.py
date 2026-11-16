#!/usr/bin/env python3
"""Reconstruye parametros_sorteados.csv (los sorteos del grupo 2 que reutiliza el grupo 3) desde los manifest.csv del
grupo 3. Los manifiestos guardan todas las columnas que lee 12_generar_set_comparacion_21504.py; t_top_s no se usa
(el generador recalcula la llegada).

Uso: python reconstruir_sorteos.py CARPETA_GRUPO3 SALIDA.csv
"""
import csv
import sys
from pathlib import Path

base, salida = Path(sys.argv[1]), Path(sys.argv[2])
comunes = ["id", "forma_origen", "catalogo_origen", "dm", "width_s", "scat_600_s", "scat_ref_s", "sp_idx", "sp_run",
           "n_canales_emision", "t_top_s"]
filas = []
for r in csv.DictReader(open(base / "A_snr_fijo" / "manifest.csv")):
    filas.append({**{k: r[k] for k in comunes}, "dm_index_negativo": r["dm_index_negativo"], "borde": "", "frac_objetivo": ""})
for r in csv.DictReader(open(base / "parcial" / "manifest.csv")):
    filas.append({**{k: r[k] for k in comunes}, "dm_index_negativo": "", "borde": r["borde"], "frac_objetivo": r["frac_objetivo"]})
with open(salida, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(filas[0])); w.writeheader(); w.writerows(filas)
print(f"{len(filas)} sorteos -> {salida}")
