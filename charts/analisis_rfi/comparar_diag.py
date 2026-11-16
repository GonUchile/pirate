#!/usr/bin/env python3
"""Diagnóstico de la agrupación del grouper v1 con la RFI del sitio: compara corridas cortas con RFI que cambian un
parámetro del grouper, usando TODOS los eventos (eventos_todos.log, sin topes --k/--k-total).

  corta_todos_rfi  referencia (tol-t 0,1 s, umbral-celda 5)
  diag_tol         --tol-t 0.002 (casi sin descarte por t_top)
  diag_celda       --umbral-celda 7 (regiones solo con celdas de S/N ≥ 7)
  corta_todos_sin_rfi  control sin RFI

Para cada una: eventos totales, por beam y chunk, Recall/FP (evaluar_prueba_64b.py sobre <corrida>/todos/) y el mejor
S/N encontrado para cada FRB de A, B y cortados con DM ≤ 1000. Salida: resultados/diag_agrupacion.md
"""
import csv
import re
import subprocess
import os
from pathlib import Path

import numpy as np

if "CHARTS_RESULTADOS_RFI" not in os.environ:
    sys.exit("ERROR: export CHARTS_RESULTADOS_RFI=/carpeta/con/resultados/<corrida> (las corridas a analizar)")
R = Path(os.environ["CHARTS_RESULTADOS_RFI"])  # carpeta con resultados/<corrida>
PC = Path(__file__).resolve().parent.parent        # raíz del repositorio (evaluar_prueba_64b.py)
PY = sys.executable   # correr este script con el Python del entorno de pirate
import sys
CORRIDAS = [("corta_todos_sin_rfi", "sin RFI (control)"), ("corta_todos_rfi", "RFI, referencia"),
            ("diag_tol", "RFI, --tol-t 0.002"), ("diag_celda", "RFI, --umbral-celda 7"),
            ("mask_m1_sin_rfi", "sin RFI + máscara m1"), ("mask_m2_sin_rfi", "sin RFI + máscara m2"),
            ("mask_m1_rfi", "RFI + máscara m1"), ("mask_m2_rfi", "RFI + máscara m2")]
SALIDA = sys.argv[1] if len(sys.argv) > 1 else "diag_agrupacion.md"
EV = re.compile(r"beam_id=(\d+), dm=([\d.e+-]+), fpga_timestamp=(-?\d+)")
K = 4.148808e3 * (299.8497764530551 ** -2 - 501.6 ** -2)


def azar(e, filas):
    """Aciertos en el tiempo real y por azar (tiempos desplazados), con los criterios del evaluador."""
    t = e[:, 2] / 200e3 - e[:, 1] * K
    def ok(r, off):
        b, dm = int(r["beam"]), float(r["dm"])
        tt = float(r["inicio_muestra"]) / 1000 + float(r["t_top_s"]) + off
        tol = max(0.05, float(r.get("boxcar_opt_ms") or r.get("boxcar_opt_visible_ms") or 0) / 1000)
        return ((e[:, 0] == b) & (np.abs(e[:, 1] - dm) <= 0.1 * dm) & (np.abs(t - tt) <= tol) & (e[:, 1] >= 20)).any()
    f = [r for r in filas if r["set"] in ("A", "B", "P") and float(r["dm"]) <= 1000]
    real = np.mean([ok(r, 0) for r in f]); az = np.mean([ok(r, o) for r in f for o in (-1.5, -1.0, 1.0, 1.5, 2.5, 3.5)])
    return real, az, 1 - (1 - real) / (1 - az)
SEQ, T = 200, 2048


def evaluar_todos(d):
    sub = d / "todos"
    sub.mkdir(exist_ok=True)
    for f in ("asignacion.csv", "emisor.json", "server.log", "gpu.csv", "vmstat.log"):
        if not (sub / f).exists():
            (sub / f).symlink_to(d / f)
    s = sub / "sifter.log"
    if s.exists() or s.is_symlink():
        s.unlink()
    s.symlink_to(d / "eventos_todos.log")
    subprocess.run([PY, str(PC / "evaluacion" / "evaluar_prueba_64b.py"), str(sub)], capture_output=True, cwd=PC)
    return (sub / "resultados.md").read_text(), np.load(sub / "mejor_snr_por_archivo.npy")


txt = ["# Diagnóstico de la agrupación del grouper v1 con la RFI del sitio\n",
       "Todos los eventos con S/N > 6, sin topes; 80 archivos (DM ≤ 1000: 6 por set). FP ruido por beam·hora.\n",
       "| Corrida | Eventos totales | Por beam y chunk (mediana) | Umbral S/N 8: FP ruido/h | Recall A | Recall B | Recall cortados | FP no-FRB % | Aciertos A+B+cortados | Por azar | Detección real corregida |",
       "|---|---|---|---|---|---|---|---|---|---|---|"]
mejores, filas = {}, None
for nom, desc in CORRIDAS:
    d = R / "resultados" / nom
    if not (d / "eventos_todos.log").exists():
        txt.append(f"| {desc} | (falta) | | | | | | | | | |")
        continue
    lg = (d / "eventos_todos.log").read_text(errors="replace")
    e = np.array([[float(x) for x in m.groups()] for m in EV.finditer(lg)]).reshape(-1, 3)
    if len(e) == 0:
        txt.append(f"| {desc} | 0 | 0 | | | | | | | | |")
        continue
    ich = (e[:, 2] // (T * SEQ)).astype(int)
    pbc = np.unique(ich * 1000 + e[:, 0].astype(int), return_counts=True)[1]
    md, mej = evaluar_todos(d)
    mejores[nom] = mej
    filas = filas or list(csv.DictReader(open(d / "asignacion.csv")))
    r8 = [l for l in md.splitlines() if l.startswith("| 8.0 |")]
    c = r8[0].split("|")[1:-1] if r8 else ["8.0"] + ["—"] * 7
    re_, az_, co_ = azar(e, list(csv.DictReader(open(d / "asignacion.csv"))))
    txt.append(f"| {desc} | {len(e)} | {np.median(pbc):.0f} | {c[1].strip()} | {c[2].strip()} | {c[3].strip()} | {c[4].strip()} | {c[5].strip()} | {re_*100:.0f} % | {az_*100:.0f} % | {co_*100:.0f} % |")

txt += ["\n## Mejor S/N encontrado para cada FRB (DM ≤ 1000; vacío = no detectado)\n",
        "| Archivo | DM | S/N esperado | " + " | ".join(desc for nom, desc in CORRIDAS if nom in mejores) + " |",
        "|---|---|---|" + "---|" * len(mejores)]
for i, r in enumerate(filas or []):
    if r["set"] not in ("A", "B", "P") or float(r["dm"]) > 1000:
        continue
    esp = r.get("snr_esperado") or r.get("snr_visible_esperado") or ("8" if r["set"] == "A" else "")
    vals = []
    for nom, desc in CORRIDAS:
        if nom in mejores:
            v = mejores[nom][i]
            vals.append(f"{v:.1f}" if np.isfinite(v) else "")
    txt.append(f"| {r['name']} | {float(r['dm']):.0f} | {float(esp):.1f} | " + " | ".join(vals) + " |")
(R / "resultados" / SALIDA).write_text("\n".join(txt) + "\n")
print("\n".join(txt))
