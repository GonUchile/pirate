#!/usr/bin/env python3
"""Corridas completas (4000 archivos) con la máscara m1: con RFI del sitio vs control sin RFI.
Evalúa con TODOS los eventos (eventos_todos.log, sin topes) y con lo que llegó al sifter (tope 4/beam), y corrige
por aciertos al azar (tiempos desplazados). Salida: resultados/completas_m1.md
"""
import csv, re, subprocess
import os
import sys
from pathlib import Path
import numpy as np

if "CHARTS_RFI_RESULTS" not in os.environ:
    sys.exit("ERROR: export CHARTS_RFI_RESULTS=/carpeta/con/resultados/<corrida> (las corridas a analizar)")
R = Path(os.environ["CHARTS_RFI_RESULTS"])  # carpeta con resultados/<corrida>
PC = Path(__file__).resolve().parent.parent        # raíz del repositorio (evaluate_test_64b.py)
PY = sys.executable   # correr este script con el Python del entorno de pirate
EV = re.compile(r"beam_id=(\d+), dm=([\d.e+-]+), fpga_timestamp=(-?\d+), width=([\d.e+-]+) ms.*snr=([\d.e+-]+)")
K = 4.148808e3 * (299.8497764530551 ** -2 - 501.6 ** -2)


def evaluar(d, log, etiqueta):
    sub = d / etiqueta; sub.mkdir(exist_ok=True)
    for f in ("asignacion.csv", "emisor.json", "server.log", "gpu.csv", "vmstat.log"):
        if not (sub / f).exists():
            (sub / f).symlink_to(d / f)
    s = sub / "sifter.log"
    if s.exists() or s.is_symlink():
        s.unlink()
    s.symlink_to(log)
    subprocess.run([PY, str(PC / "09_evaluation" / "evaluate_test_64b.py"), str(sub)], capture_output=True, cwd=PC)
    return (sub / "resultados.md").read_text()


def azar(e, filas, umbral):
    e = e[e[:, 4] >= umbral]
    t = e[:, 2] / 200e3 - e[:, 1] * K
    orden = np.argsort(e[:, 0], kind="stable"); e, t = e[orden], t[orden]
    ini = {b: np.searchsorted(e[:, 0], b) for b in range(65)}
    def ok(r, off):
        b, dm = int(r["beam"]), float(r["dm"])
        a, z = ini[b], ini[b + 1]
        tt = float(r["inicio_muestra"]) / 1000 + float(r["t_top_s"]) + off
        tol = max(0.05, float(r.get("boxcar_opt_ms") or r.get("boxcar_opt_visible_ms") or 0) / 1000)
        return ((np.abs(e[a:z, 1] - dm) <= 0.1 * dm) & (np.abs(t[a:z] - tt) <= tol) & (e[a:z, 1] >= 20)).any()
    res = {}
    for s in ("A", "B", "P"):
        f = [r for r in filas if r["set"] == s and float(r["dm"]) <= 1000]
        real = np.mean([ok(r, 0) for r in f]); az = np.mean([ok(r, o) for r in f for o in (-1.5, -1.0, 1.0, 1.5, 2.5, 3.5)])
        res[s] = (len(f), real, az, 1 - (1 - real) / (1 - az))
    return res


txt = ["# Corridas completas con máscara m1: RFI del sitio vs control sin RFI\n",
       "4000 archivos en 64 beams (3510 s de stream por beam); grouper v1 `-t 6 --sin-ventanas --max-cand 512 --log-todos`; "
       "máscara m1 (17,5 % de la banda). Métricas con DM ≤ 1000. **FP ruido por beam·hora** (10/h del sistema de 64 beams ≈ 0,16 por beam·hora).\n"]
for nom, desc in (("completa_m1_sin_rfi", "Sin RFI + m1 (control)"), ("completa_m1_rfi", "RFI del sitio + m1")):
    d = R / "resultados" / nom
    e = np.array([[float(x) for x in m.groups()] for m in EV.finditer((d / "eventos_todos.log").read_text(errors="replace"))]).reshape(-1, 5)
    filas = list(csv.DictReader(open(d / "asignacion.csv")))
    nch = 1714
    ich = (e[:, 2] // (2048 * 200)).astype(int)
    pbc = np.bincount(ich * 64 + e[:, 0].astype(int), minlength=nch * 64)
    txt += [f"\n## {desc} (`{nom}`)\n",
            f"- Eventos (todos, S/N > 6): {len(e)}; por beam y chunk: media {pbc.mean():.2f}, p99 {np.percentile(pbc, 99):.0f}, máx {pbc.max()}; "
            f"beam-chunks con > 4 eventos: {(pbc > 4).mean() * 100:.2f} %."]
    for umb in (6.0, 7.0):
        az = azar(e, filas, umb)
        txt.append(f"- Aciertos por azar con S/N ≥ {umb:g} (A / B / cortados): " +
                   " / ".join(f"{az[s][2] * 100:.1f} %" for s in ("A", "B", "P")) +
                   "; detección real corregida: " + " / ".join(f"{az[s][3] * 100:.1f} %" for s in ("A", "B", "P")) +
                   f" (n = {az['A'][0]} / {az['B'][0]} / {az['P'][0]}).")
    for etiqueta, log, t in (("todos", d / "eventos_todos.log", "todos los eventos (sin tope)"), ("tope", d / "sifter.log", "lo que llegó al sifter (tope 4 por beam)")):
        md = evaluar(d, log, etiqueta)
        filas_md = [l for l in md.splitlines() if re.match(r"^\| [0-9]", l)]
        txt += [f"\n**Con {t}:**\n", "| umbral S/N | FP ruido/h | Recall A | Recall B | Recall cortados | FP no-FRB % | A DM≤1105 | B DM≤1105 |",
                "|---|---|---|---|---|---|---|---|"] + filas_md[:4]
        if len(filas_md) > 4:
            txt += ["\n| FP objetivo/h | umbral | FP/h | A | B | cortados | no-FRB % | A≤1105 | B≤1105 |", "|---|---|---|---|---|---|---|---|---|"] + filas_md[4:]
    rend = [l for l in (d / "resultados.md").read_text().splitlines() if l.startswith("| Ritmo") or l.startswith("| Capacidad") or l.startswith("| GPU")]
    txt += ["\n**Rendimiento:**\n", "| | |", "|---|---|"] + rend
(R / "resultados" / "completas_m1.md").write_text("\n".join(txt) + "\n")
print("\n".join(txt))
