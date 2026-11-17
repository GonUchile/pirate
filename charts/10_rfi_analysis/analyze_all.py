#!/usr/bin/env python3
"""Conteo de TODOS los candidatos (grouper v1 sin topes --k/--k-total) con y sin la RFI del sitio.

Para cada corrida (resultados/corta_todos_rfi, corta_todos_sin_rfi) lee eventos_todos.log (formato del sifter.log)
y sifter.log (lo que llegó al sifter, con el tope de 4 por beam y 256 por chunk), y reporta:
  - cuántos eventos hay: total, por chunk, por beam y chunk, por hora;
  - cómo son los eventos fuera de las ventanas de señal: S/N, DM, ancho;
  - cuántos aparecen en muchos beams a la vez (mismo t_top ± 50 ms en ≥ 8 / ≥ 32 beams);
  - Recall y FP con todos los eventos: se arma <corrida>/todos/ con sifter.log → eventos_todos.log y se corre
    evaluate_test_64b.py (mismos criterios que siempre).
Salidas: resultados/conteo_todos.md, figures/all_counts.png
"""
import re
import subprocess
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

if "CHARTS_RFI_RESULTS" not in os.environ:
    sys.exit("ERROR: export CHARTS_RFI_RESULTS=/carpeta/con/resultados/<corrida> (las corridas a analizar)")
R = Path(os.environ["CHARTS_RFI_RESULTS"])  # carpeta con resultados/<corrida>
PC = Path(__file__).resolve().parent.parent        # raíz del repositorio (evaluate_test_64b.py)
PY = sys.executable   # correr este script con el Python del entorno de pirate
EV = re.compile(r"beam_id=(\d+), dm=([\d.e+-]+), fpga_timestamp=(-?\d+), width=([\d.e+-]+) ms.*snr=([\d.e+-]+)")
CH = re.compile(r"fpga=\[(\d+):(\d+)\]")
SEQ, T = 200, 2048
K_TOP = None


def k_top():
    """Segundos por unidad de DM entre el borde inferior de la banda (donde fecha pirate) y 501,6 MHz."""
    return 4.148808e3 * (299.8497764530551 ** -2 - 501.6 ** -2)


def leer(log):
    txt = Path(log).read_text(errors="replace")
    e = np.array([[float(x) for x in m.groups()] for m in EV.finditer(txt)]) if EV.search(txt) else np.zeros((0, 5))
    nch = len(set(CH.findall(txt)))
    return e, nch


def ventanas_senal(d):
    """Intervalos [t0, t1] (s del stream) por beam que contienen señal: archivo ± barrido ± 2 s, como el evaluador."""
    import csv
    v = {}
    for r in csv.DictReader(open(d / "asignacion.csv")):
        t0 = float(r["inicio_muestra"]) / 1000 + float(r["t_top_s"])
        bar = float(r.get("barrido_s") or 0)
        v.setdefault(int(r["beam"]), []).append((t0 - bar - 2, t0 + bar + 2))
    return v


def fuera_de_senal(e, v):
    t = e[:, 2] / SEQ * 1e-3 - e[:, 1] * K_TOP
    m = np.ones(len(e), bool)
    for b, iv in v.items():
        sel = e[:, 0] == b
        for a, z in iv:
            m[sel & (t >= a) & (t <= z)] = False
    return m, t


def multibeam(e, t, tol=0.05):
    """Para cada evento, en cuántos beams distintos hay un evento con el mismo t_top ± tol."""
    o = np.argsort(t); ts, bs = t[o], e[o, 0].astype(int)
    n = np.zeros(len(e), int)
    j0 = 0
    for i in range(len(ts)):
        while ts[j0] < ts[i] - tol:
            j0 += 1
        j1 = np.searchsorted(ts, ts[i] + tol, "right")
        n[o[i]] = len(set(bs[j0:j1]))
    return n


def evaluar(d, log, etiqueta):
    sub = d / etiqueta
    sub.mkdir(exist_ok=True)
    for f in ("asignacion.csv", "emisor.json", "server.log", "gpu.csv", "vmstat.log"):
        if not (sub / f).exists():
            (sub / f).symlink_to(d / f)
    s = sub / "sifter.log"
    if s.exists() or s.is_symlink():
        s.unlink()
    s.symlink_to(log)
    subprocess.run([PY, str(PC / "evaluation" / "evaluate_test_64b.py"), str(sub)], capture_output=True, cwd=PC)
    return (sub / "resultados.md").read_text()


def tabla(md):
    """Filas de las dos tablas de Recall/FP del resultados.md."""
    return [l for l in md.splitlines() if re.match(r"^\| [0-9]", l)]


K_TOP = k_top()
txt = ["# Conteo de todos los candidatos con y sin RFI del sitio (grouper v1 sin topes)\n",
       f"Umbral de los eventos: S/N > 6 (`-t 6`). `--max-cand 512` candidatos decodificados por beam y chunk. "
       f"Tiempo de cada evento llevado a 501,6 MHz con K = {K_TOP * 1e3:.2f} ms por unidad de DM.\n"]
res = {}
for v in ("sin_rfi", "rfi"):
    d = R / "resultados" / f"corta_todos_{v}"
    e, nch = leer(d / "eventos_todos.log")
    es, _ = leer(d / "sifter.log")
    vs = ventanas_senal(d)
    fuera, t = fuera_de_senal(e, vs)
    nb = multibeam(e, t)
    horas = nch * T / 1000 / 3600
    ich = (e[:, 2] // (T * SEQ)).astype(int)
    por_chunk = np.bincount(ich)
    pbc = np.unique(ich * 1000 + e[:, 0].astype(int), return_counts=True)[1]
    res[v] = dict(e=e, fuera=fuera, nb=nb)
    txt += [f"\n## {'Con RFI del sitio' if v == 'rfi' else 'Sin RFI (control, con normalización)'} (`{d.name}`)\n",
            "| | |", "|---|---|",
            f"| Chunks | {nch} ({horas * 3600:.0f} s de stream por beam; {horas * 64:.2f} beam·hora) |",
            f"| Eventos totales (todos) | {len(e)} |",
            f"| Eventos que llegaron al sifter (tope 4/beam, 256/chunk) | {len(es)} |",
            f"| Eventos por chunk (todos): mediana / máx | {np.median(por_chunk):.0f} / {por_chunk.max()} |",
            f"| Eventos por beam y chunk: mediana / p90 / máx | {np.median(pbc):.0f} / {np.percentile(pbc, 90):.0f} / {pbc.max()} |",
            f"| Fracción de beam-chunks con > 4 eventos (los que el tope recortaba) | {(pbc > 4).sum() / (nch * 64) * 100:.1f} % |",
            f"| Eventos fuera de las ventanas de señal | {fuera.sum()} ({fuera.sum() / (horas * 64):.0f} por beam·hora) |",
            f"| … de ellos con DM < 100 | {(fuera & (e[:, 1] < 100)).mean() / max(fuera.mean(), 1e-9) * 100:.0f} % |",
            f"| … en ≥ 8 beams a la vez (±50 ms) | {(fuera & (nb >= 8)).sum() / max(fuera.sum(), 1) * 100:.0f} % |",
            f"| … en ≥ 32 beams a la vez | {(fuera & (nb >= 32)).sum() / max(fuera.sum(), 1) * 100:.0f} % |",
            f"| S/N fuera de señal: p50 / p90 / p99 / máx | " + " / ".join(f"{x:.1f}" for x in np.percentile(e[fuera, 4], [50, 90, 99, 100])) + " |" if fuera.any() else "| S/N fuera de señal | — |",
            f"| Ancho fuera de señal (ms): p50 / p90 | " + " / ".join(f"{x:.0f}" for x in np.percentile(e[fuera, 3], [50, 90])) + " |" if fuera.any() else "| Ancho | — |"]
    for etiqueta, log in (("tope", d / "sifter.log"), ("todos", d / "eventos_todos.log")):
        md = evaluar(d, log, etiqueta)
        txt += [f"\n**Recall y FP con los eventos {'que llegaron al sifter (con tope)' if etiqueta == 'tope' else 'de todos (sin tope)'}** "
                f"(DM ≤ 1000; FP ruido por beam·hora):\n",
                "| umbral S/N | FP ruido/h | Recall A | Recall B | Recall cortados | FP no-FRB % | Recall A DM≤1105 | Recall B DM≤1105 |",
                "|---|---|---|---|---|---|---|---|"] + tabla(md)[:4]
        fp = [l for l in tabla(md)[4:]]
        if fp:
            txt += ["\nA igual FP por ruido (objetivo / umbral / FP / Recall A / B / cortados / FP no-FRB / A≤1105 / B≤1105):\n",
                    "| FP obj | umbral | FP/h | A | B | cortados | no-FRB % | A≤1105 | B≤1105 |", "|---|---|---|---|---|---|---|---|---|"] + fp
(R / "resultados" / "conteo_todos.md").write_text("\n".join(txt) + "\n")
print("\n".join(txt))

# figura
fig, ax = plt.subplots(2, 2, figsize=(16, 10))
col = {"sin_rfi": plt.cm.viridis(0.25), "rfi": plt.cm.viridis(0.8)}
nom = {"sin_rfi": "sin RFI", "rfi": "con RFI del sitio"}
for v, r in res.items():
    e, f = r["e"], r["fuera"]
    if not f.any():
        continue
    ax[0, 0].hist(e[f, 4], bins=np.geomspace(6, 300, 50), histtype="step", lw=2, color=col[v], label=f"{nom[v]} ({f.sum()})")
    ax[0, 1].hist(e[f, 1], bins=np.linspace(0, 1105, 56), histtype="step", lw=2, color=col[v], label=nom[v])
    ax[1, 0].hist(e[f, 3], bins=np.geomspace(1, 64, 20), histtype="step", lw=2, color=col[v], label=nom[v])
    ax[1, 1].hist(r["nb"][f], bins=np.arange(1, 66), histtype="step", lw=2, color=col[v], label=nom[v])
for a, (xl, lg) in zip(ax.ravel(), (("S/N", True), ("DM (pc cm⁻³)", False), ("Ancho (ms)", True), ("Beams con el mismo evento (±50 ms)", False))):
    a.set(xlabel=xl, ylabel="eventos fuera de las ventanas de señal", yscale="log")
    if lg:
        a.set_xscale("log")
    a.legend(); a.grid(alpha=0.3)
fig.suptitle("Todos los candidatos del grouper v1 (sin tope) fuera de las ventanas de señal: con y sin la RFI del sitio", fontsize=14)
fig.tight_layout(rect=(0, 0, 1, 0.96)); fig.savefig(R / "figuras" / "all_counts.png", dpi=110)
