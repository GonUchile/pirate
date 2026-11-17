#!/usr/bin/env python
"""T3e · Comparación archivo por archivo: subbandas [0,0,1,2,1] frente a banda completa, a igual FP por ruido.

Para cada archivo (A/B/P/NEG, DM ≤ 1000) toma el mejor evento que acierta (mismos criterios que evaluate_test_64b.py)
en cada corrida y la subbanda de ese evento. Umbrales de 10 FP de ruido por beam·hora: 6,62 (subbandas) y 6,48 (banda completa).
Uso: python paired_comparison.py
"""
import csv
import os
import sys
import re
from collections import Counter
from pathlib import Path

import numpy as np

AQUI = Path(__file__).resolve().parent
if "CHARTS_SUBBAND_RESULTS" not in os.environ:
    sys.exit("ERROR: export CHARTS_SUBBAND_RESULTS=/carpeta/con/completa_subbands/y/completa_banda_completa")
BASE = Path(os.environ["CHARTS_SUBBAND_RESULTS"])   # carpeta con completa_subbands/ y completa_banda_completa/
DD = Path(__file__).resolve().parents[2] / "06_configs" / "charts_64b_dm1000.yml"
SEQ, DM_MIN, TOL_DM, TOL_T = 200, 20.0, 0.10, 0.05
CORR = {"sb": ("completa_subbandas", 6.62), "bc": ("completa_banda_completa", 6.48)}


def retardo_borde_a_top():
    import pirate_frb as pf
    cfg = pf.DedispersionConfig.from_yaml(str(DD))
    f_lo, f_hi = cfg.zone_freq_edges[0], cfg.zone_freq_edges[-1]
    return 1e-3 / cfg.dm_per_unit_delay() * (f_lo ** -2 - 501.6 ** -2) / (f_lo ** -2 - f_hi ** -2)


K_TOP = retardo_borde_a_top()
ev_re = re.compile(r"beam_id=(\d+), dm=([\d.e+-]+), fpga_timestamp=(-?\d+), width=([\d.e+-]+) ms, "
                   r"subband=\[([\d.]+),([\d.]+)\] MHz, snr=([\d.e+-]+)")


def leer(nombre):
    filas = []
    for m in map(ev_re.search, (BASE / nombre / "sifter.log").read_text(errors="replace").splitlines()):
        if m:
            filas.append((int(m[1]), float(m[2]), int(m[3]) / SEQ * 1e-3, float(m[4]), f"{m[5]}–{m[6]}", float(m[7])))
    b, dm, t, w, sb, s = map(np.array, zip(*filas))
    sel = dm >= DM_MIN
    return dict(b=b[sel].astype(int), dm=dm[sel], t=t[sel] - dm[sel] * K_TOP, w=w[sel], sb=sb[sel], s=s[sel])


def f(r, k, d=np.nan):
    try:
        return float(r[k])
    except (KeyError, ValueError, TypeError):
        return d


EV = {k: leer(n) for k, (n, _) in CORR.items()}
filas = [r for r in csv.DictReader(open(BASE / CORR["sb"][0] / "asignacion.csv")) if float(r["dm"]) <= 1000]
res = []
for r in filas:
    b, dm = int(r["beam"]), float(r["dm"])
    t_abs = int(r["inicio_muestra"]) / 1000 + float(r["t_top_s"])
    box = (f(r, "boxcar_opt_ms", 0) or f(r, "boxcar_opt_visible_ms", 0)) / 1000
    d = dict(r=r, box_ms=box * 1e3)
    for k, e in EV.items():
        ok = (e["b"] == b) & (np.abs(e["dm"] - dm) <= max(TOL_DM * dm, 1.0)) & (np.abs(e["t"] - t_abs) <= max(TOL_T, box))
        if ok.any():
            i = np.flatnonzero(ok)[np.argmax(e["s"][ok])]
            d[k] = (e["s"][i], e["sb"][i], e["w"][i])
        else:
            d[k] = (0.0, "", 0.0)
    d["det_sb"], d["det_bc"] = d["sb"][0] >= CORR["sb"][1], d["bc"][0] >= CORR["bc"][1]
    res.append(d)

txt = "# T3e · Comparación pareada, a 10 FP de ruido por beam·hora (6,62 subbandas, 6,48 banda completa)\n\n"
txt += "| Set | n | detectados ambas | solo subbandas | solo banda completa | ninguna |\n|---|---|---|---|---|---|\n"
for s in ("A", "B", "P", "NEG"):
    p = [d for d in res if d["r"]["set"] == s]
    c = Counter((d["det_sb"], d["det_bc"]) for d in p)
    txt += f"| {s} | {len(p)} | {c[(True, True)]} | {c[(True, False)]} | {c[(False, True)]} | {c[(False, False)]} |\n"

txt += "\n## Subbanda del mejor evento, en los detectados con subbandas\n\n| Set | " + " | ".join(
    sorted({d['sb'][1] for d in res if d['det_sb']})) + " |\n"
bandas = sorted({d['sb'][1] for d in res if d['det_sb']})
txt += "|---|" + "---|" * len(bandas) + "\n"
for s in ("A", "B", "P", "NEG"):
    c = Counter(d["sb"][1] for d in res if d["r"]["set"] == s and d["det_sb"])
    txt += f"| {s} | " + " | ".join(str(c[x]) for x in bandas) + " |\n"
txt += "\nRecuperados (solo subbandas), subbanda: " + str(dict(Counter(
    d["sb"][1] for d in res if d["det_sb"] and not d["det_bc"] and d["r"]["set"] != "NEG"))) + "\n"

# eventos de ruido: subbanda de todos los eventos ≥ umbral en la corrida con subbandas
e = EV["sb"]
txt += "\nTodos los eventos ≥ 6,62 de la corrida con subbandas (casi todos ruido o FRB), por subbanda: " + str(
    dict(Counter(e["sb"][e["s"] >= CORR["sb"][1]]))) + "\n"

txt += "\n## Archivos recuperados o perdidos con subbandas (A, B, cortados)\n\n"
txt += "| archivo | set | S/N esp | DM | boxcar ms | scat600 ms | canales emisión | frac vis | S/N sb (subbanda) | S/N bc |\n|---|---|---|---|---|---|---|---|---|---|\n"
for d in sorted([d for d in res if d["det_sb"] != d["det_bc"] and d["r"]["set"] != "NEG"],
                key=lambda d: (d["det_bc"], d["r"]["set"], -d["box_ms"])):
    r = d["r"]
    esp = f(r, "snr_visible_esperado") if r["set"] == "P" else f(r, "snr_esperado")
    txt += (f"| {r['name']} | {r['set']}{' (perdido)' if d['det_bc'] else ''} | {esp:.1f} | {float(r['dm']):.0f} | "
            f"{d['box_ms']:.0f} | {f(r, 'scat_600_s') * 1e3:.1f} | {r['n_canales_emision']} | {r['frac_energia_visible']} | "
            f"{d['sb'][0]:.2f} ({d['sb'][1]}) | {d['bc'][0]:.2f} |\n")

# resumen por ancho de banda de emisión (A + B)
txt += "\n## A + B según canales de emisión (de 21 504)\n\n| canales | n | recall subbandas | recall banda completa |\n|---|---|---|---|\n"
ab = [d for d in res if d["r"]["set"] in ("A", "B")]
for lo, hi in ((0, 5000), (5000, 10000), (10000, 15000), (15000, 21505)):
    p = [d for d in ab if lo <= f(d["r"], "n_canales_emision", 21504) < hi]
    if p:
        txt += f"| {lo}–{hi} | {len(p)} | {100 * np.mean([d['det_sb'] for d in p]):.1f} % | {100 * np.mean([d['det_bc'] for d in p]):.1f} % |\n"
(BASE / "comparacion_pareada.md").write_text(txt)
print(txt)
