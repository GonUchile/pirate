#!/usr/bin/env python
"""T3b · ¿Qué FRB se pierden? Diagnóstico de cada archivo A/B/cortado (DM ≤ 1000) no detectado con el umbral 6,5.

Categorías (en orden):
  bajo_esperado       S/N esperado < umbral: el simulador ya lo deja bajo el umbral
  fuera_tolerancia    hay un evento ≥ umbral en el mismo beam a ±1 s en t_top, pero falla DM o tiempo
  medido_bajo         S/N esperado ≥ umbral, pero el mejor evento que acierta queda bajo el umbral (o no hay evento)

Uso: python perdidos.py [CARPETA_CORRIDA]
"""
import csv
import json
import re
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

AQUI = Path(__file__).resolve().parent
if len(sys.argv) < 2 and "CHARTS_RUN" not in os.environ:
    sys.exit(f"Uso: python {sys.argv[0]} CARPETA_DE_LA_CORRIDA   (o export CHARTS_RUN=...)")
RUN = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(os.environ["CHARTS_RUN"])   # corrida completa con eventos_v0.log
SALIDA = Path(os.environ.get("CHARTS_ANALYSIS_OUT", RUN / "analisis"))   # tablas y figuras
SALIDA.mkdir(parents=True, exist_ok=True)
DD = Path(__file__).resolve().parents[2] / "configs" / "charts_64b_dm1000.yml"
SEQ, DM_MIN, TOL_DM, TOL_T, UMBRAL = 200, 20.0, 0.10, 0.05, 6.50
plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white", "font.size": 10})


def retardo_borde_a_top():
    import pirate_frb as pf
    cfg = pf.DedispersionConfig.from_yaml(str(DD))
    f_lo, f_hi = cfg.zone_freq_edges[0], cfg.zone_freq_edges[-1]
    return 1e-3 / cfg.dm_per_unit_delay() * (f_lo ** -2 - 501.6 ** -2) / (f_lo ** -2 - f_hi ** -2)


K_TOP = retardo_borde_a_top()
ev_re = re.compile(r"beam_id=(\d+), dm=([\d.e+-]+), fpga_timestamp=(-?\d+), width=([\d.e+-]+) ms.*snr=([\d.e+-]+)")
ev = np.array([(int(m.group(1)), float(m.group(2)), int(m.group(3)) / SEQ * 1e-3, float(m.group(4)), float(m.group(5)))
               for m in map(ev_re.search, (RUN / "sifter.log").read_text(errors="replace").splitlines()) if m])
ev = ev[ev[:, 1] >= DM_MIN]
e_b, e_dm, e_w, e_s = ev[:, 0].astype(int), ev[:, 1], ev[:, 3], ev[:, 4]
e_t = ev[:, 2] - e_dm * K_TOP


def f(r, k, d=np.nan):
    try:
        return float(r[k])
    except (KeyError, ValueError, TypeError):
        return d


filas = [r for r in csv.DictReader(open(RUN / "asignacion.csv")) if r["set"] != "NEG" and float(r["dm"]) <= 1000]
out = []
for r in filas:
    b, dm = int(r["beam"]), float(r["dm"])
    t_abs = int(r["inicio_muestra"]) / 1000 + float(r["t_top_s"])
    box = (f(r, "boxcar_opt_ms", 0) or f(r, "boxcar_opt_visible_ms", 0)) / 1000
    esp = f(r, "snr_visible_esperado") if r["set"] == "P" else f(r, "snr_esperado")
    tol_t = max(TOL_T, box)
    sel = e_b == b
    ok = sel & (np.abs(e_dm - dm) <= max(TOL_DM * dm, 1.0)) & (np.abs(e_t - t_abs) <= tol_t)
    mejor = e_s[ok].max() if ok.any() else 0.0
    if mejor >= UMBRAL:
        continue
    cerca = sel & (np.abs(e_t - t_abs) <= 1.0) & (e_s >= UMBRAL)
    d = dict(name=r["name"], set=r["set"], borde=r.get("borde", ""), dm=dm, snr_esp=esp, mejor=mejor,
             ancho_ms=f(r, "width_s") * 1e3, scat_600_ms=f(r, "scat_600_s") * 1e3, boxcar_opt_ms=box * 1e3,
             frac_visible=f(r, "frac_energia_visible"), n_canales=f(r, "n_canales_emision"))
    if np.isfinite(esp) and esp < UMBRAL:
        d["cat"] = "bajo_esperado"
    elif cerca.any():
        k = np.flatnonzero(cerca)[np.argmax(e_s[cerca])]
        d.update(cat="fuera_tolerancia", ev_snr=e_s[k], ev_dm=e_dm[k], ev_w=e_w[k],
                 err_dm=e_dm[k] - dm, err_t_ms=(e_t[k] - t_abs) * 1e3, tol_t_ms=tol_t * 1e3)
    else:
        d["cat"] = "medido_bajo"
    out.append(d)

tot = {s: sum(r["set"] == s for r in filas) for s in ("A", "B", "P")}
cats = ("bajo_esperado", "medido_bajo", "fuera_tolerancia")
txt = f"# T3b · FRB perdidos con umbral {UMBRAL} (DM ≤ 1000)\n\n"
txt += "| Set | total | perdidos | " + " | ".join(cats) + " |\n|---|---|---|---|---|---|\n"
for s in ("A", "B", "P"):
    p = [d for d in out if d["set"] == s]
    txt += f"| {s} | {tot[s]} | {len(p)} | " + " | ".join(str(sum(d['cat'] == c for d in p)) for c in cats) + " |\n"
txt += "\n## Cortados perdidos según borde\n\n"
for bo in ("inicio", "fin"):
    p = [d for d in out if d["set"] == "P" and d["borde"] == bo]
    txt += f"- {bo}: {len(p)} perdidos de {sum(r['set'] == 'P' and r.get('borde') == bo for r in filas)}; " \
           + ", ".join(f"{c} {sum(d['cat'] == c for d in p)}" for c in cats) + "\n"
txt += "\n## medido_bajo y fuera_tolerancia, uno por uno\n\n| archivo | S/N esp | mejor | DM | ancho ms | scat600 ms | boxcar ms | frac vis | categoría | detalle |\n|---|---|---|---|---|---|---|---|---|---|\n"
for d in sorted([d for d in out if d["cat"] != "bajo_esperado"], key=lambda d: -(d["snr_esp"] if np.isfinite(d["snr_esp"]) else 0)):
    det = (f"evento S/N {d['ev_snr']:.1f} DM {d['ev_dm']:.0f} (ΔDM {d['err_dm']:+.0f}), Δt {d['err_t_ms']:+.0f} ms "
           f"(tol {d['tol_t_ms']:.0f}), ancho {d['ev_w']:g}") if d["cat"] == "fuera_tolerancia" else ""
    txt += (f"| {d['name']} | {d['snr_esp']:.1f} | {d['mejor']:.2f} | {d['dm']:.0f} | {d['ancho_ms']:.2f} | "
            f"{d['scat_600_ms']:.1f} | {d['boxcar_opt_ms']:.0f} | {d['frac_visible']:.2f} | {d['cat']} | {det} |\n")
txt += "\n## Set A (S/N 8) según el boxcar óptimo (ancho intrínseco + scattering)\n\n| boxcar óptimo | n | recall |\n|---|---|---|\n"
perd_A = {d["name"] for d in out if d["set"] == "A"}
bxA = [((f(r, "boxcar_opt_ms", 0)), r["name"] in perd_A) for r in filas if r["set"] == "A"]
for lo, hi in ((0, 8), (8, 50), (50, 200), (200, 1e9)):
    sel_ = [p_ for b_, p_ in bxA if lo <= b_ < hi]
    if sel_:
        txt += f"| {lo:g}–{hi:g} ms | {len(sel_)} | {100 * (1 - np.mean(sel_)):.1f} % |\n".replace("–1e+09", "+")
(SALIDA / "perdidos.md").write_text(txt)
(SALIDA / "perdidos.json").write_text(json.dumps(out, indent=1, default=float, ensure_ascii=False))
print(txt)

# figura: S/N esperado de los perdidos por categoría
fig, ax = plt.subplots(figsize=(8, 4.2))
col = {"bajo_esperado": "#440154", "medido_bajo": "#21918c", "fuera_tolerancia": "#fde725"}
bins = np.logspace(-1.5, 2.5, 40)
ax.hist([[d["snr_esp"] for d in out if d["cat"] == c and np.isfinite(d["snr_esp"])] for c in cats], bins=bins,
        stacked=True, color=[col[c] for c in cats], edgecolor="0.3", lw=0.3,
        label=["S/N esperado bajo el umbral", "esperado ≥ umbral, medido bajo", "detectado fuera de tolerancia"])
ax.axvline(UMBRAL, color="black", ls="--", lw=1)
ax.set_xscale("log")
ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
ax.set_xlabel("S/N esperado (cortados: S/N visible)")
ax.set_ylabel("FRB perdidos")
ax.set_title(f"FRB perdidos con umbral {UMBRAL} (A + B + cortados, DM ≤ 1000): {len(out)}")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(SALIDA / "5_perdidos.png", dpi=130)
