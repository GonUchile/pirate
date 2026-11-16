#!/usr/bin/env python
"""Evaluación de la prueba de 64 beams (correr_prueba_64b.sh): Recall + FP.

Entradas (en la carpeta de la corrida):
  asignacion.csv   verdad: beam y muestra de inicio de cada archivo dentro de su stream
  sifter.log       eventos de la búsqueda (toy_grouper: el máximo de cada chunk de 2048 muestras y cada beam, S/N >= 5)
  emisor.json, server.log, gpu.csv, vmstat.log   rendimiento

Criterios (los mismos del benchmark, bench/evaluacion.py):
  - búsqueda desde DM 20 (los eventos con DM < 20 se descartan, como --dm-min 20)
  - acierto A/B/P: mismo beam, |DM - DM real| <= max(10 %, 1) y |t - t real| <= max(50 ms, boxcar óptimo),
    con t = llegada a 501,6 MHz (el evento de pirate trae la llegada al borde inferior de la banda;
    se lleva a 501,6 MHz con el retardo de su propio DM)
  - FP no-FRB: % de negativos con algún evento (cualquier DM) a ±100 ms de t_top
  - FP por ruido: eventos fuera de [t_top - barrido - 2 s, t_top + barrido + 2 s] de todos los archivos del beam
    (y fuera del relleno inicial), por hora de tiempo sin señal, sumado en los 64 beams
  - Recall sobre archivos con DM <= 1000 (pedido de la prueba); también en todo el rango de búsqueda
Se reporta con S/N >= 5, 6, 7, 8 y a igual tasa de FP por ruido (100, 30, 10 y 1 por hora).

Uso: python evaluar_prueba_64b.py tiempo_real/ETIQUETA
"""
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np

D = Path(sys.argv[1])
AQUI = Path(__file__).resolve().parent
# config de dedispersión: la que guardó el lanzador en la corrida, o la del repositorio
DD = D / "config_dedispersion.yml" if (D / "config_dedispersion.yml").exists() else AQUI.parent / "configs" / "charts_64b_dm1000.yml"
SEQ = 200                     # seq por muestra (reloj de 1,000 ms del emisor)
DM_MIN = 20.0
TOL_DM, TOL_T, TOL_NEG = 0.10, 0.05, 0.10


def retardo_borde_a_top():
    """Segundos por unidad de DM entre el borde inferior de la banda (donde pirate fecha el evento) y 501,6 MHz."""
    import pirate_frb as pf
    cfg = pf.DedispersionConfig.from_yaml(str(DD))
    f_lo, f_hi = cfg.zone_freq_edges[0], cfg.zone_freq_edges[-1]
    s_por_dm_banda = 1e-3 / cfg.dm_per_unit_delay()
    return s_por_dm_banda * (f_lo ** -2 - 501.6 ** -2) / (f_lo ** -2 - f_hi ** -2), cfg


K_TOP, CFG = retardo_borde_a_top()
DM_MAX = max(t.dm_max for t in __import__("pirate_frb").DedispersionPlan(CFG).trees)

# ---- eventos ----
ev_re = re.compile(r"beam_id=(\d+), dm=([\d.e+-]+), fpga_timestamp=(-?\d+), width=([\d.e+-]+) ms.*snr=([\d.e+-]+)")
ev = []
for linea in (D / "sifter.log").read_text(errors="replace").splitlines():
    m = ev_re.search(linea)
    if m:
        ev.append((int(m.group(1)), float(m.group(2)), int(m.group(3)), float(m.group(4)), float(m.group(5))))
ev = np.array(ev, dtype=float).reshape(-1, 5)
ev = ev[ev[:, 1] >= DM_MIN]
ev_beam, ev_dm, ev_snr = ev[:, 0].astype(int), ev[:, 1], ev[:, 4]
ev_t = ev[:, 2] / SEQ * 1e-3 - ev_dm * K_TOP            # llegada a 501,6 MHz (s, desde el inicio del stream)

# ---- verdad ----
filas = list(csv.DictReader(open(D / "asignacion.csv")))
emisor = json.loads((D / "emisor.json").read_text())
nbeams, total_s, prepad_s = emisor["nbeams"], emisor["total_muestras"] / 1000, emisor["prepad"] / 1000
for r in filas:
    r["b"] = int(r["beam"])
    r["t0"] = int(r["inicio_muestra"]) / 1000
    r["t_abs"] = r["t0"] + float(r["t_top_s"])
    r["dmv"] = float(r["dm_equivalente_pos"] if r["set"] == "NEG" else r["dm"])
    r["bar"] = float(r["barrido_s"])
    r["box"] = float(r.get("boxcar_opt_ms") or r.get("boxcar_opt_visible_ms") or 0) / 1000
    r["snr_esp"] = float(r.get("snr_esperado") or r.get("snr_visible_esperado") or "nan")

# mejor S/N de un evento que acierta cada archivo (o -inf)
mejor = np.full(len(filas), -np.inf)
for i, r in enumerate(filas):
    sel = ev_beam == r["b"]
    if r["set"] == "NEG":
        ok = sel & (np.abs(ev_t - r["t_abs"]) <= TOL_NEG)
    else:
        ok = sel & (np.abs(ev_dm - r["dmv"]) <= max(TOL_DM * r["dmv"], 1.0)) & \
             (np.abs(ev_t - r["t_abs"]) <= max(TOL_T, r["box"]))
    if ok.any():
        mejor[i] = ev_snr[ok].max()

# ruido: eventos fuera de todas las ventanas de señal; tiempo sin señal por beam
ruido_snr, libre = [], 0.0
for b in range(nbeams):
    vent = sorted([(r["t_abs"] - r["bar"] - 2, r["t_abs"] + r["bar"] + 2) for r in filas if r["b"] == b])
    vent = [(0.0, prepad_s)] + vent
    sel = ev_beam == b
    t = ev_t[sel]
    fuera = np.ones(t.size, bool)
    for lo, hi in vent:
        fuera &= ~((t >= lo) & (t <= hi))
    ruido_snr.append(ev_snr[sel][fuera])
    union, (clo, chi) = 0.0, vent[0]
    for lo, hi in vent[1:]:
        if lo > chi:
            union += chi - clo
            clo, chi = lo, hi
        else:
            chi = max(chi, hi)
    union += chi - clo
    libre += total_s - min(union, total_s)
ruido_snr = np.concatenate(ruido_snr)
horas = libre / 3600

sets = np.array([r["set"] for r in filas])
dmv = np.array([r["dmv"] for r in filas])
en_rango = dmv <= DM_MAX / (1 - TOL_DM)
d1000 = dmv <= 1000


def fila(u):
    out = {"umbral S/N": u, "FP ruido/h": round(float((ruido_snr >= u).sum() / horas), 1)}
    for s, nom in (("A", "A"), ("B", "B"), ("P", "cortados")):
        m = (sets == s) & d1000
        out[f"Recall {nom} DM≤1000 (%)"] = round(100 * float((mejor[m] >= u).mean()), 1)
    out["FP no-FRB (% NEG DM≤1000)"] = round(100 * float((mejor[(sets == "NEG") & d1000] >= u).mean()), 1)
    for s in ("A", "B"):
        m = (sets == s) & en_rango
        out[f"Recall {s} DM≤{DM_MAX:.0f} (%)"] = round(100 * float((mejor[m] >= u).mean()), 1)
    return out


umbrales = np.round(np.arange(5.0, 15.0, 0.02), 2)
fp_curve = np.array([(ruido_snr >= u).sum() / horas for u in umbrales])
tab_fijo = [fila(u) for u in (5.0, 6.0, 7.0, 8.0)]
tab_fp = []
for obj in (100, 30, 10, 1):
    k = np.flatnonzero(fp_curve <= obj)
    if len(k):
        tab_fp.append({"FP ruido/h objetivo": obj, **fila(float(umbrales[k[0]]))})


def tabla_md(filas_t):
    cols = list(filas_t[0])
    s = "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n"
    for f in filas_t:
        s += "| " + " | ".join(str(f[c]) for c in cols) + " |\n"
    return s


# ---- rendimiento ----
srv = [l for l in (D / "server.log").read_text(errors="replace").splitlines() if "beams=" in l]
beq = np.array([float(l.split("beams=")[1].split("/")[0]) for l in srv]) if srv else np.array([np.nan])
gbps = np.array([float(l.split("Gbps=")[1].split(",")[0]) for l in srv]) if srv else np.array([np.nan])
gl = [l.split(",") for l in (D / "gpu.csv").read_text().splitlines()[1:] if l.count(",") >= 3]
gu = np.array([float(x[1].strip().split()[0]) for x in gl]) if gl else np.array([np.nan])
gm = np.array([float(x[2].strip().split()[0]) for x in gl]) if gl else np.array([np.nan])
vm = [l.split() for l in (D / "vmstat.log").read_text().splitlines()[2:] if l.split() and l.split()[0].isdigit()]
idle = np.array([float(x[14]) for x in vm]) if vm else np.array([np.nan])
libre_ram = np.array([float(x[3]) / 1048576 for x in vm]) if vm else np.array([np.nan])

n = {s: int(((sets == s) & d1000).sum()) for s in ("A", "B", "NEG", "P")}
txt = f"""# Prueba de 64 beams con pirate — {D.name}

- Archivos: {len(filas)} (grupo 3, 21 504 canales, sin RFI) en {nbeams} beams; {total_s:.0f} s de stream por beam.
- Búsqueda: DM {DM_MIN:.0f}–{DM_MAX:.0f}, boxcar ≤ 32 ms, rank 15 (un árbol), 4 bits n = 2 (centro 32,5).
- Con DM ≤ 1000: A {n['A']}, B {n['B']}, NEG {n['NEG']}, cortados {n['P']}.
- Tiempo sin señal para el FP por ruido: {horas:.2f} h (64 beams sumados). Eventos de la búsqueda (DM ≥ 20): {len(ev)}.
- Nota: el toy_grouper entrega un solo evento por chunk de 2,048 s y por beam (el máximo), con S/N ≥ 5.

## Con umbral fijo

{tabla_md(tab_fijo)}
## A igual tasa de FP por ruido

{tabla_md(tab_fp) if tab_fp else '(sin datos suficientes)'}
## Rendimiento

| | |
|---|---|
| Ritmo del emisor | {emisor.get('ritmo_x_tiempo_real', float('nan')):.2f}× tiempo real ({emisor.get('duracion_s', float('nan')):.0f} s para {total_s:.0f} s de datos) |
| Capacidad del servidor (línea beams=X/{nbeams}) | mediana {np.nanmedian(beq):.1f} beams equivalentes ({np.nanmedian(beq) / nbeams:.2f}× tiempo real) |
| Entrada | mediana {np.nanmedian(gbps):.1f} Gbps |
| GPU | uso medio {np.nanmean(gu):.0f} %, VRAM máx {np.nanmax(gm):.0f} MiB |
| CPU ociosa | media {np.nanmean(idle):.0f} % |
| RAM libre mínima | {np.nanmin(libre_ram):.1f} GB |
| Muestras recortadas en ±7 (4 bits) | {100 * emisor.get('frac_recortada', float('nan')):.4f} % (solo ruido gaussiano: 0,126 %) |
"""
(D / "resultados.md").write_text(txt)
np.save(D / "mejor_snr_por_archivo.npy", mejor)
print(txt)
