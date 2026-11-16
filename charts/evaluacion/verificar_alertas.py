#!/usr/bin/env python3
"""Verifica las alertas de una corrida contra los FRB inyectados por el emisor.

Para cada alerta de <corrida>/alertas.jsonl busca el archivo del mismo beam cuya llegada a 501,6 MHz coincide
(mismos criterios que evaluar_prueba_64b.py: |ΔDM| ≤ 10 % y |Δt| ≤ max(50 ms, boxcar óptimo); negativos ±100 ms).
Clasifica cada alerta como FRB verdadero, no-FRB (negativo) o falsa alarma, y lista los FRB que deberían haber
generado alerta (S/N esperado ≥ umbral y DM en rango) y no la generaron.

Escribe <corrida>/alertas.md. Uso: python evaluacion/verificar_alertas.py CARPETA_DE_LA_CORRIDA
"""
import csv
import json
import sys
from pathlib import Path

import yaml

TOL_DM, TOL_T, TOL_NEG = 0.10, 0.05, 0.10

if len(sys.argv) < 2:
    sys.exit(__doc__)
D = Path(sys.argv[1])
if not (D / "alertas.jsonl").exists():
    sys.exit(f"ERROR: no está {D / 'alertas.jsonl'} (¿la corrida usó servidor/sifter/sifter_alertas.py?)")
alertas = [json.loads(x) for x in (D / "alertas.jsonl").read_text().splitlines() if x.strip()]
cfg = yaml.safe_load(open(D / "prueba.yml")) if (D / "prueba.yml").exists() else {}
reg = (cfg.get("servidor") or {}).get("alertas") or {}
umbral, dm_min, dm_max = reg.get("umbral", 7.6), reg.get("dm_min", 20), reg.get("dm_max")
dm_busqueda = 1105.0

filas = list(csv.DictReader(open(D / "asignacion.csv")))
for r in filas:
    r["b"], r["t_abs"] = int(r["beam"]), int(r["inicio_muestra"]) / 1000 + float(r["t_top_s"])
    r["dmv"] = float(r["dm_equivalente_pos"] if r["set"] == "NEG" else r["dm"])
    r["box"] = float(r.get("boxcar_opt_ms") or r.get("boxcar_opt_visible_ms") or 0) / 1000
    r["snr_esp"] = float(r.get("snr_esperado") or r.get("snr_visible_esperado") or "nan")


def coincide(al, r, beams=None):
    if r["b"] not in (beams or [al["beam"]]):
        return False
    dt = abs(al["t_stream_s"] - r["t_abs"])
    if r["set"] == "NEG":
        return dt <= TOL_NEG
    return abs(al["dm"] - r["dmv"]) <= max(TOL_DM * r["dmv"], 1.0) and dt <= max(TOL_T, r["box"])


alertas = [al for al in alertas if al.get("tipo", "alerta") == "alerta"]        # las actualizaciones no cuentan aparte
filas_al, detectados = [], set()
for al in alertas:
    r = next((r for r in filas if coincide(al, r)), None)
    tipo = "falsa alarma (ruido)" if r is None else ("no-FRB (negativo)" if r["set"] == "NEG" else "**FRB verdadero**")
    # un mismo pulso en varios beams a la vez da una sola alerta (beams_coincidentes): cuenta para todos ellos
    otros = [x["name"] for x in filas if x["set"] != "NEG" and coincide(al, x, al.get("beams_coincidentes"))]
    detectados.update(otros)
    if r is None and otros:
        r, tipo = next(x for x in filas if x["name"] == otros[0]), "**FRB verdadero** (beam coincidente)"
    filas_al.append((al, r, tipo))

esperables = [r for r in filas if r["set"] != "NEG" and r["snr_esp"] >= umbral and r["dmv"] >= dm_min
              and r["dmv"] <= min(dm_busqueda, dm_max or 1e9)]
perdidos = [r for r in esperables if r["name"] not in detectados]
lat = sorted(al["latencia_s"] for al in alertas)

o = [f"# Alertas de FRB — {D.name}", "",
     f"Regla: S/N ≥ {umbral}, DM {dm_min}–{dm_max or dm_busqueda:g}, veto si > {reg.get('max_beams', 8)} beams a la vez.", "",
     f"- Alertas: **{len(alertas)}** → FRB verdaderos {sum(t.startswith('**FRB') for _, _, t in filas_al)}, "
     f"no-FRB {sum(t.startswith('no-FRB') for _, _, t in filas_al)}, falsas alarmas {sum(t.startswith('falsa') for _, _, t in filas_al)}",
     f"- FRB inyectados que debían dar alerta (S/N esperado ≥ {umbral}, DM en rango): {len(esperables)}; "
     f"con alerta {len(esperables) - len(perdidos)}; sin alerta {len(perdidos)}"]
if lat:
    o.append(f"- Latencia (fin del pulso en el borde inferior → alerta): mediana {lat[len(lat) // 2]:.2f} s, "
             f"mín {lat[0]:.2f} s, máx {lat[-1]:.2f} s")
o += ["", "| alerta | beam | DM | S/N | llegada (UTC, 501,6 MHz) | latencia (s) | resultado | archivo | DM inyectado | S/N esperado |",
      "|---|---|---|---|---|---|---|---|---|---|"]
for al, r, tipo in filas_al:
    arch, dmi, snre = (r["name"], f"{r['dmv']:.1f}", f"{r['snr_esp']:.1f}") if r else ("—", "—", "—")
    o.append(f"| {al['n']} | {al['beam']} | {al['dm']} | {al['snr']} | {al['utc_llegada']} | {al['latencia_s']} | {tipo} | "
             f"{arch} | {dmi} | {snre} |")
if perdidos:
    o += ["", "FRB sin alerta:", "", "| archivo | beam | DM | S/N esperado |", "|---|---|---|---|"]
    o += [f"| {r['name']} | {r['b']} | {r['dmv']:.1f} | {r['snr_esp']:.1f} |" for r in perdidos]
(D / "alertas.md").write_text("\n".join(o) + "\n")
print("\n".join(o))
