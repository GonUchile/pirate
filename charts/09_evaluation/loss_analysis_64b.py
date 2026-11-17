#!/usr/bin/env python3
"""Dónde se pierden los FRBs en la prueba de 64 beams: recall por S/N inyectado, ancho, scattering, duración y DM.

Fuentes (de evaluate_test_64b.py, sin volver a cruzar eventos):
  CORRIDA/asignacion.csv            parámetros de cada archivo inyectado (una fila por archivo, mismo orden)
  CORRIDA/mejor_snr_por_archivo.npy mejor S/N de un evento que acierta cada archivo (-inf si ninguno):
                                    mismo beam, DM ±10 %, |Δt| ≤ 50 ms (NEG: cualquier DM a ±100 ms)

Definiciones
  detectado        mejor S/N ≥ UMBRAL (6,54 = 10 FP de ruido por hora en esta corrida, ver resultados.md)
  universo         positivos A, B y cortados (P) con DM ≤ 1000, como en resultados.md
  S/N inyectado    A y B: snr_esperado (boxcar óptimo, banda completa); P: snr_visible_esperado (parte visible)
  duración         boxcar óptimo del burst ya dedispersado (boxcar_opt_ms; P: boxcar_opt_visible_ms):
                   ancho intrínseco + scattering + smearing; pirate prueba boxcars de hasta 32 ms
  scattering       scat_ref_s, a 400,2 MHz (a 300 MHz es 3,2 veces más)
  FP no-FRB        % de archivos NEG (mismo DM equivalente, ancho y scattering que su par) con evento ≥ umbral
  causa probable   (en orden) S/N inyectado < umbral → «débil»; hay evento ≥ umbral en la ventana de tiempo pero
                   a DM fuera de ±10 % → «DM desplazado»; hay evento a DM correcto pero con S/N < umbral →
                   «bajo el umbral»; si no → «sin evento cercano» (de nearby_events_64b.py, entorno pirate)

Uso: environments/simulador_frb/bin/python loss_analysis_64b.py [CORRIDA] [--umbral 6.54 --fp-h 8,9]
Salida en CORRIDA/analisis_por_parametro/: tabla_por_archivo.csv, recall_por_bin.csv, causas.csv,
recall_por_parametro.png, snr_medido_vs_duracion.png
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

AQUI = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument("corrida", type=Path)
ap.add_argument("--umbral", type=float, default=6.54)
ap.add_argument("--dm-max", type=float, default=1000.0)
ap.add_argument("--fp-h", default="8,9", help="FP de ruido por hora a ese umbral (de resultados.md), solo para el título")
a = ap.parse_args()
OUT = a.corrida / "analisis_por_parametro"
OUT.mkdir(exist_ok=True)
U, BOX_MAX = a.umbral, 32.0

d = pd.read_csv(a.corrida / "asignacion.csv")
d["mejor"] = np.load(a.corrida / "mejor_snr_por_archivo.npy")
assert len(d) == len(d["mejor"]), "asignacion.csv y mejor_snr_por_archivo.npy no tienen el mismo largo"
d["det"] = d["mejor"] >= U
d["dm_v"] = np.where(d["set"] == "NEG", d["dm_equivalente_pos"], d["dm"])
d["snr_iny"] = np.where(d["set"] == "P", d["snr_visible_esperado"], d["snr_esperado"])
d["dur_ms"] = np.where(d["set"] == "P", d["boxcar_opt_visible_ms"], d["boxcar_opt_ms"])
d["ancho_ms"] = d["width_s"] * 1e3
d["scat_ms"] = d["scat_ref_s"] * 1e3
# NEG: duración del par positivo (mismos ancho, scattering y DM equivalente), si existe
par = d.set_index("name")["dur_ms"]
m_neg = (d["set"] == "NEG") & d["par_de"].notna()
d.loc[m_neg, "dur_ms"] = d.loc[m_neg, "par_de"].map(par).values

u = d[d["dm_v"] <= a.dm_max].copy()
pos = u[u["set"].isin(["A", "B", "P"])].copy()
neg = u[u["set"] == "NEG"]
CAUSAS = ["débil: S/N inyectado < umbral", "DM desplazado > 10 % (evento ≥ umbral a otro DM)",
          "DM correcto, S/N medido 5–6,5 (bajo el umbral)", "sin evento a su DM (ni bajo el umbral)"]
ec = a.corrida / "analisis_por_parametro" / "eventos_cercanos.csv"
if not ec.exists():
    raise SystemExit(f"falta {ec}: correr antes environments/pirate/bin/python nearby_events_64b.py {a.corrida}")
pos = pos.merge(pd.read_csv(ec), on="name", how="left")
pos["causa"] = np.select([pos["snr_iny"] < U, pos["snr_otro_dm"] >= U, pos["snr_dm_ok"].notna()], CAUSAS[:3], CAUSAS[3])
pos.loc[pos["det"], "causa"] = "detectado"
u.to_csv(OUT / "tabla_por_archivo.csv", index=False)
print(f"fuente: {a.corrida}  umbral {U}  DM ≤ {a.dm_max:.0f}: " + ", ".join(f"{s} {int((u.set == s).sum())}" for s in ("A", "B", "P", "NEG")))

NOM = {"A": "A · S/N 8", "B": "B · flujo fijo", "P": "cortados"}
COL = {"A": plt.cm.viridis(0.15), "B": plt.cm.viridis(0.5), "P": plt.cm.viridis(0.82)}
BINS = {
    "snr_iny": ([0, 4, 6.54, 8.01, 12, 20, 40, 1e4], "S/N inyectado (boxcar óptimo)", ["<4", "4–6,5", "6,5–8", "8–12", "12–20", "20–40", ">40"]),
    "dur_ms": ([0, 2, 4, 8, 16, 32, 64, 1e5], "duración efectiva del burst (ms)", ["≤2", "2–4", "4–8", "8–16", "16–32", "32–64", ">64"]),
    "ancho_ms": ([0, 0.3, 1, 3, 10, 100], "ancho intrínseco (ms)", ["<0,3", "0,3–1", "1–3", "3–10", ">10"]),
    "scat_ms": ([-1, 0.01, 0.1, 1, 10, 1e5], "scattering a 400 MHz (ms)", ["<0,01", "0,01–0,1", "0,1–1", "1–10", ">10"]),
    "dm_v": ([0, 250, 500, 750, 1000.01], "DM (pc cm⁻³)", ["≤250", "250–500", "500–750", "750–1000"]),
}

filas = []
for var, (b, _, et) in BINS.items():
    for s in ("A", "B", "P", "NEG"):
        g = u[u["set"] == s]
        if g[var].isna().all():
            continue
        k = pd.cut(g[var], b, labels=et, right=False)
        for e in et:
            gg = g[k == e]
            filas.append(dict(variable=var, bin=e, set=s, n=len(gg), detectados=int(gg["det"].sum()),
                              pct=round(100 * gg["det"].mean(), 1) if len(gg) else np.nan))
rb = pd.DataFrame(filas)
rb.to_csv(OUT / "recall_por_bin.csv", index=False)

# Figura 1: recall por parámetro (un panel por variable), NEG en gris = FP no-FRB del mismo bin
fig, axs = plt.subplots(2, 3, figsize=(17, 9.5))
for ax, (var, (b, xl, et)) in zip(axs.flat, BINS.items()):
    x = np.arange(len(et))
    sets = [s for s in ("A", "B", "P") if not rb[(rb.variable == var) & (rb.set == s)].empty]
    w = 0.8 / (len(sets) + 1)
    for j, s in enumerate(sets + ["NEG"]):
        r = rb[(rb.variable == var) & (rb.set == s)].set_index("bin").reindex(et)
        if r["n"].isna().all():
            continue
        xs = x + (j - len(sets) / 2) * w
        y = r["pct"].values.astype(float)
        c = "0.6" if s == "NEG" else COL[s]
        ax.bar(xs, np.nan_to_num(y), w * 0.92, color=c, label=("NEG: FP no-FRB" if s == "NEG" else NOM[s]),
               hatch="//" if s == "NEG" else None, edgecolor="white" if s != "NEG" else "0.4", linewidth=0.5)
        for xi, yi, ni in zip(xs, y, r["n"].values):
            if ni and ni > 0:
                ax.text(xi, (0 if np.isnan(yi) else yi) + 1.5, f"{int(ni)}", ha="center", va="bottom", fontsize=7, color="0.35", rotation=90)
    ax.set_xticks(x, et)
    ax.set_ylim(0, 112)
    ax.set_xlabel(xl)
    ax.set_ylabel("% con evento ≥ umbral")
    ax.grid(axis="y", alpha=0.3)
    if var == "dur_ms":
        ax.axvline(4.5, color="#C2551F", ls="--", lw=1.2)
        ax.text(4.55, 104, "boxcar máx.\nde pirate (32 ms)", color="#C2551F", fontsize=9, va="top")
    if var == "snr_iny":
        ax.axvline(1.5, color="#C2551F", ls="--", lw=1.2)
        ax.text(1.55, 104, f"umbral {U}".replace(".", ","), color="#C2551F", fontsize=9, va="top")
from matplotlib.patches import Patch
h0, _ = axs.flat[0].get_legend_handles_labels()
axs.flat[0].legend(handles=h0 + [Patch(fc="0.6", ec="0.4", hatch="//", label="NEG: FP no-FRB (otros paneles)")], loc="upper left", fontsize=9, bbox_to_anchor=(0, 0.92))

# último panel: causas de pérdida por set
ax = axs.flat[5]
causas = CAUSAS
cc = dict(zip(CAUSAS, [plt.cm.viridis(0.95), plt.cm.viridis(0.6), plt.cm.viridis(0.35), "0.45"]))
tab = pos.groupby(["set", "causa"]).size().unstack(fill_value=0).reindex(["A", "B", "P"])
izq = np.zeros(3)
for c in causas:
    v = tab.get(c, pd.Series(0, index=tab.index)).values
    ax.barh(np.arange(3), v, left=izq, color=cc[c], label=c, edgecolor="white")
    for i, (vi, li) in enumerate(zip(v, izq)):
        if vi >= 5:
            ax.text(li + vi / 2, i, str(vi), ha="center", va="center", color="black" if c == CAUSAS[0] else "white", fontsize=10)
    izq += v
for i, s in enumerate(["A", "B", "P"]):
    n, det = int((pos.set == s).sum()), int(pos[pos.set == s]["det"].sum())
    ax.text(izq[i] + 1, i, f"{n - det} perdidos de {n}", va="center", fontsize=9)
ax.set_yticks(range(3), [NOM[s] for s in ["A", "B", "P"]])
ax.invert_yaxis()
ax.set_xlim(0, max(izq) * 1.6 + 5)
ax.set_xlabel("FRBs perdidos (sin evento ≥ umbral)")
ax.set_title("Causa probable de cada pérdida", fontsize=11, loc="left")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), fontsize=8.5, frameon=False, ncol=1)
fp_h = a.fp_h
fig.suptitle(f"Prueba de 64 beams en tiempo real (grupo 3, sin RFI, DM ≤ {a.dm_max:.0f}): % de archivos con evento, umbral S/N {U} "
             f"(= {fp_h} FP de ruido por hora) · número = archivos en el bin".replace(".", ",", 1), fontsize=12.5, x=0.01, ha="left")
fig.tight_layout(rect=(0, 0, 1, 0.96))
fig.savefig(OUT / "recall_por_parametro.png", dpi=110, facecolor="white")
plt.close(fig)

# Figura 2: S/N medido / inyectado contra duración; perdidos en una franja aparte
fig, ax = plt.subplots(figsize=(11, 7.5))
for s in ("A", "B", "P"):
    g = pos[pos.set == s]
    dt = g[g.det]
    ax.scatter(dt["dur_ms"], dt["mejor"] / dt["snr_iny"], s=16, color=COL[s], alpha=0.75, label=f"{NOM[s]}: detectados ({len(dt)})")
    pr = g[~g.det]
    yj = 0.03 * np.exp(np.random.default_rng(1).uniform(-0.25, 0.25, len(pr)))
    ax.scatter(pr["dur_ms"], yj, s=22, marker="x", color=COL[s], alpha=0.9, label=f"{NOM[s]}: perdidos ({len(pr)})")
ax.axvline(BOX_MAX, color="#C2551F", ls="--")
ax.text(BOX_MAX * 1.05, 2.2, "boxcar máximo\nde pirate (32 ms)", color="#C2551F", fontsize=10, va="top")
ax.axhline(1, color="0.4", lw=0.8)
ax.axhspan(0.02, 0.045, color="0.92", zorder=0)
ax.text(1.05, 0.047, "perdidos (sin evento ≥ umbral; sin valor de S/N)", fontsize=9, color="0.35")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_ylim(0.02, 2.5)
ax.set_xlabel("duración efectiva del burst (boxcar óptimo, ms)")
ax.set_ylabel("S/N medido por pirate / S/N inyectado")
ax.set_title(f"Cuánto S/N recupera pirate según la duración del burst (DM ≤ {a.dm_max:.0f})", loc="left")
ax.grid(alpha=0.3, which="both")
ax.legend(fontsize=9, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.11), frameon=False)
fig.tight_layout()
fig.savefig(OUT / "snr_medido_vs_duracion.png", dpi=110, facecolor="white")
plt.close(fig)

# Resúmenes en consola
c = pos.groupby(["set", "causa"]).size().unstack(fill_value=0)
c.to_csv(OUT / "causas.csv")
print("\nCausa probable por set (DM ≤ %.0f):\n%s" % (a.dm_max, c.to_string()))
print(f"\nRecall: " + ", ".join(f"{s} {pos[pos.set == s]['det'].mean() * 100:.1f} % ({int(pos[pos.set == s]['det'].sum())}/{int((pos.set == s).sum())})" for s in ("A", "B", "P")))
lib = pos["det"] | (pos["snr_otro_dm"] >= U)
print("Recall sin exigir DM ±10 %: " + ", ".join(f"{s} {lib[pos.set == s].mean() * 100:.1f} %" for s in ("A", "B", "P")))
print(f"FP no-FRB (NEG): {neg['det'].mean() * 100:.1f} % ({int(neg['det'].sum())}/{len(neg)})")
dt = pos[pos.det]
for lo, hi in ((0, 8), (8, 32), (32, 1e9)):
    m = dt[(dt.dur_ms > lo) & (dt.dur_ms <= hi)]
    if len(m):
        r = m["mejor"] / m["snr_iny"]
        print(f"S/N medido/inyectado, duración {lo}–{hi:.0f} ms: mediana {r.median():.2f}  p10 {r.quantile(.1):.2f}  p90 {r.quantile(.9):.2f}  (n {len(m)})")
print("\n" + rb.pivot_table(index=["variable", "bin"], columns="set", values=["n", "pct"], sort=False).to_string())
print("\nsalida:", OUT)
