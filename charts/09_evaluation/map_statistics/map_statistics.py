#!/usr/bin/env python
"""T3b · Estadística del mapa DM-tiempo de pirate: dónde se cruzan señal y ruido.

Usa la corrida completa del grouper v1 (T2), sin correr nada nuevo:
  eventos_v0.log   máximo por chunk y beam (casi todos los beam-chunks tienen máximo ≥ 5, así que es la
                   distribución completa del máximo por chunk sobre 5)
  sifter.log       eventos del v1 (hasta 4 por beam y chunk, S/N > 5)
  asignacion.csv   verdad

Análisis:
  1. Ruido: distribución del máximo por beam-chunk en beam-chunks sin señal (sin ningún trozo de diagonal de un
     archivo). Ajuste de N_eff (pruebas gaussianas independientes equivalentes): P(máx ≥ T) = 1 − (1 − Q(T))^N_eff.
  2. Ruido según DM y ancho: ¿los picos de ruido se reparten parejo en DM o se concentran?
  3. Señal: S/N medido por pirate frente al S/N esperado (inyectado) y recall según S/N esperado.
  4. Cruce: con qué S/N esperado un FRB queda en la zona del ruido.

Uso: python map_statistics.py [CARPETA_CORRIDA]   (escribe en resultados/ y figures/ junto a este script)
"""
import csv
import json
import re
import os
import sys
from pathlib import Path

import numpy as np
from scipy import stats, optimize
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker

AQUI = Path(__file__).resolve().parent
if len(sys.argv) < 2 and "CHARTS_RUN" not in os.environ:
    sys.exit(f"Uso: python {sys.argv[0]} CARPETA_DE_LA_CORRIDA   (o export CHARTS_RUN=...)")
RUN = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(os.environ["CHARTS_RUN"])   # corrida completa con eventos_v0.log
SALIDA = Path(os.environ.get("CHARTS_ANALYSIS_OUT", RUN / "analisis"))   # tablas y figuras
SALIDA.mkdir(parents=True, exist_ok=True)
DD = Path(__file__).resolve().parents[2] / "06_configs" / "charts_64b_dm1000.yml"
SEQ, DM_MIN, TOL_DM, TOL_T = 200, 20.0, 0.10, 0.05
T_CHUNK = 2.048
UMBRAL_OP = 6.50                                   # 10 FP ruido/h en T2
plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white", "font.size": 10})
(SALIDA).mkdir(exist_ok=True)
(SALIDA).mkdir(exist_ok=True)


def retardo_borde_a_top():
    import pirate_frb as pf
    cfg = pf.DedispersionConfig.from_yaml(str(DD))
    f_lo, f_hi = cfg.zone_freq_edges[0], cfg.zone_freq_edges[-1]
    s_por_dm_banda = 1e-3 / cfg.dm_per_unit_delay()
    dm_max = max(t.dm_max for t in pf.DedispersionPlan(cfg).trees)
    return s_por_dm_banda * (f_lo ** -2 - 501.6 ** -2) / (f_lo ** -2 - f_hi ** -2), s_por_dm_banda, dm_max


K_TOP, K_BANDA, DM_MAX = retardo_borde_a_top()
ev_re = re.compile(r"beam_id=(\d+), dm=([\d.e+-]+), fpga_timestamp=(-?\d+), width=([\d.e+-]+) ms.*snr=([\d.e+-]+)")
ch_re = re.compile(r"fpga=\[(-?\d+):(-?\d+)\]")


def leer_log(p, con_chunks=False):
    ev, chunks, ic = [], [], -1
    for linea in p.read_text(errors="replace").splitlines():
        m = ch_re.search(linea)
        if m:
            ic += 1
            chunks.append(int(m.group(1)) / SEQ * 1e-3)
            continue
        m = ev_re.search(linea)
        if m:
            ev.append((int(m.group(1)), float(m.group(2)), int(m.group(3)) / SEQ * 1e-3, float(m.group(4)),
                       float(m.group(5)), ic))
    ev = np.array(ev, dtype=float).reshape(-1, 6)
    return (ev, np.array(chunks)) if con_chunks else ev


# ---- verdad ----
filas = list(csv.DictReader(open(RUN / "asignacion.csv")))
emisor = json.loads((RUN / "emisor.json").read_text())
NB, PREPAD = emisor["nbeams"], emisor["prepad"] / 1000
for r in filas:
    r["b"] = int(r["beam"])
    r["t_abs"] = int(r["inicio_muestra"]) / 1000 + float(r["t_top_s"])
    r["dmv"] = float(r["dm_equivalente_pos"] if r["set"] == "NEG" else r["dm"])
    r["bar"] = float(r["barrido_s"])
    r["box"] = float(r.get("boxcar_opt_ms") or r.get("boxcar_opt_visible_ms") or 0) / 1000
    r["snr_esp"] = float(r.get("snr_visible_esperado") or r.get("snr_esperado") or "nan")
    r["t_ini"] = int(r["inicio_muestra"]) / 1000
    r["t_fin"] = r["t_ini"] + float(r["largo_s"])

# ---- 1. ruido: máximo por beam-chunk sin señal ----
ev0, chunk_t0 = leer_log(RUN / "eventos_v0.log", con_chunks=True)
nchunks = len(chunk_t0)
# Un beam-chunk es "sin señal" si su intervalo (tiempo en el borde inferior de la banda, que es como fecha
# pirate) no toca [inicio del archivo − 2 s, fin del archivo + barrido + 2 s] de ningún archivo del beam:
# así no entra ningún trozo de diagonal (a ningún DM de prueba) ni el relleno inicial.
limpio = np.ones((NB, nchunks), bool)
c0 = chunk_t0
c1 = chunk_t0 + T_CHUNK
limpio[:, c1 <= PREPAD + DM_MAX * K_BANDA] = False
for r in filas:
    lo, hi = r["t_ini"] - 2, r["t_fin"] + DM_MAX * K_BANDA + 2
    limpio[r["b"], (c1 > lo) & (c0 < hi)] = False
limpio[:, -2:] = False                              # últimos chunks (apagado)
maxchunk = np.full((NB, nchunks), np.nan)
info = {}
for b, dm, t, w, s, ic in ev0:
    if dm >= DM_MIN and 0 <= ic < nchunks:
        maxchunk[int(b), int(ic)] = s
        info[(int(b), int(ic))] = (dm, w)
sel = limpio & ~np.isnan(maxchunk)
n_limpios = int(limpio.sum())
m_ruido = maxchunk[sel]
falta = n_limpios - m_ruido.size                     # beam-chunks limpios sin máximo ≥ 5 (o con DM < 20)
print(f"beam-chunks sin señal: {n_limpios} ({n_limpios / limpio.size:.1%}); con máximo registrado: {m_ruido.size}")

T = np.round(np.arange(5.0, 8.01, 0.05), 2)
surv = np.array([(m_ruido >= u).sum() / n_limpios for u in T])
chunks_por_h = 3600 / T_CHUNK
fp_h = surv * chunks_por_h * NB


def p_max(u, logn):
    q = stats.norm.sf(u)
    return -np.expm1(10 ** logn * np.log1p(-q))


ok = (surv > 0) & (T >= 5.5)
res = optimize.minimize_scalar(lambda ln: np.sum((np.log(p_max(T[ok], ln)) - np.log(surv[ok])) ** 2),
                               bounds=(4, 12), method="bounded")
LOGN = res.x
print(f"N_eff ≈ 10^{LOGN:.2f} pruebas gaussianas por beam-chunk")

# ---- 2. ruido según DM y ancho ----
dm_r = np.array([info[(b, c)][0] for b, c in zip(*np.nonzero(sel))])
w_r = np.array([info[(b, c)][1] for b, c in zip(*np.nonzero(sel))])
bins_dm = np.linspace(DM_MIN, DM_MAX, 12)
idx = np.digitize(dm_r, bins_dm) - 1
por_dm = []
for k in range(len(bins_dm) - 1):
    mk = idx == k
    por_dm.append(dict(dm_lo=bins_dm[k], dm_hi=bins_dm[k + 1], n=int(mk.sum()),
                       frac=float(mk.mean()), frac_ge_6=float((m_ruido[mk] >= 6).mean()) if mk.any() else np.nan,
                       mediana=float(np.median(m_ruido[mk])) if mk.any() else np.nan))
anchos = np.unique(w_r)
por_ancho = [dict(ancho_ms=float(a), n=int((w_r == a).sum()), frac=float((w_r == a).mean()),
                  mediana=float(np.median(m_ruido[w_r == a]))) for a in anchos]
por_beam = np.array([np.nanmedian(np.where(sel[b], maxchunk[b], np.nan)) for b in range(NB)])

# ---- 3. señal: mejor S/N del v1 que acierta cada archivo ----
ev1 = leer_log(RUN / "sifter.log")
ev1 = ev1[ev1[:, 1] >= DM_MIN]
e_b, e_dm, e_s = ev1[:, 0].astype(int), ev1[:, 1], ev1[:, 4]
e_t = ev1[:, 2] - e_dm * K_TOP
mejor = np.full(len(filas), np.nan)
ancho_det = np.full(len(filas), np.nan)
for i, r in enumerate(filas):
    if r["set"] == "NEG":
        continue
    ok_ = (e_b == r["b"]) & (np.abs(e_dm - r["dmv"]) <= max(TOL_DM * r["dmv"], 1.0)) & \
          (np.abs(e_t - r["t_abs"]) <= max(TOL_T, r["box"]))
    mejor[i] = e_s[ok_].max() if ok_.any() else 0.0
    if ok_.any():
        ancho_det[i] = ev1[ok_, 3][np.argmax(e_s[ok_])]
sets = np.array([r["set"] for r in filas])
dmv = np.array([r["dmv"] for r in filas])
esp = np.array([r["snr_esp"] for r in filas])
usar = (sets != "NEG") & (dmv <= 1000) & np.isfinite(esp)

bins_s = np.array([0, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 30, 50, 1e9])
recall = []
for k in range(len(bins_s) - 1):
    m = usar & (esp >= bins_s[k]) & (esp < bins_s[k + 1])
    det = mejor[m]
    recall.append(dict(snr_esp_lo=bins_s[k], snr_esp_hi=bins_s[k + 1], n=int(m.sum()),
                       recall_5=float((det >= 5).mean()) if m.any() else np.nan,
                       recall_op=float((det >= UMBRAL_OP).mean()) if m.any() else np.nan,
                       mediana_medido=float(np.median(det[det > 0])) if (det > 0).any() else np.nan))
det = usar & (mejor > 0) & (esp >= 6) & (esp <= 30) & (sets != "P")
razon = mejor[det] / esp[det]

# umbrales para una tasa dada, con el modelo gaussiano ajustado (el medido no alcanza por estadística)
def umbral(fp_obj, nbeams):
    return float(optimize.brentq(lambda u: p_max(u, LOGN) * chunks_por_h * nbeams - fp_obj, 5, 10))


puntos = []
for nom, fp_obj, nb in (("10 FP por beam·hora (criterio de T1–T3)", 10, 1), ("10 FP/h del sistema (64 beams)", 10, NB),
                        ("1 FP/h del sistema (64 beams)", 1, NB)):
    u = umbral(fp_obj, nb)
    fila_p = {"criterio": nom, "umbral": round(u, 2)}
    for s_, n_ in (("A", "A"), ("B", "B"), ("P", "cortados")):
        m = (sets == s_) & (dmv <= 1000)
        fila_p[f"recall {n_}"] = round(100 * float((mejor[m] >= u).mean()), 1)
    puntos.append(fila_p)
w_frb = ancho_det[usar & (mejor >= UMBRAL_OP)]

# ---- tablas ----
resumen = {
    "corrida": str(RUN),
    "beam_chunks_sin_senal": n_limpios,
    "beam_chunks_sin_maximo_registrado": int(falta),
    "log10_N_eff": round(float(LOGN), 2),
    "maximo_ruido_percentiles": {p: round(float(np.percentile(m_ruido, p)), 2) for p in (50, 90, 99, 99.9)},
    "umbral_gauss_10FPh": round(float(optimize.brentq(lambda u: p_max(u, LOGN) * chunks_por_h * NB - 10, 5, 9)), 2),
    "umbral_medido_10FPh": float(T[np.flatnonzero(fp_h <= 10)[0]]) if (fp_h <= 10).any() else None,
    "razon_medido_esperado_mediana": round(float(np.median(razon)), 3),
    "razon_medido_esperado_p10_p90": [round(float(np.percentile(razon, p)), 3) for p in (10, 90)],
    "puntos_operacion": puntos,
    "por_dm": por_dm, "por_ancho": por_ancho, "recall_por_snr_esperado": recall,
    "mediana_por_beam_min_max": [round(float(por_beam.min()), 2), round(float(por_beam.max()), 2)],
}
(SALIDA / "metricas.json").write_text(json.dumps(resumen, indent=1, ensure_ascii=False))


def md(rows, cols, fmt):
    s = "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n"
    return s + "".join("| " + " | ".join(f(r) for f in fmt) + " |\n" for r in rows)


txt = "# T3b · tablas\n\n## Umbral y recall según el criterio de FP (modelo gaussiano ajustado)\n\n"
txt += md(puntos, list(puntos[0]), [lambda r, c=c: str(r[c]) for c in puntos[0]])
txt += "\n## Máximo de ruido por beam-chunk\n\n"
txt += md([dict(u=u, s=s, f=f) for u, s, f in zip(T, surv, fp_h) if abs(u * 4 - round(u * 4)) < 1e-6],
          ["S/N ≥", "P por beam-chunk", "FP/h del sistema (64 beams)", "gauss N_eff"],
          [lambda r: f"{r['u']:.2f}", lambda r: f"{r['s']:.2e}", lambda r: f"{r['f']:.1f}",
           lambda r: f"{p_max(r['u'], LOGN):.2e}"])
txt += "\n## Ruido según DM (máximos de beam-chunks sin señal)\n\n"
txt += md(por_dm, ["DM", "n", "fracción", "mediana S/N", "P(S/N ≥ 6)"],
          [lambda r: f"{r['dm_lo']:.0f}–{r['dm_hi']:.0f}", lambda r: str(r["n"]), lambda r: f"{r['frac']:.1%}",
           lambda r: f"{r['mediana']:.2f}", lambda r: f"{r['frac_ge_6']:.3f}"])
txt += "\n## Ruido y FRB detectados según ancho\n\n"
txt += md(por_ancho, ["ancho (ms)", "ruido n", "ruido fracción", "ruido mediana S/N", f"FRB detectados (S/N ≥ {UMBRAL_OP}) fracción"],
          [lambda r: f"{r['ancho_ms']:g}", lambda r: str(r["n"]), lambda r: f"{r['frac']:.1%}",
           lambda r: f"{r['mediana']:.2f}", lambda r: f"{(w_frb == r['ancho_ms']).mean():.1%}"])
txt += "\n## Recall según S/N esperado (A, B y cortados, DM ≤ 1000)\n\n"
txt += md(recall, ["S/N esperado", "n", "recall S/N ≥ 5", f"recall S/N ≥ {UMBRAL_OP}", "mediana S/N medido"],
          [lambda r: f"{r['snr_esp_lo']:g}–{r['snr_esp_hi']:g}" if r["snr_esp_hi"] < 1e8 else f"≥ {r['snr_esp_lo']:g}",
           lambda r: str(r["n"]), lambda r: f"{r['recall_5']:.1%}", lambda r: f"{r['recall_op']:.1%}",
           lambda r: f"{r['mediana_medido']:.2f}"])
(SALIDA / "tablas.md").write_text(txt)
print(json.dumps({k: v for k, v in resumen.items() if not isinstance(v, list)}, indent=1, ensure_ascii=False))
print(txt)

# ---- figuras ----
fig, axs = plt.subplots(1, 2, figsize=(12, 4.4))
ax = axs[0]
ax.hist(m_ruido, bins=np.arange(5, 8.5, 0.05), color="#3b528b", alpha=0.85, label="máximo por beam-chunk (ruido)")
ax.axvline(UMBRAL_OP, color="black", ls="--", lw=1)
ax.text(UMBRAL_OP + 0.05, ax.get_ylim()[1] * 0.9, "6,5: 10 por beam·hora", fontsize=9)
u_sis = umbral(10, NB)
ax.axvline(u_sis, color="black", ls=":", lw=1)
ax.text(u_sis + 0.05, ax.get_ylim()[1] * 0.75, f"{u_sis:.2f}: 10 por hora del sistema".replace(".", ","), fontsize=9)
ax.set_xlabel("S/N")
ax.set_ylabel("beam-chunks")
ax.set_title(f"Ruido: máximo por beam-chunk ({n_limpios} beam-chunks sin señal)")
ax = axs[1]
ax.semilogy(T, fp_h, "o", ms=4, color="#21918c", label="medido")
tt = np.linspace(5, 8, 200)
for ln, ls in ((LOGN, "-"), (LOGN - 1, ":"), (LOGN + 1, ":")):
    ax.semilogy(tt, p_max(tt, ln) * chunks_por_h * NB, ls, color="#440154",
                label=f"gaussiano, N_eff = 10^{ln:.1f}" if ln == LOGN else (f"10^{ln:.1f}"))
ax.axhline(10, color="gray", lw=0.8)
ax.axhline(10 * NB, color="gray", lw=0.8, ls="--")
ax.text(5.05, 10 * NB * 1.3, "10 por beam·hora", fontsize=8, color="gray")
ax.text(5.05, 13, "10 por hora del sistema", fontsize=8, color="gray")
ax.set_ylim(0.3, 3e5)
ax.set_xlabel("umbral S/N")
ax.set_ylabel("FP por ruido por hora, sistema de 64 beams")
ax.set_title("Tasa de falsas alarmas frente a umbral")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(SALIDA / "1_ruido_maximo_por_chunk.png", dpi=130)
plt.close(fig)

fig, axs = plt.subplots(1, 3, figsize=(14, 4))
centros = [(r["dm_lo"] + r["dm_hi"]) / 2 for r in por_dm]
axs[0].bar(centros, [r["frac"] for r in por_dm], width=np.diff(bins_dm)[0] * 0.9, color="#3b528b")
axs[0].axhline(1 / (len(bins_dm) - 1), color="black", ls="--", lw=1, label="reparto parejo")
axs[0].set_xlabel("DM del máximo de ruido")
axs[0].set_ylabel("fracción de beam-chunks")
axs[0].legend(fontsize=8)
axs[0].set_title("¿Dónde cae el máximo de ruido en DM?")
axs[1].plot(centros, [r["frac_ge_6"] for r in por_dm], "o-", color="#21918c")
axs[1].set_xlabel("DM")
axs[1].set_ylabel("P(máximo ≥ 6 | DM)")
axs[1].set_title("Cola del ruido según DM")
xa = np.arange(len(por_ancho))
axs[2].bar(xa - 0.2, [r["frac"] for r in por_ancho], width=0.4, color="#3b528b", label="ruido")
axs[2].bar(xa + 0.2, [(w_frb == r["ancho_ms"]).mean() for r in por_ancho], width=0.4, color="#fde725",
           edgecolor="0.4", lw=0.3, label="FRB detectados")
axs[2].set_xticks(xa, [f"{r['ancho_ms']:g}" for r in por_ancho])
axs[2].set_xlabel("ancho (ms)")
axs[2].set_ylabel("fracción")
axs[2].legend(fontsize=8)
axs[2].set_title("Ruido y FRB según ancho")
fig.tight_layout()
fig.savefig(SALIDA / "2_ruido_dm_ancho.png", dpi=130)
plt.close(fig)

fig, axs = plt.subplots(1, 2, figsize=(12, 4.6))
ax = axs[0]
colores = {"A": "#440154", "B": "#21918c", "P": "#fde725"}
for s in ("B", "A", "P"):
    m = usar & (sets == s)
    y = np.where(mejor[m] > 0, mejor[m], 4.2)
    ax.scatter(esp[m], y, s=9, alpha=0.6, color=colores[s], edgecolor="none" if s != "P" else "0.4",
               linewidths=0.3, label={"A": "A (S/N 8)", "B": "B", "P": "cortados (S/N visible)"}[s])
xx = np.array([1, 300])
ax.plot(xx, xx, color="black", lw=0.8, label="medido = esperado")
ax.axhline(UMBRAL_OP, color="gray", ls="--", lw=1)
ax.axhspan(4, 4.4, color="0.9")
ax.text(1.1, 4.25, "no detectado", fontsize=8, va="center")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(1, 300)
ax.set_ylim(4, 300)
ax.set_xlabel("S/N esperado (inyectado)")
ax.set_ylabel("mejor S/N medido por pirate (v1)")
ax.set_title(f"Medido / esperado: mediana {np.median(razon):.2f}")
ax.legend(fontsize=8, loc="upper left")
ax = axs[1]
xc = [np.sqrt(max(r["snr_esp_lo"], 2) * min(r["snr_esp_hi"], 100)) for r in recall]
ax.semilogx(xc, [100 * r["recall_5"] for r in recall], "o-", color="#3b528b", label="S/N ≥ 5")
ax.semilogx(xc, [100 * r["recall_op"] for r in recall], "o-", color="#21918c", label=f"S/N ≥ {UMBRAL_OP} (10 FP por beam·hora)")
ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
ax.set_xticks([3, 5, 7, 10, 15, 20, 30, 50, 70])
ax.set_xlabel("S/N esperado (por tramo)")
ax.set_ylabel("recall (%)")
ax.set_ylim(0, 102)
ax.set_title("Recall según S/N esperado (A + B + cortados, DM ≤ 1000)")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(SALIDA / "3_senal_medido_vs_esperado.png", dpi=130)
plt.close(fig)

fig, ax = plt.subplots(figsize=(7.5, 4.4))
ax.hist(m_ruido, bins=np.arange(4, 16, 0.1), density=True, color="#3b528b", alpha=0.6, label="ruido: máximo por beam-chunk")
d8 = mejor[usar & (sets == "A") & (mejor > 0)]
ax.hist(d8, bins=np.arange(4, 16, 0.1), density=True, color="#fde725", alpha=0.7, edgecolor="0.4", lw=0.3,
        label="FRB del set A (S/N 8 inyectado), medido")
ax.axvline(UMBRAL_OP, color="black", ls="--", lw=1)
ax.set_xlabel("S/N medido")
ax.set_ylabel("densidad")
ax.set_title("Cruce entre ruido y un FRB de S/N 8")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(SALIDA / "4_cruce_ruido_senal.png", dpi=130)
plt.close(fig)
