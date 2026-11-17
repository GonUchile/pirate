#!/usr/bin/env python3
"""
Set de comparación de pipelines (diseño acordado en el equipo CHARTS, 2026-09-23).

El set de outputs/ hereda el flujo del catálogo CHIME (la amplitud no está
controlada), busca DM solo hasta 1000 (14% de los positivos queda fuera) y
no incluye el ensanchamiento por dispersión dentro del canal. Este set
controla la amplitud explícitamente (ver outputs_comparacion/README.md):

  A_snr_fijo/        N FRB, todos con el MISMO SNR esperado (boxcar
                     óptimo, dedispersado al DM real). DM, ancho,
                     scattering y espectro distintos en cada uno.
  B_amplitud_fija/   Los MISMOS N sorteos de parámetros que A, pero todos
                     con la misma amplitud S0 (Jy): flujo de pico intrínseco
                     (antes de scattering y smearing) en el canal más
                     brillante de la banda CHARTS. El SNR sale de la física.
                     S0 se calibra para que la mediana del SNR quede en
                     --snr-b-mediana.
  negativos/         N no-FRB, pareados con A: mismos parámetros y misma
                     escala que A_i, pero con dm_index inválido (|idx| en
                     [1.5,1.8) o (2.2,2.5], como 02_generar_negativos.py) y
                     el DM reescalado para que el barrido cruce la banda en
                     el mismo tiempo que el positivo (no se sale del archivo).
  parcial/           N_parcial FRB cortados por el borde del archivo: el
                     barrido empieza antes del inicio ("inicio", solo queda
                     la parte baja de la banda) o termina después del final
                     ("fin", solo queda la parte alta). Se sortea la
                     fracción de energía visible (uniforme 5-95%) y se ubica
                     el corte para lograrla. El burst completo se
                     normaliza a --snr-parcial; el manifest guarda la banda
                     visible, la fracción de energía dentro y el SNR
                     esperado de la parte visible.

Todo con ruido gaussiano del radiómetro de CHARTS (SEFD de 256 antenas,
n_pol=1, engine/charts_sefd.py), sin RFI, cuantizado a 8 bits con el ruido
del sistema en PISO_NIVELES y sigma = NIVELES_POR_SIGMA niveles (ver
escribir()). Es la escala acordada para el dataset base al que después se
le suma RFI como potencia (sube la media): deja ~56 sigma de margen arriba
y 8 sigma abajo.

Diferencias físicas respecto de 01_/02_ (a propósito):
  - Ensanchamiento por dispersión DENTRO de cada canal (0.3 MHz): fitburst
    no lo aplica en SpectrumModeler (compute_time_dm_smear existe pero no
    se usa). En CHARTS no es despreciable: ~92 ms a 300 MHz con DM 1000,
    ~277 ms con DM 3000. Se aplica como boxcar de ancho
    |idx|·DM·K·Δν·f^(idx-1) por canal.
  - Espectro normalizado al canal más brillante de la banda y scattering
    con el PBF normalizado (conserva energía); ver modelo().
  - Integración dentro de la muestra de 1 ms (sobremuestreo x4), para que
    bursts de ~0.2 ms no dependan de la fase de muestreo.
  - Archivos de largo fijo (NSAMP) para todos los sets, búsqueda de DM
    20-3000 y DM sorteado uniforme en 100-3000 (no la distribución del
    catálogo, que deja ~14% sobre 1000). Ancho, scattering y espectro se
    toman juntos de un FRB del catálogo al azar.

El perfil por canal usa las mismas rutinas de fitburst que
SpectrumModeler.compute_model (compute_profile_pbf/gaussian, el running
power law de compute_spectrum_rpl, compute_time_dm_delay), pero evaluadas
solo en una ventana local alrededor de la llegada de cada canal.

No se guardan .npz (serían ~320 MB por archivo): cada .fil se reproduce
desde parametros_sorteados.csv + la semilla derivada de su nombre.

Salida (--out, default outputs_comparacion/):
  config.json, parametros_sorteados.csv
  A_snr_fijo/*.fil + manifest.csv, B_amplitud_fija/..., negativos/..., parcial/...

Uso:
  python 07_generate_comparison_set.py [--n 1000] [--n-parcial 1000]
      [--snr-a 8] [--snr-b-mediana 10] [--snr-parcial 15]
      [--workers 8] [--seed 20260923] [--out outputs_comparacion]
"""
import argparse
import csv
import hashlib
import importlib.util
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.stats import norm
from fitburst.routines import ism as fb_ism
from fitburst.routines import profile as fb_profile

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "engine"))

import charts_sefd  # noqa: E402
from fitburst_sim import DM_CONST_SECONDS, DM_INDEX, REF_FREQ, SCATTERING_INDEX  # noqa: E402

# write_fil de 03_ (mismo header SIGPROC y misma cuantización fija que el set de outputs/)
_spec = importlib.util.spec_from_file_location("convertir_a_fil", HERE / "03_convert_to_fil.py")
_conv = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_conv)
write_fil = _conv.write_fil

# Cuantización (acordada en el análisis del modelo de RFI): el piso del
# ruido del sistema en el nivel 32 y sigma = 4 niveles. Margen de
# (255-32)/4 ~ 56 sigma arriba, para sumar después RFI como potencia (sube la
# media) y para bursts brillantes en pocos canales; 8 sigma abajo. El ruido
# de cuantización agrega 1/(12·4²) = 0.5% de varianza (~0.26% de SNR).
PISO_NIVELES = 32
NIVELES_POR_SIGMA = 4

CATALOGOS_DIR = HERE / "catalogs"
EXCLUIR = {"FRB20200302A", "FRB20200427A"}  # sp_idx/sp_run inválidos, ver 01_

# ---- Instrumento (mismo que 01_: banda CHARTS, 672 canales, 1 ms) ----
FREQ_LO = 300.0
NUM_FREQ = 672
FREQ_HI = FREQ_LO + NUM_FREQ * 0.3  # 501.6 MHz
FREQS = np.linspace(FREQ_LO, FREQ_HI, NUM_FREQ)
RES_FREQ_MHZ = (FREQ_HI - FREQ_LO) / (NUM_FREQ - 1)
TSAMP = 1e-3
SEFD_JY = charts_sefd.calcular_sefd_charts(256)
N_POL = charts_sefd.CHARTS_N_POL
SIGMA_CH = SEFD_JY / np.sqrt(N_POL * RES_FREQ_MHZ * 1e6 * TSAMP)  # Jy, igual para los 672 canales

# ---- Diseño del set ----
DM_SORTEO = (100.0, 3000.0)
DM_SEARCH = (20.0, 3000.0)
T_TOP_RANGO = (5.0, 15.0)   # llegada al canal más alto (501.6 MHz), segundos desde el inicio del archivo
GUARD_S = 10.0              # mismo margen que PAD_GUARD_SAMPLES de 03_ (redondeo de prepsubband)
DM_INDEX_MAG_BAJO = (1.5, 1.8)
DM_INDEX_MAG_ALTO = (2.2, 2.5)
ANCHO_MIN_S = 1e-4          # con sobremuestreo x4 (0.25 ms), error de muestreo <5% para sigma >= 0.1 ms
SOBREMUESTREO_T = 4
BOXCAR_MAX = 4096           # muestras; igual al boxcar máximo por defecto de Heimdall
FRAC_VISIBLE_RANGO = (0.05, 0.95)  # subset parcial: fracción de energía del burst dentro del archivo
# Espectros del catálogo con menos canales que esto sobre el 50% del pico en la
# banda se descartan. En 0 A PROPÓSITO (equipo CHARTS, 2026-09-24): los ajustes
# extremos (sp_idx ~1e3) que meten toda la energía en 1-4 canales saturan la
# escala de 8 bits (hasta 47% de pérdida de señal con SNR 8) y se quieren ver
# y entender. Quedan identificados en el manifest con n_canales_emision y
# perdida_recorte_pct.
MIN_CANALES_EMISION = 0


def delay_s(dm, idx, f):
    """Retardo relativo a REF_FREQ, misma rutina que usa fitburst."""
    return fb_ism.compute_time_dm_delay(dm, DM_CONST_SECONDS, idx, np.asarray(f, dtype=float), freq2=REF_FREQ)


def barrido_s(dm, idx):
    return float(delay_s(dm, idx, FREQ_LO) - delay_s(dm, idx, FREQ_HI))


def smear_s(dm, idx, f):
    """Ensanchamiento dentro de un canal: |d retardo/df| · Δν."""
    return abs(idx) * dm * DM_CONST_SECONDS * RES_FREQ_MHZ * np.asarray(f, dtype=float) ** (idx - 1)


NSAMP = int(np.ceil((T_TOP_RANGO[1] + barrido_s(DM_SEARCH[1], DM_INDEX) + GUARD_S + 5.0) / 1000 / TSAMP)) * 1000


def semilla(base, nombre):
    return int(hashlib.sha256(f"{base}:{nombre}".encode()).hexdigest()[:8], 16)


# ============================================================================
# Catálogo: pool de formas (ancho, scattering, espectro) de FRB reales
# ============================================================================
def _float(v):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if np.isfinite(x) else None


def n_canales_emision(sp_idx, sp_run):
    """Canales de la banda con el espectro sobre el 50% de su máximo en la banda."""
    x = np.log(FREQS / REF_FREQ)
    l = sp_idx * x + sp_run * x ** 2
    return int((l - l.max() >= np.log(0.5)).sum())


def cargar_pool():
    """Ancho/scattering/espectro de catalog1 + catalog2 con el mismo filtro
    que 01_ (espectro con pico, sp_run<0; sin excluidos/sidelóbulo), sin el
    corte por SNR predicha: acá la amplitud la fija el diseño, no el catálogo."""
    pool = {}
    with open(CATALOGOS_DIR / "catalog1_clean.csv") as f:
        for r in csv.DictReader(f):
            w, s = _float(r["width_fitb_s"]), _float(r["scat_s"])
            idx, run = _float(r["sp_idx"]), _float(r["sp_run"])
            if None in (w, idx, run) or run >= 0 or w < ANCHO_MIN_S or r["name"] in EXCLUIR:
                continue
            pool[r["name"]] = ("catalog1", w, max(s or 0.0, 0.0), idx, run)
    with open(CATALOGOS_DIR / "chimefrbcat2.csv") as f:
        for r in csv.DictReader(f):
            if r["excluded_flag"] != "0" or r["sidelobe_flag"] != "0" or r["tns_name"] in EXCLUIR:
                continue
            w, s = _float(r["width_fitb"]), _float(r["scat_time"])
            idx, run = _float(r["sp_idx"]), _float(r["sp_run"])
            if None in (w, idx, run) or run >= 0 or w < ANCHO_MIN_S:
                continue
            pool[r["tns_name"]] = ("catalog2", w, max(s or 0.0, 0.0), idx, run)  # catalog2 pisa a catalog1; scat<0 del catálogo -> 0
    return sorted((k,) + v for k, v in pool.items() if n_canales_emision(v[3], v[4]) >= MIN_CANALES_EMISION)


def sortear(n, rng, pool, prefijo):
    filas = []
    for i in range(n):
        nombre, cat, w, s600, sp_idx, sp_run = pool[rng.integers(len(pool))]
        filas.append({
            "id": f"{prefijo}{i:03d}",
            "forma_origen": nombre, "catalogo_origen": cat,
            "dm": float(rng.uniform(*DM_SORTEO)),
            "width_s": w,
            "scat_600_s": s600,
            # misma conversión 600 MHz -> REF_FREQ que engine/fitburst_sim.py
            "scat_ref_s": s600 * (600.0 / REF_FREQ) ** (-SCATTERING_INDEX),
            "sp_idx": sp_idx, "sp_run": sp_run,
            "n_canales_emision": n_canales_emision(sp_idx, sp_run),
            "t_top_s": float(rng.uniform(*T_TOP_RANGO)),
        })
    return filas


def sortear_dm_index(rng):
    rango = DM_INDEX_MAG_BAJO if rng.random() < 0.5 else DM_INDEX_MAG_ALTO
    return -float(rng.uniform(*rango))


# ============================================================================
# Modelo por canal
# ============================================================================
def _media_movil(x, m):
    """Boxcar centrado de m muestras que conserva la suma."""
    if m <= 1:
        return x
    xp = np.concatenate([np.zeros(m // 2), x, np.zeros(m - 1 - m // 2)])
    c = np.concatenate([[0.0], np.cumsum(xp)])
    return (c[m:] - c[:-m]) / m


def modelo(dm, idx, width_s, scat_ref_s, sp_idx, sp_run, t_top_s):
    """Señal de amplitud unitaria (1 Jy de pico intrínseco en el canal más
    brillante), canal por canal.

    Devuelve (canales, t_c): canales[c] = (i0, valores) en muestras
    absolutas del archivo (i0 puede ser <0 y el final puede pasar NSAMP,
    para los bursts parciales); t_c = llegada de cada canal en segundos.
    """
    arrival_ref = t_top_s - float(delay_s(dm, idx, FREQS[-1]))
    t_c = arrival_ref + delay_s(dm, idx, FREQS)
    # Mismo running power law que fb_spectrum.compute_spectrum_rpl, pero
    # normalizado al canal más brillante de la banda (no a REF_FREQ): con
    # espectros angostos del catálogo (ej. sp_idx=1134, sp_run=-1090) el
    # valor en REF_FREQ difiere del pico por factores de hasta ~1e85.
    x = np.log(FREQS / REF_FREQ)
    log_esp = sp_idx * x + sp_run * x ** 2
    espectro = np.exp(log_esp - log_esp.max())
    smear = smear_s(dm, idx, FREQS)
    tau = scat_ref_s * (FREQS / REF_FREQ) ** SCATTERING_INDEX
    U = SOBREMUESTREO_T
    canales = []
    for c in range(NUM_FREQ):
        cola = min(10.0 * tau[c], 200.0)
        i0 = int(np.floor((t_c[c] - 6 * width_s - smear[c] / 2) / TSAMP)) - 1
        i1 = int(np.ceil((t_c[c] + 6 * width_s + cola + smear[c] / 2) / TSAMP)) + 2
        # muestra k integra [k·T - T/2, k·T + T/2) (centrada, como el muestreo puntual de fitburst)
        tt = (i0 + (np.arange((i1 - i0) * U) + 0.5) / U - 0.5) * TSAMP
        rel = tt - t_c[c]
        if tau[c] > 0.0:
            rel = np.maximum(rel, -5 * width_s)  # mismo recorte que model.compute_profile
            # normalize=True: gaussiana (pico 1) convolucionada con una
            # exponencial de área 1 -> conserva la energía. Con el default
            # de fitburst (normalize_pbf: False) la energía crece con tau y
            # el espectro se inclina como (f/REF_FREQ)^4.
            prof = fb_profile.compute_profile_pbf(rel, 0.0, width_s, np.asarray(FREQS[c]), REF_FREQ,
                                                  scat_ref_s, sc_index=SCATTERING_INDEX, normalize=True)
        else:
            prof = fb_profile.compute_profile_gaussian(rel, 0.0, width_s)
        prof = _media_movil(prof, int(round(smear[c] / TSAMP * U)))
        vals = prof.reshape(-1, U).mean(axis=1) * espectro[c]
        canales.append((i0, vals))
    return canales, t_c


def serie_dedispersada(canales, t_c, t_top_s, solo_visible=False):
    """Suma de canales alineada al DM real con desplazamientos enteros (como
    las pipelines). solo_visible=True descarta lo que cae fuera de [0, NSAMP)."""
    shift = np.round((t_c - t_top_s) / TSAMP).astype(int)
    ini = min(i0 - s for (i0, _), s in zip(canales, shift))
    fin = max(i0 + len(v) - s for (i0, v), s in zip(canales, shift))
    out = np.zeros(fin - ini + BOXCAR_MAX)
    for (i0, v), s in zip(canales, shift):
        if solo_visible:
            k = i0 + np.arange(len(v))
            v = np.where((k >= 0) & (k < NSAMP), v, 0.0)
        a = i0 - s - ini
        out[a:a + len(v)] += v
    return out


ANCHOS = np.unique(np.concatenate([np.arange(1, 65), np.round(np.geomspace(64, BOXCAR_MAX, 60)).astype(int)]))


def snr_boxcar_optimo(serie):
    """SNR esperado (sin ruido) con el mejor boxcar de ANCHOS; el ruido de la
    suma de los 672 canales es SIGMA_CH·sqrt(672·w). Devuelve (snr, w_muestras)."""
    c = np.concatenate([[0.0], np.cumsum(serie)])
    mejor, w_mejor = 0.0, 1
    for w in ANCHOS:
        v = (c[w:] - c[:-w]).max() / np.sqrt(w)
        if v > mejor:
            mejor, w_mejor = v, int(w)
    return mejor / (SIGMA_CH * np.sqrt(NUM_FREQ)), w_mejor


def energia(canales):
    """(energía total, energía dentro del archivo, fracción por canal dentro)."""
    tot = np.zeros(NUM_FREQ)
    dentro = np.zeros(NUM_FREQ)
    for c, (i0, v) in enumerate(canales):
        tot[c] = v.sum()
        a, b = max(0, -i0), min(len(v), NSAMP - i0)
        dentro[c] = v[a:b].sum() if b > a else 0.0
    return tot, dentro


# ============================================================================
# Escritura
# ============================================================================
def escribir(nombre, canales, escala_jy, out_path, seed_base):
    rng = np.random.default_rng(semilla(seed_base, nombre))
    data = rng.standard_normal((NUM_FREQ, NSAMP), dtype=np.float32)
    data *= SIGMA_CH
    for c, (i0, v) in enumerate(canales):
        a, b = max(0, -i0), min(len(v), NSAMP - i0)
        if b > a:
            data[c, i0 + a:i0 + b] += (escala_jy * v[a:b]).astype(np.float32)
    # cuentas = floor(PISO + 0.5 + NIVELES·x/sigma) recortado a 0-255, es decir
    # redondeo al nivel más cercano. write_fil mapea [lo, hi] -> [0, 255] con
    # floor, así que se le pasa lo = -(PISO+0.5)/NIVELES·sigma y hi - lo = 255/NIVELES·sigma.
    lo_s = -(PISO_NIVELES + 0.5) / NIVELES_POR_SIGMA   # en sigma (-8.125)
    hi_s = lo_s + 255.0 / NIVELES_POR_SIGMA            # en sigma (+55.625)
    n_clip = int((data < lo_s * SIGMA_CH).sum() + (data >= hi_s * SIGMA_CH).sum())
    # Pérdida esperada de señal por el recorte (valor esperado sobre el ruido,
    # no la realización): sum(s - (E[clip(s+n)] - E[clip(n)])) / sum(s)
    s = np.concatenate([v for _, v in canales]) * (escala_jy / SIGMA_CH)
    s = s[s > 0.05]
    e0 = _media_recortada(np.zeros(1), lo_s, hi_s)[0]
    perdida = float((s - (_media_recortada(s, lo_s, hi_s) - e0)).sum() / s.sum()) if s.size else 0.0
    centro = (lo_s + hi_s) / 2 * SIGMA_CH
    write_fil(data, FREQS, np.array([0.0, TSAMP]), out_path, nbits=8, source_name=nombre,
              escala_fija=(centro, (hi_s - lo_s) / 2 * SIGMA_CH))
    return n_clip, 100.0 * perdida


def _media_recortada(s, lo, hi):
    """E[clip(s + n, lo, hi)] con n ~ N(0, 1), todo en unidades de sigma."""
    a, b = lo - s, hi - s
    return (s * (norm.cdf(b) - norm.cdf(a)) + norm.pdf(a) - norm.pdf(b)
            + lo * norm.cdf(a) + hi * norm.sf(b))


def _ctx(p, idx, dm=None):
    return modelo(p["dm"] if dm is None else dm, idx, p["width_s"], p["scat_ref_s"],
                  p["sp_idx"], p["sp_run"], p["t_top_s"])


def tarea_snr_unitario(p):
    canales, t_c = _ctx(p, DM_INDEX)
    snr, w = snr_boxcar_optimo(serie_dedispersada(canales, t_c, p["t_top_s"]))
    return p["id"], snr, w


def tarea_par_AB(args):
    """A_i y B_i comparten modelo: solo cambia la escala."""
    p, escala_a, escala_b, snr_u, w_opt, out, seed_base = args
    canales, _ = _ctx(p, DM_INDEX)
    filas = []
    for set_, escala, carpeta in (("A", escala_a, "A_snr_fijo"), ("B", escala_b, "B_amplitud_fija")):
        nombre = f"{set_}_{p['id']}"
        path = out / carpeta / f"{nombre}.fil"
        n_clip, perdida = escribir(nombre, canales, escala, path, seed_base)
        filas.append((carpeta, {
            "name": nombre, "fil_path": str(path), "set": set_, **p, "dm_index": DM_INDEX,
            "escala_jy": escala, "snr_esperado": escala * snr_u, "boxcar_opt_ms": w_opt * TSAMP * 1e3,
            "barrido_s": barrido_s(p["dm"], DM_INDEX), "n_muestras_clip": n_clip, "perdida_recorte_pct": perdida,
        }))
    return filas


def tarea_negativo(args):
    p, idx, escala, out, seed_base = args
    dm_neg = p["dm"] * barrido_s(1.0, DM_INDEX) / barrido_s(1.0, idx)
    canales, _ = _ctx(p, idx, dm=dm_neg)
    nombre = f"NEG_{p['id']}"
    path = out / "negativos" / f"{nombre}.fil"
    n_clip, perdida = escribir(nombre, canales, escala, path, seed_base)
    return [("negativos", {
        "name": nombre, "fil_path": str(path), "set": "NEG", **p,
        "dm_equivalente_pos": p["dm"], "dm": dm_neg, "dm_index": idx,
        "escala_jy": escala, "par_de": f"A_{p['id']}",
        "barrido_s": barrido_s(dm_neg, idx), "n_muestras_clip": n_clip, "perdida_recorte_pct": perdida,
    })]


def ubicar_corte(p):
    """t_top que deja dentro del archivo ~p['frac_objetivo'] de la energía.

    Con el modelo en t_top=0, cada canal tiene su energía E_c y su centroide
    temporal; al mover t_top el canal queda dentro si su centroide cae en
    [0, NSAMP). "inicio" deja los canales que llegan al final (parte baja de
    la banda), "fin" los que llegan primero (parte alta). La fracción exacta
    queda en frac_energia_visible del manifest."""
    canales, _ = modelo(p["dm"], DM_INDEX, p["width_s"], p["scat_ref_s"], p["sp_idx"], p["sp_run"], 0.0)
    E = np.array([v.sum() for _, v in canales])
    cen = np.array([(i0 + np.dot(np.arange(len(v)), v) / e) * TSAMP if e > 0 else np.nan
                    for (i0, v), e in zip(canales, E)])
    ok = np.isfinite(cen)
    E, cen = E[ok], cen[ok]
    orden = np.argsort(-cen if p["borde"] == "inicio" else cen)
    j = min(int(np.searchsorted(np.cumsum(E[orden]) / E.sum(), p["frac_objetivo"])), orden.size - 1)
    c = cen[orden[j]]
    return float(-c + 0.5 * TSAMP if p["borde"] == "inicio" else NSAMP * TSAMP - c - 0.5 * TSAMP)


def tarea_parcial(args):
    p, snr_full, out, seed_base = args
    p = {**p, "t_top_s": ubicar_corte(p)}
    canales, t_c = _ctx(p, DM_INDEX)
    snr_u, w_u = snr_boxcar_optimo(serie_dedispersada(canales, t_c, p["t_top_s"]))
    snr_vis_u, w_vis = snr_boxcar_optimo(serie_dedispersada(canales, t_c, p["t_top_s"], solo_visible=True))
    tot, dentro = energia(canales)
    frac_canal = np.divide(dentro, tot, out=np.zeros_like(tot), where=tot > 0)
    visibles = FREQS[frac_canal >= 0.5]
    escala = snr_full / snr_u
    nombre = f"P_{p['id']}_{p['borde']}"
    path = out / "parcial" / f"{nombre}.fil"
    n_clip, perdida = escribir(nombre, canales, escala, path, seed_base)
    return [("parcial", {
        "name": nombre, "fil_path": str(path), "set": "P", **p, "dm_index": DM_INDEX,
        "escala_jy": escala, "snr_burst_completo": snr_full,
        "snr_visible_esperado": escala * snr_vis_u, "boxcar_opt_visible_ms": w_vis * TSAMP * 1e3,
        "frac_energia_visible": float(dentro.sum() / tot.sum()) if tot.sum() > 0 else 0.0,
        "f_min_visible_mhz": float(visibles.min()) if visibles.size else np.nan,
        "f_max_visible_mhz": float(visibles.max()) if visibles.size else np.nan,
        "n_canales_visibles": int(visibles.size),
        "barrido_s": barrido_s(p["dm"], DM_INDEX), "n_muestras_clip": n_clip, "perdida_recorte_pct": perdida,
    })]


def guardar_csv(path, filas):
    path.parent.mkdir(parents=True, exist_ok=True)
    campos = list(dict.fromkeys(k for f in filas for k in f))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=campos)
        w.writeheader()
        w.writerows(filas)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--n", type=int, default=1000, help="Sorteos para A, B y negativos (cada uno)")
    ap.add_argument("--n-parcial", type=int, default=1000)
    ap.add_argument("--snr-a", type=float, default=8.0)
    ap.add_argument("--snr-b-mediana", type=float, default=10.0,
                    help="S0 de B se calibra para que la mediana del SNR esperado sea este valor")
    ap.add_argument("--snr-parcial", type=float, default=15.0,
                    help="SNR del burst COMPLETO (antes del corte) en el subset parcial")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=20260923)
    ap.add_argument("--out", type=Path, default=HERE / "outputs_comparacion")
    args = ap.parse_args()

    out = args.out.resolve()
    if out == (HERE / "outputs").resolve():
        sys.exit("✗ --out no puede ser outputs/ (dataset de referencia, intocable)")
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    rng = np.random.default_rng(args.seed)
    pool = cargar_pool()
    principales = sortear(args.n, rng, pool, "")
    idx_neg = [sortear_dm_index(rng) for _ in principales]
    parciales = sortear(args.n_parcial, rng, pool, "p")
    for j, p in enumerate(parciales):
        # inicio: el canal alto llegó antes del archivo; fin: el canal bajo llega después.
        # Se sortea la fracción de energía visible, no el tiempo: con espectros del
        # catálogo concentrados arriba de la banda (38% con pico en 501.6 MHz), un
        # corte uniforme en tiempo deja casi todos los "inicio" en ~0% y los "fin"
        # en ~100%. t_top se fija en ubicar_corte() y queda en el manifest.
        p["borde"] = "inicio" if j % 2 == 0 else "fin"
        p["frac_objetivo"] = float(rng.uniform(*FRAC_VISIBLE_RANGO))
        p["t_top_s"] = np.nan
    guardar_csv(out / "parametros_sorteados.csv",
                [{**p, "dm_index_negativo": i} for p, i in zip(principales, idx_neg)] + parciales)
    print(f"Pool del catálogo: {len(pool)} formas | NSAMP={NSAMP} ({NSAMP*TSAMP:.0f} s) | "
          f"SEFD={SEFD_JY:.2f} Jy | sigma_canal={SIGMA_CH:.3f} Jy", flush=True)

    with Pool(args.workers) as pool_proc:
        print(f"[fase 1] SNR unitario de {len(principales)} sorteos...", flush=True)
        snr_u = {pid: (s, w) for pid, s, w in pool_proc.imap_unordered(tarea_snr_unitario, principales)}
        snr_arr = np.array([snr_u[p["id"]][0] for p in principales])
        s0_b = args.snr_b_mediana / float(np.median(snr_arr))
        print(f"  SNR por Jy de amplitud: mediana {np.median(snr_arr):.4g}, "
              f"rango {snr_arr.min():.3g}-{snr_arr.max():.3g} | S0 (B) = {s0_b:.4g} Jy", flush=True)

        tareas = []
        for p, idx in zip(principales, idx_neg):
            s, w = snr_u[p["id"]]
            escala_a = args.snr_a / s
            tareas.append((tarea_par_AB, (p, escala_a, s0_b, s, w, out, args.seed)))
            tareas.append((tarea_negativo, (p, idx, escala_a, out, args.seed)))
        tareas += [(tarea_parcial, (p, args.snr_parcial, out, args.seed)) for p in parciales]

        print(f"[fase 2] escribiendo {2 * len(principales) + len(principales) + len(parciales)} .fil "
              f"con {args.workers} workers...", flush=True)
        manifests = {}
        hechos = 0
        for filas in pool_proc.imap_unordered(_despachar, tareas):
            for carpeta, fila in filas:
                manifests.setdefault(carpeta, []).append(fila)
                hechos += 1
            if hechos % 50 < len(filas):
                print(f"  {hechos} archivos ({(time.time() - t0) / 60:.1f} min)", flush=True)

    for carpeta, filas in manifests.items():
        filas.sort(key=lambda f: f["name"])
        guardar_csv(out / carpeta / "manifest.csv", filas)

    config = {
        "fecha": time.strftime("%Y-%m-%d %H:%M"), "script": Path(__file__).name, **vars(args),
        "out": str(out), "S0_B_jy": s0_b, "nsamp": NSAMP, "tsamp_s": TSAMP, "nbits": 8,
        "freq_lo_mhz": FREQ_LO, "freq_hi_mhz": FREQ_HI, "num_freq": NUM_FREQ,
        "sefd_jy": SEFD_JY, "n_pol": N_POL, "sigma_canal_jy": float(SIGMA_CH),
        "cuantizacion": f"uint8, ruido del sistema en {PISO_NIVELES} con sigma = {NIVELES_POR_SIGMA} niveles, redondeo",
        "piso_niveles": PISO_NIVELES, "niveles_por_sigma": NIVELES_POR_SIGMA,
        "dm_sorteo": DM_SORTEO, "dm_search": DM_SEARCH, "t_top_rango_s": T_TOP_RANGO,
        "sobremuestreo_t": SOBREMUESTREO_T, "boxcar_max_muestras": BOXCAR_MAX,
        "smearing_intracanal": True, "rfi": False, "pool_catalogo": len(pool), "min_canales_emision": MIN_CANALES_EMISION,
    }
    (out / "config.json").write_text(json.dumps(config, indent=2, default=str))
    print(f"✓ Listo en {(time.time() - t0) / 60:.1f} min -> {out}")


def _despachar(t):
    fn, a = t
    return fn(a)


if __name__ == "__main__":
    main()
