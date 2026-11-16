#!/usr/bin/env python3
"""
Grupo 3: set de comparación a 21 504 canales (plan en INVENTARIO_DATASETS.md, acordado
con el equipo CHARTS 2026-10-05/08).

Los MISMOS sorteos que outputs_comparacion/ (grupo 2, 07_generar_set_comparacion.py,
leídos de su parametros_sorteados.csv), con la misma física (modelo() de 07_: smearing
dentro del canal, espectro normalizado al canal más brillante, scattering que conserva
la energía, sobremuestreo x4), sobre una grilla fina:

  - 21 504 canales = cada canal de 672 (300,45 kHz) partido en 32 (9,389 kHz).
    Centros de 299,855 a 501,745 MHz, sigma = SEFD / sqrt(9,389 kHz · 1 ms) = 768,5 Jy.
  - A: SNR 8 recalculado con el modelo de 21 504 canales (boxcar óptimo, dedispersado al DM real).
  - B: S0 = 616,87 Jy, el MISMO que en el grupo 2 (la diferencia de SNR viene solo de la grilla).
  - negativos: par de A_i con su dm_index_negativo y la escala de A_i.
  - parcial: burst completo a SNR 15 (recalculado), mismo borde y fracción de energía objetivo.
  - Largo VARIABLE, sin padding: llegada al canal de 501,6 MHz sorteada en 3-5 s
    (semilla por id) + barrido de su propio DM + cola del pulso en el canal más bajo + 1 s,
    redondeado hacia arriba al segundo. En parcial, el mismo largo y el corte de 07_.
  - t_top_s del manifest = llegada a 501,6 MHz (misma convención que el grupo 2 y el bench).
  - 8 bits, piso 32, 1 sigma = 4 niveles, redondeo (misma cuantización que write_fil de 03_).

Se escribe por bloques de tiempo (un archivo de 98 s pesa ~2 GB en uint8 y 8,4 GB en float32).
Reanudable: los archivos terminados quedan en _hechos.jsonl y se saltan.

Uso (venv del simulador):
  .venv/bin/python 12_generar_set_comparacion_21504.py [--limite 10] [--workers 24]
      [--out outputs_comparacion_21504] [--sets A,B,NEG,P]
"""
import argparse
import csv
import importlib.util
import json
import math
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("gen07", HERE / "07_generar_set_comparacion.py")
G = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(G)

BASE_G2 = HERE / "outputs_comparacion"          # grupo 2 (enlace al HDD)
# sorteos del grupo 2: los del grupo 2 si están; si no, la copia reconstruida desde los manifiestos del grupo 3
SORTEOS = BASE_G2 / "parametros_sorteados.csv"
if not SORTEOS.exists():
    SORTEOS = HERE / "parametros_sorteados_grupo2.csv"
SEED = 20260923                                 # semilla del grupo 2
S0_B_JY = 616.8725247769761                     # config.json del grupo 2
SNR_A, SNR_PARCIAL = 8.0, 15.0
T_TOP_LLEGADA = (3.0, 5.0)
MARGEN_S = 1.0
F_REF_TOP = 501.6                               # t_top = llegada a 501,6 MHz
BLOQUE = 4096                                   # muestras por bloque de escritura

# ---- Grilla fina: 672 canales x 32 ----
N_SUB = 32
COARSE = np.linspace(300.0, 501.6, 672)
DNU_COARSE = (501.6 - 300.0) / 671
DNU = DNU_COARSE / N_SUB
FREQS = (COARSE[:, None] + (np.arange(N_SUB) - (N_SUB - 1) / 2) * DNU).ravel()   # ascendente
G.FREQS = FREQS
G.NUM_FREQ = FREQS.size
G.RES_FREQ_MHZ = DNU
G.SIGMA_CH = G.SEFD_JY / np.sqrt(G.N_POL * DNU * 1e6 * G.TSAMP)
SIGMA = G.SIGMA_CH
TSAMP = G.TSAMP


def t_top_modelo(dm, idx, t_top_501_6):
    """modelo() de 07_ usa t_top = llegada al canal más alto (FREQS[-1] = 501,745 MHz)."""
    return t_top_501_6 + float(G.delay_s(dm, idx, FREQS[-1]) - G.delay_s(dm, idx, F_REF_TOP))


def largo_muestras(dm, idx, t_top, p):
    tau_lo = p["scat_ref_s"] * (FREQS[0] / G.REF_FREQ) ** G.SCATTERING_INDEX
    cola = 6 * p["width_s"] + min(10 * tau_lo, 200.0) + float(G.smear_s(dm, idx, FREQS[0])) / 2
    fin = t_top + G.barrido_s(dm, idx) + cola + MARGEN_S
    return int(math.ceil(fin)) * 1000


def t_llegada(pid):
    return float(np.random.default_rng(G.semilla(SEED, f"t_top_21504:{pid}")).uniform(*T_TOP_LLEGADA))


# ---------------------------------------------------------------------------
def escribir(nombre, canales, escala_jy, path, nsamp):
    """Ruido + señal por bloques, cuantizado como write_fil(escala_fija) de 03_:
    nivel = floor(32,5 + 4·x/sigma) recortado a 0-255. Orden SIGPROC: canales descendentes."""
    tmp = path.with_suffix(".fil.tmp")
    G.write_fil(np.zeros((FREQS.size, 0)), FREQS, np.array([0.0, TSAMP]), tmp, nbits=8, source_name=nombre,
                escala_fija=(0.0, 1.0))                      # solo el header (no depende de la escala)
    rng = np.random.default_rng(G.semilla(SEED, nombre))
    i0s = np.array([i0 for i0, _ in canales])
    i1s = np.array([i0 + len(v) for i0, v in canales])
    n_clip = 0
    with open(tmp, "ab") as f:
        for a in range(0, nsamp, BLOQUE):
            b = min(a + BLOQUE, nsamp)
            x = rng.standard_normal((b - a, FREQS.size), dtype=np.float32)    # (tiempo, canal asc.), en sigma
            for c in np.flatnonzero((i1s > a) & (i0s < b)):
                i0, v = canales[c]
                lo, hi = max(a, i0), min(b, i0 + len(v))
                x[lo - a:hi - a, c] += (escala_jy / SIGMA) * v[lo - i0:hi - i0]
            q = np.floor(x * G.NIVELES_POR_SIGMA + (G.PISO_NIVELES + 0.5))
            n_clip += int((q < 0).sum() + (q > 255).sum())
            f.write(np.clip(q, 0, 255).astype(np.uint8)[:, ::-1].tobytes())
    tmp.rename(path)
    return n_clip


def base(p, idx, t_top, nsamp, dm=None):
    G.NSAMP = nsamp                                       # serie_dedispersada / energia / ubicar_corte
    dm = p["dm"] if dm is None else dm
    canales, t_c = G.modelo(dm, idx, p["width_s"], p["scat_ref_s"], p["sp_idx"], p["sp_run"],
                            t_top_modelo(dm, idx, t_top))
    return canales, t_c


def tarea_principal(args):
    """A_i, B_i y NEG_i (comparten sorteo, llegada y largo)."""
    p, out, sets = args
    t_top = t_llegada(p["id"])
    nsamp = largo_muestras(p["dm"], G.DM_INDEX, t_top, p)
    canales, t_c = base(p, G.DM_INDEX, t_top, nsamp)
    snr_u, w = G.snr_boxcar_optimo(G.serie_dedispersada(canales, t_c, t_c[-1]))
    escala_a = SNR_A / snr_u
    comun = {**p, "t_top_s": t_top, "nsamp": nsamp, "largo_s": nsamp * TSAMP, "snr_por_jy": snr_u}
    filas = []
    for set_, escala, carpeta in (("A", escala_a, "A_snr_fijo"), ("B", S0_B_JY, "B_amplitud_fija")):
        if set_ not in sets:
            continue
        nombre = f"{set_}_{p['id']}"
        path = out / carpeta / f"{nombre}.fil"
        n_clip = escribir(nombre, canales, escala, path, nsamp)
        filas.append((carpeta, {"name": nombre, "fil_path": str(path), "set": set_, **comun,
                                "dm_index": G.DM_INDEX, "escala_jy": escala, "snr_esperado": escala * snr_u,
                                "boxcar_opt_ms": w * TSAMP * 1e3, "barrido_s": G.barrido_s(p["dm"], G.DM_INDEX),
                                "n_muestras_clip": n_clip}))
    if "NEG" in sets:
        idx = p["dm_index_negativo"]
        dm_neg = p["dm"] * G.barrido_s(1.0, G.DM_INDEX) / G.barrido_s(1.0, idx)
        canales_n, _ = base(p, idx, t_top, nsamp, dm=dm_neg)
        nombre = f"NEG_{p['id']}"
        path = out / "negativos" / f"{nombre}.fil"
        n_clip = escribir(nombre, canales_n, escala_a, path, nsamp)
        filas.append(("negativos", {"name": nombre, "fil_path": str(path), "set": "NEG", **comun,
                                    "dm_equivalente_pos": p["dm"], "dm": dm_neg, "dm_index": idx,
                                    "escala_jy": escala_a, "par_de": f"A_{p['id']}",
                                    "barrido_s": G.barrido_s(dm_neg, idx), "n_muestras_clip": n_clip}))
    return filas


def tarea_parcial(args):
    p, out, _ = args
    nsamp = largo_muestras(p["dm"], G.DM_INDEX, T_TOP_LLEGADA[1], p)
    G.NSAMP = nsamp
    t_top_mod = G.ubicar_corte(p)                         # llegada al canal más alto de la grilla fina
    t_top = t_top_mod - float(G.delay_s(p["dm"], G.DM_INDEX, FREQS[-1]) - G.delay_s(p["dm"], G.DM_INDEX, F_REF_TOP))
    canales, t_c = base(p, G.DM_INDEX, t_top, nsamp)
    snr_u, _ = G.snr_boxcar_optimo(G.serie_dedispersada(canales, t_c, t_c[-1]))
    snr_vis_u, w_vis = G.snr_boxcar_optimo(G.serie_dedispersada(canales, t_c, t_c[-1], solo_visible=True))
    tot, dentro = G.energia(canales)
    frac_canal = np.divide(dentro, tot, out=np.zeros_like(tot), where=tot > 0)
    visibles = FREQS[frac_canal >= 0.5]
    escala = SNR_PARCIAL / snr_u
    nombre = f"P_{p['id']}_{p['borde']}"
    path = out / "parcial" / f"{nombre}.fil"
    n_clip = escribir(nombre, canales, escala, path, nsamp)
    return [("parcial", {"name": nombre, "fil_path": str(path), "set": "P", **p, "t_top_s": t_top,
                         "nsamp": nsamp, "largo_s": nsamp * TSAMP, "dm_index": G.DM_INDEX, "escala_jy": escala,
                         "snr_burst_completo": SNR_PARCIAL, "snr_visible_esperado": escala * snr_vis_u,
                         "boxcar_opt_visible_ms": w_vis * TSAMP * 1e3,
                         "frac_energia_visible": float(dentro.sum() / tot.sum()) if tot.sum() > 0 else 0.0,
                         "f_min_visible_mhz": float(visibles.min()) if visibles.size else np.nan,
                         "f_max_visible_mhz": float(visibles.max()) if visibles.size else np.nan,
                         "n_canales_visibles": int(visibles.size),
                         "barrido_s": G.barrido_s(p["dm"], G.DM_INDEX), "n_muestras_clip": n_clip})]


def _despachar(t):
    fn, a = t
    t0 = time.time()
    filas = fn(a)
    return filas, time.time() - t0


def leer_sorteos():
    num = {"dm", "width_s", "scat_600_s", "scat_ref_s", "sp_idx", "sp_run", "t_top_s",
           "dm_index_negativo", "frac_objetivo"}
    principales, parciales = [], []
    with open(SORTEOS, newline="") as f:
        for r in csv.DictReader(f):
            p = {k: (float(v) if k in num and v not in ("", None) else v) for k, v in r.items()}
            p["n_canales_emision"] = G.n_canales_emision(p["sp_idx"], p["sp_run"])   # en la grilla fina
            (parciales if p["id"].startswith("p") else principales).append(p)
    for p in principales:
        p.pop("borde", None), p.pop("frac_objetivo", None)
    for p in parciales:
        p.pop("dm_index_negativo", None)
    return principales, parciales


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--out", type=Path, default=HERE / "outputs_comparacion_21504")
    ap.add_argument("--limite", type=int, default=None, help="primeros N sorteos de cada tipo (prueba)")
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--sets", default="A,B,NEG,P")
    a = ap.parse_args()
    sets = set(a.sets.split(","))
    out = a.out.resolve()
    for d in ("A_snr_fijo", "B_amplitud_fija", "negativos", "parcial"):
        (out / d).mkdir(parents=True, exist_ok=True)
    hechos_path = out / "_hechos.jsonl"
    hechos = {}
    if hechos_path.exists():
        for line in hechos_path.read_text().splitlines():
            r = json.loads(line)
            hechos[r["clave"]] = r["filas"]

    principales, parciales = leer_sorteos()
    if a.limite:
        principales, parciales = principales[:a.limite], parciales[:a.limite]
    tareas = []
    if sets & {"A", "B", "NEG"}:
        tareas += [(tarea_principal, (p, out, sets)) for p in principales if f"main:{p['id']}" not in hechos]
    if "P" in sets:
        tareas += [(tarea_parcial, (p, out, sets)) for p in parciales if f"p:{p['id']}" not in hechos]
    print(f"Grilla: {FREQS.size} canales, {FREQS[0]:.4f}-{FREQS[-1]:.4f} MHz, dnu={DNU * 1e3:.4f} kHz, "
          f"sigma={SIGMA:.2f} Jy | tareas pendientes {len(tareas)} ({len(hechos)} hechas) | {a.workers} workers",
          flush=True)

    t0 = time.time()
    with Pool(a.workers, maxtasksperchild=20) as pool, open(hechos_path, "a") as fh:
        for n, (filas, seg) in enumerate(pool.imap_unordered(_despachar, tareas), 1):
            clave = ("main:" + filas[0][1]["name"].split("_", 1)[1]) if filas[0][0] != "parcial" \
                else "p:" + filas[0][1]["id"]
            fh.write(json.dumps({"clave": clave, "seg": seg, "filas": filas}, default=float) + "\n")
            fh.flush()
            hechos[clave] = filas
            if n % 10 == 0 or n == len(tareas):
                el = time.time() - t0
                print(f"  {n}/{len(tareas)} tareas, {el / 60:.1f} min, ETA {(len(tareas) - n) * el / n / 3600:.2f} h",
                      flush=True)

    manifests = {}
    for filas in hechos.values():
        for carpeta, fila in filas:
            manifests.setdefault(carpeta, []).append(fila)
    for carpeta, filas in manifests.items():
        filas.sort(key=lambda f: f["name"])
        G.guardar_csv(out / carpeta / "manifest.csv", filas)
    (out / "config.json").write_text(json.dumps({
        "fecha": time.strftime("%Y-%m-%d %H:%M"), "script": Path(__file__).name, "base_sorteos": str(BASE_G2),
        "seed": SEED, "num_freq": int(FREQS.size), "freq_min_mhz": float(FREQS[0]), "freq_max_mhz": float(FREQS[-1]),
        "dnu_khz": DNU * 1e3, "sigma_canal_jy": float(SIGMA), "sefd_jy": G.SEFD_JY, "tsamp_s": TSAMP, "nbits": 8,
        "piso_niveles": G.PISO_NIVELES, "niveles_por_sigma": G.NIVELES_POR_SIGMA, "snr_a": SNR_A,
        "S0_B_jy": S0_B_JY, "snr_parcial": SNR_PARCIAL, "t_top_llegada_s": T_TOP_LLEGADA, "margen_s": MARGEN_S,
        "t_top_referencia_mhz": F_REF_TOP, "largo": "variable: ceil(t_top + barrido + cola del pulso + 1 s)",
        "smearing_intracanal": True, "rfi": False, "limite": a.limite,
    }, indent=2))
    print(f"✓ {sum(len(v) for v in manifests.values())} archivos en {out} ({(time.time() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    main()
