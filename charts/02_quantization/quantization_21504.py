#!/usr/bin/env python3
"""Efecto de la cuantización sobre el S/N, en el set de prueba del grupo 3 (21 504 canales).

Para cada archivo se reconstruye EXACTAMENTE el dato en float (mismo modelo y misma semilla
que 12_generate_comparison_set_21504.py; se verifica que al cuantizarlo da los mismos bytes
del .fil) y se compara la MISMA realización de ruido + señal con distintas cuantizaciones:

  float          sin cuantizar (referencia)
  8 bits         el .fil (piso 32, 1 sigma = 4 niveles)
  4b 2niv fija   desde los 8 bits: q = redondeo((x - 32) / 2), recorte a [-7, 7]  (2 niveles/sigma, ±3,5 sigma)
  4b 1niv fija   desde los 8 bits: 1 nivel/sigma, recorte a ±7 sigma
  4b 3niv fija   desde los 8 bits: 3 niveles/sigma, recorte a ±2,3 sigma
  4b pirate      formato de red de pirate: escala/offset por (canal, 256 muestras), 2 niveles/sigma

S/N: dedispersado al DM real (desplazamientos enteros), mejor boxcar, ruido medido fuera del pulso
en la serie de cada variante.

Uso: .venv/bin/python quantization_21504.py SET_DIR [--sets A,B] [--salida plots_21504]
"""
import argparse
import csv
import importlib.util
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
(HERE / "plots_21504").mkdir(exist_ok=True)
SIM = HERE.parent / "01_simulator"                   # generador y física del simulador


def cargar(nombre_mod, archivo):
    spec = importlib.util.spec_from_file_location(nombre_mod, (SIM if archivo[:2].isdigit() else HERE) / archivo)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


GEN = cargar("gen21504", "12_generate_comparison_set_21504.py")
G = GEN.G
VAL = cargar("val21504", "validate_21504.py")

ANCHOS = G.ANCHOS[G.ANCHOS <= 1024]


def reconstruir_float(r):
    """Dato en sigma, (tiempo, canal ASCENDENTE), igual al de escribir() antes de cuantizar."""
    p = {k: float(r[k]) for k in ("dm", "width_s", "scat_ref_s", "sp_idx", "sp_run")}
    idx, t_top, nsamp, escala = float(r["dm_index"]), float(r["t_top_s"]), int(r["nsamp"]), float(r["escala_jy"])
    canales, _ = GEN.base(p, idx, t_top, nsamp, dm=p["dm"])
    rng = np.random.default_rng(G.semilla(GEN.SEED, r["name"]))
    x = np.empty((nsamp, GEN.FREQS.size), np.float32)
    i0s = np.array([i0 for i0, _ in canales])
    i1s = np.array([i0 + len(v) for i0, v in canales])
    for a in range(0, nsamp, GEN.BLOQUE):
        b = min(a + GEN.BLOQUE, nsamp)
        blk = rng.standard_normal((b - a, GEN.FREQS.size), dtype=np.float32)
        for c in np.flatnonzero((i1s > a) & (i0s < b)):
            i0, v = canales[c]
            lo, hi = max(a, i0), min(b, i0 + len(v))
            blk[lo - a:hi - a, c] += (escala / GEN.SIGMA) * v[lo - i0:hi - i0]
        x[a:b] = blk
    return x


def variantes(xf, u8):
    """Todas en sigma, (tiempo, canal descendente = orden del .fil)."""
    xf = xf[:, ::-1]
    d8 = (u8.astype(np.float32) - 32.0)                    # en niveles (1 sigma = 4)
    out = {"float": xf, "8 bits": d8 / 4.0}
    for niv, nom in ((2, "4b 2niv fija"), (1, "4b 1niv fija"), (3, "4b 3niv fija")):
        q = np.clip(np.rint(d8 / 4.0 * niv), -7, 7)
        out[nom] = q / niv
    # pirate: escala/offset por (canal, 256 muestras), desde los 8 bits
    n = (d8.shape[0] // 256) * 256
    blk = (d8[:n] / 4.0).reshape(-1, 256, d8.shape[1])
    mu = blk.mean(axis=1, keepdims=True)
    sd = blk.std(axis=1, keepdims=True)
    esc = np.where(sd > 0, sd / 2.0, 1.0)
    q = np.clip(np.rint((blk - mu) / esc), -7, 7)
    out["4b pirate"] = (q * esc + mu).reshape(n, -1)
    return out


def snr(serie, i_esp, nchan):
    fuera = np.r_[serie[:max(1, i_esp - 2000)], serie[i_esp + 3000:]]
    s0 = fuera.std()
    serie = serie - np.median(fuera)
    c = np.concatenate([[0.0], np.cumsum(serie)])
    mejor = -np.inf
    for w in ANCHOS:
        v = (c[w:] - c[:-w]) / (s0 * np.sqrt(w))
        lo, hi = max(0, i_esp - 300 - w), min(len(v), i_esp + 300)
        mejor = max(mejor, float(v[lo:hi].max()))
    return mejor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("set_dir", type=Path)
    ap.add_argument("--sets", default="A,B")
    ap.add_argument("--salida", type=Path, default=HERE / "plots_21504")
    a = ap.parse_args()
    cfg = json.loads((a.set_dir / "config.json").read_text())
    f = VAL.freqs_fil(cfg)
    carpetas = {"A": "A_snr_fijo", "B": "B_amplitud_fija"}
    filas, perfiles = [], {}
    for s in a.sets.split(","):
        for r in csv.DictReader(open(a.set_dir / carpetas[s] / "manifest.csv")):
            u8 = np.asarray(VAL.leer(r["fil_path"]))
            xf = reconstruir_float(r)
            q8 = np.clip(np.floor(xf * 4 + 32.5), 0, 255).astype(np.uint8)[:, ::-1]
            iguales = float((q8 == u8).mean())
            vs = variantes(xf, u8)
            del xf
            dm, idx = float(r["dm"]), float(r["dm_index"])
            fila = {"name": r["name"], "set": s, "dm": dm, "snr_esperado": float(r["snr_esperado"]),
                    "bytes_iguales": iguales}
            for nom, x in vs.items():
                serie, off = VAL.dedispersar(x, f, dm, idx)
                i_esp = int(round(float(r["t_top_s"]) / 1e-3)) + off
                fila[nom] = snr(serie, i_esp, x.shape[1])
                if r["name"] in ("A_001", "B_001", "B_005"):
                    perfiles.setdefault(r["name"], {})[nom] = serie[i_esp - 150:i_esp + 150] / serie[:i_esp - 2000].std()
            filas.append(fila)
            print(f"{r['name']:>6} DM {dm:6.0f}  esperado {fila['snr_esperado']:6.1f}  bytes iguales {iguales:.6f}  " +
                  "  ".join(f"{k}: {fila[k]:6.2f}" for k in vs), flush=True)

    noms = list(vs)
    with open(a.salida / "quantization_21504.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(filas[0]))
        w.writeheader()
        w.writerows(filas)

    print("\nS/N relativo al float (mediana [min, max]):")
    tabla = {}
    for s in a.sets.split(","):
        sub = [r for r in filas if r["set"] == s and r["float"] > 4]
        for nom in noms[1:]:
            rel = np.array([r[nom] / r["float"] for r in sub])
            tabla[(s, nom)] = rel
            print(f"  {s} {nom:>13}: {np.median(rel):.4f}  [{rel.min():.4f}, {rel.max():.4f}]  (n={len(rel)})")

    fig, axs = plt.subplots(1, 2, figsize=(14, 5.2), gridspec_kw={"width_ratios": [1.2, 1]})
    ax = axs[0]
    for k, nom in enumerate(noms[1:]):
        for j, s in enumerate(a.sets.split(",")):
            rel = tabla[(s, nom)]
            xs = k + (j - 0.5) * 0.3 + np.random.default_rng(k).uniform(-0.06, 0.06, rel.size)
            ax.plot(xs, 100 * (rel - 1), "o", ms=5, color=f"C{j}", label=f"set {s}" if k == 0 else None, alpha=0.8)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(range(len(noms) - 1), noms[1:])
    ax.set_ylabel("S/N respecto al dato sin cuantizar (%)")
    ax.set_title("misma realización de ruido y señal; solo cambia la cuantización\n(cada punto = un archivo con S/N > 4)")
    ax.legend()
    ax = axs[1]
    nombre = "B_005" if "B_005" in perfiles else next(iter(perfiles))
    t = np.arange(-150, 150)
    for k, nom in enumerate(noms):
        ax.plot(t, perfiles[nombre][nom], lw=1.0 if nom != "float" else 2.0, label=nom, color=f"C{k}",
                alpha=0.9 if nom != "float" else 0.5)
    ax.set_xlabel("ms respecto a la llegada esperada")
    ax.set_ylabel("S/N por muestra de 1 ms")
    ax.set_title(f"{nombre}: serie dedispersada con cada cuantización")
    ax.legend(fontsize=8)
    out = a.salida / "quantization_21504.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    print(f"figura: {out}")


if __name__ == "__main__":
    main()
