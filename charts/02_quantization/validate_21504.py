#!/usr/bin/env python3
"""Validación visual de un archivo del grupo 3 (21 504 canales) y del S/N de A.

Figura por archivo: waterfall de toda la banda (promediado) con la curva de dispersión
teórica, zoom a resolución completa en el fondo de la banda, serie dedispersada al DM real
con el S/N medido (boxcar óptimo) frente al esperado del manifest, e histograma del ruido.
Además mide el S/N de todos los A del set (debe dar ~8).

Uso: .venv/bin/python validate_21504.py SET_DIR [NOMBRE] [--salida plots_21504]
"""
import argparse
import csv
import importlib.util
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
(HERE / "plots_21504").mkdir(exist_ok=True)
SIM = HERE.parent / "01_simulator"                   # generador y física del simulador
_spec = importlib.util.spec_from_file_location("gen07", SIM / "07_generate_comparison_set.py")
G = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(G)

ANCHOS = G.ANCHOS[G.ANCHOS <= 1024]


def leer(path):
    with open(path, "rb") as f:
        raw = f.read(4096)
    hlen = raw.index(b"HEADER_END") + len(b"HEADER_END")
    nchan = 21504
    d = np.memmap(path, dtype=np.uint8, mode="r", offset=hlen)
    return d.reshape(-1, nchan)          # (tiempo, canal descendente)


def freqs_fil(cfg):
    f = np.linspace(cfg["freq_min_mhz"], cfg["freq_max_mhz"], 21504)
    return f[::-1]                        # orden del .fil


def dedispersar(x, f, dm, idx, f_ref=501.6):
    """x: (tiempo, canal) en sigma. Suma con desplazamientos enteros al DM real."""
    """Devuelve la serie y el corrimiento: la llegada a f_ref (muestra i) queda en i + off."""
    sh = np.round((G.delay_s(dm, idx, f) - G.delay_s(dm, idx, f_ref)) / 1e-3).astype(int)
    off = int(sh.min())
    sh = sh - off
    n = x.shape[0] - sh.max()
    out = np.zeros(n, np.float64)
    for c in range(x.shape[1]):
        out += x[sh[c]:sh[c] + n, c]
    return out, off


def snr_boxcar(serie, nchan, i_esp, ventana=300):
    c = np.concatenate([[0.0], np.cumsum(serie)])
    mejor = (-np.inf, 1, 0)
    for w in ANCHOS:
        v = (c[w:] - c[:-w]) / np.sqrt(w * nchan)
        lo, hi = max(0, i_esp - ventana - w), min(len(v), i_esp + ventana)
        j = lo + int(np.argmax(v[lo:hi]))
        if v[j] > mejor[0]:
            mejor = (float(v[j]), int(w), j)
    return mejor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("set_dir", type=Path)
    ap.add_argument("nombre", nargs="?", default="A_001")
    ap.add_argument("--salida", type=Path, default=HERE / "plots_21504")
    a = ap.parse_args()
    import json
    cfg = json.loads((a.set_dir / "config.json").read_text())
    f = freqs_fil(cfg)
    a.salida.mkdir(exist_ok=True)

    # --- S/N de todos los A ---
    man = {r["name"]: r for s in ("A_snr_fijo", "B_amplitud_fija", "negativos", "parcial")
           if (a.set_dir / s / "manifest.csv").exists()
           for r in csv.DictReader(open(a.set_dir / s / "manifest.csv"))}
    print(f"{'archivo':>8} {'DM':>7} {'S/N esperado':>12} {'S/N medido':>10} {'boxcar ms':>9} {'ruido serie':>11}")
    medidos = []
    for nombre, r in sorted(man.items()):
        if r["set"] != "A":
            continue
        x = (leer(r["fil_path"]).astype(np.float32) - 32.0) / 4.0
        dm, t_top = float(r["dm"]), float(r["t_top_s"])
        s, off = dedispersar(x, f, dm, G.DM_INDEX)
        i_esp = int(round(t_top / 1e-3)) + off
        fuera = np.r_[s[:max(1, i_esp - 2000)], s[i_esp + 3000:]]
        snr, w, _ = snr_boxcar(s, x.shape[1], i_esp)
        medidos.append((snr, float(r["snr_esperado"])))
        print(f"{nombre:>8} {dm:7.0f} {float(r['snr_esperado']):12.2f} {snr:10.2f} {w:9d} "
              f"{fuera.std() / np.sqrt(x.shape[1]):11.3f}")
    d = np.array([m - e for m, e in medidos])
    print(f"S/N medido - esperado (A): {d.mean():+.2f} ± {d.std():.2f}  (n={len(d)}; el máximo sobre ruido sesga ~+0,5)")

    # --- figura de un archivo ---
    r = man[a.nombre]
    dm, t_top, idx = float(r["dm"]), float(r["t_top_s"]), float(r["dm_index"])
    raw = leer(r["fil_path"])
    x = (raw.astype(np.float32) - 32.0) / 4.0
    nt = x.shape[0]
    fig = plt.figure(figsize=(14, 11))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.4, 1, 0.8], hspace=0.35, wspace=0.22)

    ax = fig.add_subplot(gs[0, :])
    fb, tb = 128, 8                                          # 168 sub-bandas de 1,2 MHz, 8 ms
    nt8 = nt // tb * tb
    wf = x[:nt8].reshape(nt8 // tb, tb, -1, fb).mean(axis=(1, 3))     # (t, banda desc.)
    fc = f.reshape(-1, fb).mean(axis=1)
    sig = 1 / np.sqrt(tb * fb)
    ax.imshow(wf.T / sig, aspect="auto", origin="upper", cmap="viridis", vmin=-2, vmax=4,
              extent=[0, nt8 * 1e-3, fc[-1], fc[0]])
    tt = t_top + G.delay_s(dm, idx, f) - G.delay_s(dm, idx, 501.6)
    ax.plot(tt, f, "w--", lw=0.8, alpha=0.7, label="retardo teórico (DM del manifest)")
    ax.set_xlabel("tiempo desde el inicio del archivo (s)")
    ax.set_ylabel("frecuencia (MHz)")
    ax.set_title(f"{a.nombre}: DM {dm:.1f}, ancho {float(r['width_s']) * 1e3:.2f} ms, "
                 f"scattering(600 MHz) {float(r['scat_600_s']) * 1e3:.2f} ms, S/N esperado {float(r.get('snr_esperado') or 0):.1f}, "
                 f"largo {nt / 1000:.0f} s\nwaterfall promediado a 168 sub-bandas × 8 ms (en σ del promedio)")
    ax.legend(loc="lower left", fontsize=8)

    ax = fig.add_subplot(gs[1, 0])
    sel = slice(len(f) - 256, len(f))                        # 256 canales finos del fondo (~300 MHz)
    t_lo = G.delay_s(dm, idx, f[sel][-1]) - G.delay_s(dm, idx, 501.6) + t_top
    t_hi = G.delay_s(dm, idx, f[sel][0]) - G.delay_s(dm, idx, 501.6) + t_top
    i0, i1 = int((t_hi - 0.15) / 1e-3), int((t_lo + 0.15) / 1e-3)
    z = x[i0:i1, sel]
    z4 = z[: (z.shape[0] // 4) * 4].reshape(-1, 4, z.shape[1]).mean(1)
    ax.imshow(z4.T / 0.5, aspect="auto", origin="upper", cmap="viridis", vmin=-2, vmax=4,
              extent=[i0 * 1e-3, (i0 + z4.shape[0] * 4) * 1e-3, f[sel][-1], f[sel][0]])
    ax.plot(tt[sel], f[sel], "w--", lw=0.8, alpha=0.7)
    ax.set_xlabel("tiempo (s)")
    ax.set_ylabel("frecuencia (MHz)")
    ax.set_title("zoom: 256 canales finos (9,39 kHz) en el fondo de la banda, 4 ms")

    ax = fig.add_subplot(gs[1, 1])
    s, off = dedispersar(x, f, dm, idx)
    i_esp = int(round(t_top / 1e-3)) + off
    snr, w, j = snr_boxcar(s, x.shape[1], i_esp)
    k = np.arange(max(0, i_esp - 400), min(len(s), i_esp + 400))
    ax.plot((k - i_esp) * 1.0, s[k] / np.sqrt(x.shape[1]), lw=0.6, color="0.4", label="1 ms")
    c = np.concatenate([[0.0], np.cumsum(s)])
    bx = (c[w:] - c[:-w]) / np.sqrt(w * x.shape[1])
    kb = k[k < len(bx)]
    ax.plot((kb - i_esp) + w / 2, bx[kb], color="C1", label=f"boxcar {w} ms")
    ax.axvline(0, color="C3", ls=":", label="t_top del manifest")
    ax.set_xlabel("ms respecto a la llegada esperada a 501,6 MHz")
    ax.set_ylabel("S/N")
    ax.set_title(f"serie dedispersada al DM real: S/N medido {snr:.2f} (esperado {float(r.get('snr_esperado') or 0):.2f})")
    ax.legend(fontsize=8)

    ax = fig.add_subplot(gs[2, 0])
    ruido = raw[:2000].ravel()
    h = np.bincount(ruido, minlength=256)[:80] / ruido.size
    ax.bar(np.arange(80), h, width=1.0, color="C0", label="primeros 2 s (sin burst)")
    lv = np.arange(80)
    from scipy.stats import norm
    ax.plot(lv, norm.cdf(lv + 1 - 32.5, scale=4) - norm.cdf(lv - 32.5, scale=4), "k.", ms=4,
            label="esperado: piso 32, σ = 4 niveles")
    ax.set_xlabel("nivel (8 bits)")
    ax.set_title(f"ruido: media {ruido.mean():.3f}, σ {ruido.std():.3f} niveles")
    ax.legend(fontsize=8)

    ax = fig.add_subplot(gs[2, 1])
    if medidos:
        m = np.array(medidos)
        ax.plot(m[:, 1], m[:, 0], "o")
        ax.axhline(8, color="k", ls=":")
        ax.set_xlim(6, 10)
        ax.set_ylim(5, 11)
        ax.set_xlabel("S/N esperado (manifest)")
        ax.set_ylabel("S/N medido en el .fil")
        ax.set_title(f"set A ({len(m)} archivos): medido − esperado = {d.mean():+.2f} ± {d.std():.2f}")
    out = a.salida / f"validacion_{a.nombre}.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    print(f"figura: {out}")


if __name__ == "__main__":
    main()
