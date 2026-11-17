#!/usr/bin/env python3
"""
Convierte los .npz (positivos y negativos) a .fil (filterbank SIGPROC),
listos para alimentar Heimdall/PRESTO/AstroAccelerate.

Reescribe a mano el header SIGPROC (formato con prefijo de longitud por
etiqueta) -- no depende de sigpyproc. Idéntico en lógica a
2 npz_to_fil/batch_npz_to_fil.py + comparacion_deteccion_chime/
convertir_negativos_a_fil.py del proyecto original: rellena con ruido
gaussiano (estadística robusta por canal, mediana/MAD) a ambos lados del
burst para que una búsqueda ciega de DM hasta DM_SEARCH_MAX no lo pierda
en los bordes.

Requiere haber corrido antes 01_generar_positivos.py y (opcional)
02_generar_negativos.py -- o, si se quiere convertir la versión con RFI
real, 04_agregar_rfi_real.py (usar --positivos-dir/--negativos-dir para
apuntar a outputs/positivos_con_rfi, outputs/negativos_con_rfi).

Salida (nombre de carpeta de salida = nombre de la carpeta de entrada +
"_fil", ej. positivos -> positivos_fil, positivos_con_rfi -> positivos_con_rfi_fil):
  outputs/<positivos-dir>_fil/<catalogo>/*.fil, .../manifest.csv
  outputs/<negativos-dir>_fil/*.fil,            .../manifest.csv

Uso:
  python 03_convert_to_fil.py [--limit N] [--nbits 8] [--skip-negativos]
                                [--positivos-dir outputs/positivos]
                                [--negativos-dir outputs/negativos]
"""
import argparse
import csv
import struct
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "engine"))

from charts_sefd import noise_std_teorico  # noqa: E402

N_SIGMA_CLIP = 5.0  # ver 06_fil_con_rfi.py -- misma referencia (sigma teorico de CHARTS)

# Ventana de búsqueda de DM fija para las herramientas de detección (mismo
# criterio que el proyecto original: "DM desde 20 a 1000, eso para
# todos"). PAD_GUARD_SAMPLES=10000 porque prepsubband (PRESTO) redondea su
# longitud de salida hacia arriba (FFT-friendly) antes de calcular cuántas
# muestras son válidas -- con guard chico eso puede vaciar el archivo
# entero para bursts de ventana corta (ver REPORTE_CORRIDA.md del proyecto
# original, confirmado en 27/2839 archivos).
DM_SEARCH_MIN = 20.0
DM_SEARCH_MAX = 1000.0
PAD_GUARD_SAMPLES = 10000


def write_fil(data_full, freqs, times, out_path, nbits=8, source_name="FRB", escala_fija=None):
    data = np.asarray(data_full, dtype=np.float64)

    if nbits in (8, 16) and escala_fija is not None:
        # Escala FIJA, anclada al sigma TEORICO del ruido de sistema
        # (ecuacion del radiometro, ver engine/charts_sefd.py::noise_std_teorico)
        # -- no minimo/maximo GLOBAL de este archivo en particular. Con
        # minimo/maximo por archivo, el pico del burst domina el rango y
        # aplasta el piso de ruido a una fraccion distinta de 0-255 en cada
        # archivo segun que tan brillante sea su burst (confirmado
        # 2026-08-30: piso en 130/255 para SNR=0.03 vs 9/255 para SNR=6.20,
        # mismo instrumento, mismo ruido fisico) -- con escala fija el piso
        # se ve igual en todos los archivos, y ya no depende de cuan
        # brillante sea el burst de ESE archivo en particular.
        centro, medio_ancho = escala_fija
        lo, hi = centro - medio_ancho, centro + medio_ancho
        norm01 = np.clip((data - lo) / (hi - lo), 0.0, 1.0)
        if nbits == 8:
            norm = (norm01 * 255).astype(np.uint8)
        else:
            norm = (norm01 * 65535).astype(np.uint16)
    elif nbits == 8:
        dmin, dmax = float(np.nanmin(data)), float(np.nanmax(data))
        drange = dmax - dmin if dmax > dmin else 1.0
        norm = ((data - dmin) / drange * 255).astype(np.uint8)
    elif nbits == 16:
        dmin, dmax = float(np.nanmin(data)), float(np.nanmax(data))
        drange = dmax - dmin if dmax > dmin else 1.0
        norm = ((data - dmin) / drange * 65535).astype(np.uint16)
    elif nbits == 32:
        norm = data.astype(np.float32)
    else:
        raise ValueError(f"nbits={nbits} no soportado")

    # SIGPROC: frecuencias descendentes, filas = muestras de tiempo
    data_fil = norm[::-1, :].T.copy()

    tsamp = float(times[1] - times[0])
    fch1 = float(freqs[-1])
    foff = float(freqs[0] - freqs[-1]) / (len(freqs) - 1)

    def wraw_str(buf, s):
        b = s.encode("ascii")
        return buf + struct.pack("<i", len(b)) + b

    def wlabel_str(buf, label, value):
        buf = wraw_str(buf, label)
        return wraw_str(buf, value)

    def wlabel_num(buf, label, value, fmt):
        buf = wraw_str(buf, label)
        return buf + struct.pack(fmt, value)

    hdr = wraw_str(b"", "HEADER_START")
    hdr = wlabel_str(hdr, "source_name", source_name[:80])
    hdr = wlabel_num(hdr, "machine_id", 0, "<i")
    hdr = wlabel_num(hdr, "telescope_id", 0, "<i")
    hdr = wlabel_num(hdr, "data_type", 1, "<i")
    hdr = wlabel_num(hdr, "nbits", nbits, "<i")
    hdr = wlabel_num(hdr, "nchans", len(freqs), "<i")
    hdr = wlabel_num(hdr, "fch1", fch1, "<d")
    hdr = wlabel_num(hdr, "foff", foff, "<d")
    hdr = wlabel_num(hdr, "nifs", 1, "<i")
    hdr = wlabel_num(hdr, "tstart", 58000.0, "<d")
    hdr = wlabel_num(hdr, "tsamp", tsamp, "<d")
    hdr = wraw_str(hdr, "HEADER_END")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(hdr)
        f.write(data_fil.tobytes())
    return out_path.stat().st_size


def dispersion_delay_s(dm, freq_lo_mhz, freq_hi_mhz, dm_const_seconds):
    return dm_const_seconds * dm * (1.0 / freq_lo_mhz ** 2 - 1.0 / freq_hi_mhz ** 2)


def compute_pad_samples(tsamp, freq_lo_mhz, freq_hi_mhz, dm_const_seconds):
    delay_max = dispersion_delay_s(DM_SEARCH_MAX, freq_lo_mhz, freq_hi_mhz, dm_const_seconds)
    return int(np.ceil(delay_max / tsamp)) + PAD_GUARD_SAMPLES


def pad_with_noise(data_full, tsamp, freq_lo_mhz, freq_hi_mhz, dm_const_seconds):
    pad_samples = compute_pad_samples(tsamp, freq_lo_mhz, freq_hi_mhz, dm_const_seconds)

    nchan = data_full.shape[0]
    med = np.median(data_full, axis=1, keepdims=True)
    mad = np.median(np.abs(data_full - med), axis=1, keepdims=True)
    sigma = np.where(mad > 0, mad * 1.4826, np.std(data_full, axis=1, keepdims=True))
    sigma = np.where(sigma > 0, sigma, 1.0)

    rng = np.random.default_rng()
    noise_before = med + sigma * rng.standard_normal((nchan, pad_samples))
    noise_after = med + sigma * rng.standard_normal((nchan, pad_samples))

    data_padded = np.concatenate([noise_before, data_full, noise_after], axis=1).astype(data_full.dtype)
    return data_padded, pad_samples


def _scalar(bp, key, default=None):
    v = bp.get(key, default)
    if isinstance(v, (list, np.ndarray)):
        return float(v[0]) if len(v) else default
    return v


def load_meta(npz_path):
    d = np.load(npz_path, allow_pickle=True)
    meta = d["metadata"]
    if hasattr(meta, "item"):
        meta = meta.item()
    bp = d["burst_parameters"]
    if hasattr(bp, "item"):
        bp = bp.item()

    num_freq = int(meta["num_freq"])
    freqs = meta["freqs_bin0"] + meta["res_freq"] * np.arange(num_freq)
    tsamp = float(meta["res_time"])
    dm_const_seconds = float(meta["dm_const_seconds"])

    pad_before = compute_pad_samples(tsamp, float(freqs[0]), float(freqs[-1]), dm_const_seconds)
    arrival_time_s = _scalar(bp, "arrival_time")
    arrival_time_in_fil_s = arrival_time_s + pad_before * tsamp if arrival_time_s is not None else None

    return {
        "dm_truth": _scalar(bp, "dm"),
        "arrival_time_in_fil_s": arrival_time_in_fil_s,
        "pad_samples_before": pad_before,
        "burst_width_s": _scalar(bp, "burst_width"),
        "snr_peak_truth": _scalar(bp, "snr_peak"),
        "snr_integrated_truth": _scalar(bp, "snr_integrated"),
        "num_freq": num_freq,
        "res_time_s": tsamp,
    }, meta, bp


def convert_one(npz_path, out_path, nbits, name):
    d = np.load(npz_path, allow_pickle=True)
    data_full = np.asarray(d["data_full"], dtype=np.float64)
    info, meta, bp = load_meta(npz_path)

    num_freq = info["num_freq"]
    freqs = meta["freqs_bin0"] + meta["res_freq"] * np.arange(num_freq)
    tsamp = info["res_time_s"]
    dm_const_seconds = float(meta["dm_const_seconds"])

    data_padded, pad_before = pad_with_noise(
        data_full, tsamp, float(freqs[0]), float(freqs[-1]), dm_const_seconds
    )
    num_time_padded = data_padded.shape[1]
    times_padded = np.arange(num_time_padded) * tsamp

    sigma_teo = noise_std_teorico(meta)
    escala_fija = (0.0, N_SIGMA_CLIP * sigma_teo)  # sin RFI: un solo sigma, uniforme entre canales
    size = write_fil(data_padded, freqs, times_padded, out_path, nbits=nbits, source_name=name,
                      escala_fija=escala_fija)
    info["fil_size_bytes"] = size
    info["cuantizacion_sigma_teo_jy"] = sigma_teo
    info["cuantizacion_n_sigma"] = N_SIGMA_CLIP
    return info


def convertir_positivos(nbits, limit, src_root, out_dir):
    resumen_csv = src_root / "frb_simulados_resumen.csv"
    manifest_csv = out_dir / "manifest.csv"
    if not resumen_csv.exists():
        print(f"  (omitido: no existe {resumen_csv})")
        return
    with open(resumen_csv) as f:
        rows = list(csv.DictReader(f))
    if limit:
        rows = rows[:limit]

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows = []
    n_ok, n_err = 0, 0
    for i, row in enumerate(rows, 1):
        name = row["name"]
        catalogo = row["catalogo"]
        npz_path = src_root / row["npz_path"]
        out_path = out_dir / catalogo / f"{catalogo}_{name}.fil"
        try:
            info = convert_one(npz_path, out_path, nbits, name)
            n_ok += 1
            if n_ok % 100 == 0:
                print(f"  [positivos {i}/{len(rows)}] convertidos={n_ok} errores={n_err}", flush=True)
        except Exception as e:
            print(f"  [positivos {i}/{len(rows)}] ERROR {name}: {e}", file=sys.stderr)
            n_err += 1
            continue
        manifest_rows.append({
            "name": name, "catalogo": catalogo, "fil_path": str(out_path), "npz_path": str(npz_path),
            "dm_truth": info["dm_truth"], "arrival_time_in_fil_s": info["arrival_time_in_fil_s"],
            "pad_samples_before": info["pad_samples_before"], "burst_width_s": info["burst_width_s"],
            "snr_peak_truth": info["snr_peak_truth"], "snr_integrated_truth": info["snr_integrated_truth"],
            "res_time_s": info["res_time_s"], "dm_search_min": DM_SEARCH_MIN, "dm_search_max": DM_SEARCH_MAX,
            "cuantizacion_sigma_teo_jy": info["cuantizacion_sigma_teo_jy"],
            "cuantizacion_n_sigma": info["cuantizacion_n_sigma"],
        })

    if manifest_rows:
        with open(manifest_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()))
            w.writeheader()
            w.writerows(manifest_rows)
    print(f"✓ Positivos: {n_ok} convertidos, {n_err} errores -> {manifest_csv}")


def convertir_negativos(nbits, limit, src_root, out_dir):
    manifest_in = src_root / "manifest_negativos.csv"
    manifest_csv = out_dir / "manifest.csv"
    if not manifest_in.exists():
        print(f"  (omitido: no existe {manifest_in} -- corré 02_generar_negativos.py primero)")
        return
    with open(manifest_in) as f:
        rows = list(csv.DictReader(f))
    if limit:
        rows = rows[:limit]

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows = []
    n_ok, n_err = 0, 0
    for i, row in enumerate(rows, 1):
        name = row["name"]
        npz_path = src_root / row["npz_path"]
        out_path = out_dir / f"{name}.fil"
        try:
            info = convert_one(npz_path, out_path, nbits, name)
            n_ok += 1
            if n_ok % 100 == 0:
                print(f"  [negativos {i}/{len(rows)}] convertidos={n_ok} errores={n_err}", flush=True)
        except Exception as e:
            print(f"  [negativos {i}/{len(rows)}] ERROR {name}: {e}", file=sys.stderr)
            n_err += 1
            continue
        manifest_rows.append({
            "name": name, "name_origen": row["name_origen"], "fil_path": str(out_path),
            "dm_truth": info["dm_truth"], "dm_index_usado": row["dm_index_usado"],
            "arrival_time_in_fil_s": info["arrival_time_in_fil_s"],
            "pad_samples_before": info["pad_samples_before"], "burst_width_s": info["burst_width_s"],
            "snr_peak_truth": info["snr_peak_truth"], "snr_integrated_truth": info["snr_integrated_truth"],
            "res_time_s": info["res_time_s"], "dm_search_min": DM_SEARCH_MIN, "dm_search_max": DM_SEARCH_MAX,
            "cuantizacion_sigma_teo_jy": info["cuantizacion_sigma_teo_jy"],
            "cuantizacion_n_sigma": info["cuantizacion_n_sigma"],
        })

    if manifest_rows:
        with open(manifest_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()))
            w.writeheader()
            w.writerows(manifest_rows)
    print(f"✓ Negativos: {n_ok} convertidos, {n_err} errores -> {manifest_csv}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--nbits", type=int, default=8)
    ap.add_argument("--skip-negativos", action="store_true")
    ap.add_argument("--positivos-dir", type=Path, default=HERE / "outputs" / "positivos",
                     help="Carpeta con frb_simulados_resumen.csv (salida de 01_ o 04_). "
                          "Default: outputs/positivos")
    ap.add_argument("--negativos-dir", type=Path, default=HERE / "outputs" / "negativos",
                     help="Carpeta con manifest_negativos.csv (salida de 02_ o 04_). "
                          "Default: outputs/negativos")
    args = ap.parse_args()

    pos_out_dir = args.positivos_dir.parent / f"{args.positivos_dir.name}_fil"
    neg_out_dir = args.negativos_dir.parent / f"{args.negativos_dir.name}_fil"

    print("Convirtiendo positivos...")
    convertir_positivos(args.nbits, args.limit, args.positivos_dir, pos_out_dir)

    if not args.skip_negativos:
        print("\nConvirtiendo negativos...")
        convertir_negativos(args.nbits, args.limit, args.negativos_dir, neg_out_dir)


if __name__ == "__main__":
    main()
