#!/usr/bin/env python
"""Búsqueda de FRBs con pirate (dedispersor en árbol en GPU) sobre un .fil SIGPROC de 8 bits.

No pasa por int4 ni por ASDF: cada chunk de nt_in muestras se normaliza por canal
((x - media) / sigma, en float16) y se escribe directo en el buffer de entrada del
GpuDedisperser. Los pesos de S/N se calculan con varianza 1 por canal.

Salida: .npy con columnas (dm, t_top_s, snr, ancho_muestras), con t_top_s = llegada a la
frecuencia más alta del .fil (centro del canal; 501,6 MHz en CHARTS). Es el mismo formato
que bench/pipelines.

Uso (entorno conda de pirate):
    python pirate_fil_search.py ARCHIVO.fil SALIDA.npy [--config configs/charts_search.yml]
        [--snr 5] [--dm-min 20] [--dm-max 3000] [--cola-chunks 3] [--sin-agrupar]
        [--orden-freq {auto,asc,desc}]
"""
import argparse
import json
import struct
import sys
import time
from pathlib import Path

import numpy as np

AQUI = Path(__file__).resolve().parent
CONFIG_DEFAULT = AQUI.parent / "configs" / "charts_search.yml"

_HDR_INT = {"nchans", "nbits", "nifs", "telescope_id", "machine_id", "data_type", "barycentric",
            "pulsarcentric", "nbeams", "ibeam"}
_HDR_DBL = {"tsamp", "fch1", "foff", "tstart", "src_raj", "src_dej", "az_start", "za_start", "refdm"}


def leer_header(path):
    with open(path, "rb") as f:
        raw = f.read(8192)
    hlen = raw.index(b"HEADER_END") + len(b"HEADER_END")
    hdr, i = {}, len(b"HEADER_START") + 4
    while i < hlen - len(b"HEADER_END"):
        n = struct.unpack("<i", raw[i:i + 4])[0]
        clave = raw[i + 4:i + 4 + n].decode()
        i += 4 + n
        if clave in _HDR_INT:
            hdr[clave] = struct.unpack("<i", raw[i:i + 4])[0]
            i += 4
        elif clave in _HDR_DBL:
            hdr[clave] = struct.unpack("<d", raw[i:i + 8])[0]
            i += 8
        elif clave in ("source_name", "rawdatafile"):
            m = struct.unpack("<i", raw[i:i + 4])[0]
            hdr[clave] = raw[i + 4:i + 4 + m].decode(errors="replace")
            i += 4 + m
    return hlen, hdr


class BuscadorPirate:
    """Un GpuDedisperser de 1 beam, configurado a partir del header del .fil.

    Se puede reusar para varios archivos con la misma banda/canales/tsamp
    (el estado del árbol se reinicia creando un buscador nuevo)."""

    def __init__(self, config_path, hdr, orden_freq="asc"):
        import cupy as cp
        import ksgpu
        import pirate_frb as pf
        from pirate_frb.core import BumpAllocator, CudaStreamPool

        self.cp = cp
        nchan, foff, fch1 = hdr["nchans"], hdr["foff"], hdr["fch1"]
        centros = fch1 + foff * np.arange(nchan)
        self.f_lo_borde = float(centros.min() - abs(foff) / 2)
        self.f_hi_borde = float(centros.max() + abs(foff) / 2)
        self.f_top = float(centros.max())          # referencia de t_top (centro del canal más alto)
        self.tsamp = hdr["tsamp"]
        # El .fil viene en orden descendente si foff < 0. 'asc' = el canal 0 de pirate es la
        # frecuencia más baja.
        self.invertir = (foff < 0) == (orden_freq == "asc")

        cfg = pf.DedispersionConfig.from_yaml(str(config_path))
        cfg.zone_nfreq = [nchan]
        cfg.zone_freq_edges = [self.f_lo_borde, self.f_hi_borde]
        cfg.time_sample_ms = self.tsamp * 1e3
        cfg.beams_per_gpu = 1
        cfg.beams_per_batch = 1
        cfg.num_active_batches = 1
        cfg.validate()
        self.config = cfg
        self.dm_por_muestra = cfg.dm_per_unit_delay()

        self.plan = pf.DedispersionPlan(cfg)
        self.nt_in = self.plan.nt_in
        self.ntrees = self.plan.ntrees
        self.trees = self.plan.trees
        self.pool = CudaStreamPool(1)
        self.dd = pf.GpuDedisperser(self.plan, self.pool, cuda_device_id=0, num_consumers=1,
                                    nbatches_out=1, nbatches_wt=1, initial_chunk=0)
        self._ga = BumpAllocator(ksgpu.af_gpu | ksgpu.af_zero, -1)
        self._ha = BumpAllocator(ksgpu.af_rhost | ksgpu.af_zero, -1)
        self.dd.allocate(self._ga, self._ha)
        self.dd.fill_analytic_weights(np.ones(nchan, dtype=np.float64))
        self.Dcores = list(self.dd.Dcores)
        self.seq = 0

        # Retardo (en s) desde el borde inferior de la banda hasta f_top, por unidad de DM.
        k_borde = self.tsamp / self.dm_por_muestra   # s por (pc cm^-3) entre bordes de banda
        frac = (self.f_lo_borde ** -2 - self.f_top ** -2) / (self.f_lo_borde ** -2 - self.f_hi_borde ** -2)
        self.s_por_dm_a_top = k_borde * frac

    def chunk(self, bloque_gpu_f16, umbral, agrupar=True):
        """bloque_gpu_f16: cupy (nfreq, nt_in) float16, ya normalizado y en el orden de pirate.
        Devuelve lista de (dm, t_bottom_s, snr, ancho_muestras) del chunk."""
        cp = self.cp
        from cupyx.scipy.ndimage import maximum_filter
        stream = cp.cuda.get_current_stream()
        with self.dd.get_input(self.seq, stream=stream) as arr:
            arr[0] = bloque_gpu_f16
        filas = []
        with self.dd.get_output(self.seq, stream=stream) as out:
            for it in range(self.ntrees):
                m = cp.asarray(out.out_max[it])[0].astype(cp.float32)   # (ndm, nt)
                sel = m >= umbral
                if agrupar:
                    sel &= m == maximum_filter(m, size=3, mode="constant", cval=0.0)
                idm, itm = cp.nonzero(sel)
                if idm.size == 0:
                    continue
                snr = m[idm, itm].get()
                tok = cp.asarray(out.out_argmax[it])[0][idm, itm].get()
                idm, itm = idm.get(), itm.get()
                for k in range(len(snr)):
                    fmin, fmax, tlo, thi, p = self.plan.decode_argmax(
                        int(tok[k]), it, self.Dcores[it], int(idm[k]), int(itm[k]))
                    _, _, dm, ts, w = self.plan.decode_argmax2(it, fmin, fmax, tlo, thi, p)
                    t_bot = (self.seq * self.nt_in + ts) * self.tsamp
                    filas.append((dm, t_bot, float(snr[k]), w))
        self.seq += 1
        return filas

    def a_t_top(self, dm, t_bot):
        return t_bot - dm * self.s_por_dm_a_top


def agrupar_nms(c, dt_s=0.1, dm_frac=0.1):
    """Supresión de no-máximos: recorre por S/N descendente y descarta los candidatos a
    menos de dt_s y dm_frac de uno ya aceptado (los duplicados entre árboles/chunks)."""
    if not len(c):
        return c
    c = c[np.argsort(-c[:, 2])]
    ok = []
    for row in c:
        if not any(abs(row[1] - o[1]) <= dt_s and abs(row[0] - o[0]) <= dm_frac * max(o[0], 1.0) for o in ok):
            ok.append(row)
    return np.array(ok, dtype=np.float64)


def cuantizar_int4(bloque, niveles_sigma):
    """Simula el formato de red de pirate: int4 en [-7, 7] con escala/offset por
    (canal, bloque de 256 muestras); -8 queda reservado para muestras enmascaradas.
    niveles_sigma = cuántos niveles int4 mide una sigma del ruido."""
    import cupy as cp
    nf, T = bloque.shape
    x = bloque.astype(cp.float32).reshape(nf, T // 256, 256)
    mu = x.mean(axis=2, keepdims=True)
    sd = x.std(axis=2, keepdims=True)
    esc = cp.where(sd > 0, sd / niveles_sigma, 1.0)
    q = cp.clip(cp.rint((x - mu) / esc), -7, 7)
    return (q * esc + mu).reshape(nf, T).astype(cp.float16)


def buscar_fil(path, config_path=CONFIG_DEFAULT, snr=5.0, dm_min=20.0, dm_max=3000.0,
               cola_chunks=3, agrupar=True, orden_freq="asc", semilla=0, int4_niveles_sigma=0.0):
    import cupy as cp
    hlen, hdr = leer_header(path)
    assert hdr["nbits"] == 8 and hdr.get("nifs", 1) == 1, hdr
    nchan = hdr["nchans"]
    data = np.memmap(path, dtype=np.uint8, mode="r", offset=hlen)
    data = data[: (data.size // nchan) * nchan].reshape(-1, nchan)
    nsamp = data.shape[0]

    t0 = time.time()
    b = BuscadorPirate(config_path, hdr, orden_freq=orden_freq)
    t_init = time.time() - t0

    # Estadística por canal sobre todo el archivo (un burst de S/N ~10 no la mueve).
    d_gpu = cp.asarray(np.ascontiguousarray(data))
    mu = d_gpu.mean(axis=0, dtype=cp.float32)
    sd = d_gpu.std(axis=0, dtype=cp.float32)
    inv = cp.where(sd > 0, 1.0 / cp.maximum(sd, 1e-6), 0.0).astype(cp.float32)
    if b.invertir:
        d_gpu, mu, inv = d_gpu[:, ::-1], mu[::-1], inv[::-1]
    rng = cp.random.default_rng(semilla)

    T = b.nt_in
    nchunks = -(-nsamp // T) + cola_chunks
    cands = []
    t1 = time.time()
    for i in range(nchunks):
        a, z = i * T, min((i + 1) * T, nsamp)
        bloque = cp.empty((nchan, T), dtype=cp.float16)
        n = max(0, z - a)
        if n:
            bloque[:, :n] = ((d_gpu[a:z].astype(cp.float32) - mu) * inv).T.astype(cp.float16)
        if n < T:   # cola: ruido gaussiano normalizado (canales muertos en 0)
            bloque[:, n:] = (rng.standard_normal((nchan, T - n), dtype=cp.float32)
                             * (inv > 0)[:, None]).astype(cp.float16)
        if int4_niveles_sigma:
            bloque = cuantizar_int4(bloque, int4_niveles_sigma)
        cands += b.chunk(bloque, snr, agrupar=agrupar)
    cp.cuda.get_current_stream().synchronize()
    t_busq = time.time() - t1

    c = np.array(cands, dtype=np.float64).reshape(-1, 4)
    if len(c):
        c[:, 1] = b.a_t_top(c[:, 0], c[:, 1])
        c = c[(c[:, 0] >= dm_min) & (c[:, 0] <= dm_max)]
    if agrupar:
        c = agrupar_nms(c)
    info = dict(nsamp=nsamp, nchan=nchan, nt_in=T, nchunks=nchunks, ntrees=b.ntrees,
                dm_max_arbol=[t.dm_max for t in b.trees], t_init_s=t_init, t_busqueda_s=t_busq,
                n_cands=int(len(c)), invertir=b.invertir)
    return c, info


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("fil")
    ap.add_argument("salida", help=".npy de candidatos (dm, t_top_s, snr, ancho_muestras); se escribe además .json")
    ap.add_argument("--config", default=str(CONFIG_DEFAULT))
    ap.add_argument("--snr", type=float, default=5.0)
    ap.add_argument("--dm-min", type=float, default=20.0)
    ap.add_argument("--dm-max", type=float, default=3000.0)
    ap.add_argument("--cola-chunks", type=int, default=3)
    ap.add_argument("--sin-agrupar", action="store_true")
    ap.add_argument("--orden-freq", choices=["asc", "desc"], default="asc")
    ap.add_argument("--int4", type=float, default=0.0, metavar="NIVELES_POR_SIGMA",
                    help="simular re-cuantización a int4 (p. ej. 2 o 3 niveles por sigma); 0 = float16 directo")
    a = ap.parse_args()
    c, info = buscar_fil(a.fil, a.config, a.snr, a.dm_min, a.dm_max, a.cola_chunks,
                         not a.sin_agrupar, a.orden_freq, int4_niveles_sigma=a.int4)
    np.save(a.salida, c.astype(np.float32))
    Path(a.salida).with_suffix(".json").write_text(json.dumps(info))
    print(json.dumps(info))


if __name__ == "__main__":
    sys.exit(main())
