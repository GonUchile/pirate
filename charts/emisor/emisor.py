#!/usr/bin/env python3
"""Emisor independiente: manda datos de intensidad a pirate con su protocolo de red, como lo haría un X-engine.

No depende de pirate: Python ≥ 3.9, numpy y PyYAML (cupy opcional, con `gpu: true`). Puede correr en otro PC.

  Fuentes ─► procesos de preparación ─► memoria compartida (2 búferes) ─► 1 hilo por conexión TCP ─► receptores de pirate
             (leer, RFI opcional,         [beam, canal, tiempo/2] 4 bits      (su rango de canales,
              cuantizar, máscara)          + escala/offset por minichunk       minichunk a minichunk)

Cada conexión imita un nodo X-engine que manda un rango contiguo de canales para todos los beams.
Ritmo: `tiempo-real` (un minichunk cada 256 muestras, como el X-engine; `velocidad` lo escala), `ack` (cada
conexión espera la confirmación de pirate antes del siguiente minichunk: lo más rápido que pirate acepta) o
`libre` (sin pausa).

Uso:
  python emisor/emisor.py --config configs/prueba_64b.yml --salida resultados/corrida [opciones que pisan el YAML]
  p. ej. --limite 20 --rfi --mascara rfi_sitio/mascara_m1.npy --ritmo ack --receptores 10.0.0.2:5000,10.0.0.2:5001

Salidas en --salida: asignacion.csv (qué archivo va en qué beam y desde qué muestra), emisor.json, emisor.log.
"""
import argparse
import json
import multiprocessing as mp
import os
import queue
import signal
import sys
import threading
import time
from multiprocessing import shared_memory
from pathlib import Path

import numpy as np
import yaml

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
from fuentes import CARPETAS, LectorBeam, armar_streams, relleno_precalculado, ruido_8bits  # noqa: E402
from protocolo import MUESTRAS_MINICHUNK, Conexion, empaquetar_int4, metadatos_yaml  # noqa: E402

DT_NS_PER_SEQ = 5000                                 # 5 µs por cuenta de seq; 1 ms = 200 cuentas


# ---------------------------------------------------------------------------------------------- configuración
def cargar_config(a):
    cfg = yaml.safe_load(open(a.config)) if a.config else {}
    p, red, e = cfg.get("prueba", {}), cfg.get("red", {}), cfg.get("emisor", {})
    raiz = AQUI.parent                              # el repositorio: rfi_sitio/ y rutas relativas (no depende de dónde esté el YAML)
    c = dict(
        beams=p.get("beams", 64), canales=p.get("canales", 21504), banda=p.get("banda_mhz", [299.8497764530551, 501.7502235469449]),
        muestra_ms=p.get("muestra_ms", 1.0), T=p.get("bloque_muestras", 2048),
        receptores=red.get("receptores", ["127.0.0.1:5000"]), conexiones=red.get("conexiones_por_receptor", 4),
        set_dir=os.path.expandvars(str(e.get("set_dir") or "$CHARTS_SET_DIR")), sets=e.get("sets", ["A", "B", "NEG", "P"]),
        limite=e.get("limite"), prepad_s=e.get("prepad_s", 10.0), cola_s=e.get("cola_s", 60.0),
        beams_con_datos=e.get("beams_con_datos"), relleno=e.get("relleno", "ruido"),
        cuantizacion=e.get("cuantizacion", "fija"), mascara=e.get("mascara"), rfi=e.get("rfi", {}) or {},
        ritmo=e.get("ritmo", "tiempo-real"), velocidad=e.get("velocidad", 1.0), procesos=e.get("procesos", 8),
        gpu=e.get("gpu", False), semilla=e.get("semilla", 1), rfi_dir=str(raiz / "rfi_sitio"),
        ruido_precalculado=e.get("ruido_precalculado", True),
    )
    for k in ("receptores", "conexiones", "set_dir", "limite", "beams_con_datos", "relleno", "cuantizacion", "mascara",
              "ritmo", "velocidad", "procesos", "semilla", "beams"):
        v = getattr(a, k, None)
        if v is not None:
            c[k] = v.split(",") if k == "receptores" and isinstance(v, str) else v
    if a.rfi:
        c["rfi"] = dict(c["rfi"], activa=True)
    if a.gpu:
        c["gpu"] = True
    if c["rfi"].get("activa"):
        c["cuantizacion"] = "por_canal"              # con RFI hace falta normalizar por canal (ver docs)
    if c["mascara"] and not os.path.isabs(c["mascara"]):
        c["mascara"] = str((Path.cwd() / c["mascara"]).resolve()) if (Path.cwd() / c["mascara"]).exists() else str(raiz / c["mascara"])
    return c


# ---------------------------------------------------------------------------------------- procesos de preparación
def preparar(w, beams, c, tramos, n_datos, total, shm_datos, shm_so, tareas, hechas):
    """Proceso w: prepara los beams que le tocan, bloque a bloque, en el búfer (slot) que se le indica."""
    padre = os.getppid()
    try:                                           # Linux: el kernel mata este proceso si muere el emisor principal
        import ctypes
        ctypes.CDLL(None, use_errno=True).prctl(1, signal.SIGKILL)       # 1 = PR_SET_PDEATHSIG
    except (OSError, AttributeError):
        pass                                       # otro sistema: queda la revisión de os.getppid() de abajo
    if os.getppid() != padre:                      # murió justo antes de pedirlo
        return
    nb, nch, T = c["beams"], c["canales"], c["T"]
    shm_d, shm_s = shared_memory.SharedMemory(name=shm_datos), shared_memory.SharedMemory(name=shm_so)  # mantener referencias
    nmc = T // 256
    datos = np.ndarray((2, nmc, nb, nch, 128), np.uint8, buffer=shm_d.buf)    # disposición de envío
    so = np.ndarray((2, nmc, nb, nch, 2), np.float16, buffer=shm_s.buf)
    xp = np
    if c["gpu"]:
        import cupy as xp                            # noqa: F811
    import cuantizacion as cz
    por_canal = c["cuantizacion"] == "por_canal"
    so_base = cz.escalas_base(nch, T, por_canal)
    mascara = None
    if c["mascara"]:
        f_asc = c["banda"][0] + (np.arange(nch) + 0.5) * (c["banda"][1] - c["banda"][0]) / nch
        fm = np.load(Path(c["rfi_dir"]) / "frecuencias_mhz.npy"); mk = np.load(c["mascara"]).astype(bool)
        idx = np.clip(np.searchsorted(fm, f_asc), 1, len(fm) - 1)
        idx = np.where(np.abs(fm[idx - 1] - f_asc) < np.abs(fm[idx] - f_asc), idx - 1, idx)
        mascara = mk[idx]                            # orden ascendente
        cz.aplicar_mascara(np.zeros((nch, 1), np.uint8), so_base, mascara)
    rfi = None
    if c["rfi"].get("activa"):
        sys.path.insert(0, c["rfi_dir"])
        from rfi_sitio import RfiSitio
        f_fil = c["banda"][1] - (np.arange(nch) + 0.5) * (c["banda"][1] - c["banda"][0]) / nch
        r = c["rfi"]
        rfi = RfiSitio(f_fil, umbral_lineas_db=r.get("umbral_lineas_db", 10.0), atenuacion_db=r.get("atenuacion_db", 0.0),
                       rapida=r.get("rapida", True), persistente=r.get("persistente", True), semilla=c["semilla"], xp=xp)
    relleno = relleno_precalculado(nch, T, ceros=c["relleno"] == "ceros") if n_datos < nb else None
    if relleno is not None:                          # [n, nch, T/2] -> [n, nmc, nch, 128]
        relleno = np.ascontiguousarray(relleno.reshape(len(relleno), nch, nmc, 128).transpose(0, 2, 1, 3))
    so_base_t = np.ascontiguousarray(so_base.transpose(1, 0, 2))             # [nmc, nch, 2]
    pool = ruido_8bits(T, nch, semilla=99 + w) if c.get("ruido_precalculado", True) else None
    lectores = {b: LectorBeam(tramos[b], nch, semilla=1000 + b, pool=pool, beam=b) for b in beams if b < n_datos}
    buf = np.empty((T, nch), np.uint8)
    rng = np.random.default_rng(7000 + w) if xp is np else xp.random.RandomState(7000 + w)
    while True:
        try:
            tarea = tareas.get(timeout=2)
        except queue.Empty:
            if os.getppid() != padre:              # el emisor principal murió (kill): no quedar huérfano
                return
            continue
        if tarea is None:
            return
        ichunk, slot = tarea
        for b in beams:
            if b >= n_datos:
                blq, s = relleno[(ichunk + b) % len(relleno)], (b * 2687) % nch     # canales rotados por beam
                datos[slot, :, b, :nch - s] = blq[..., s:, :]
                datos[slot, :, b, nch - s:] = blq[..., :s, :]
                so[slot, :, b] = so_base_t
                continue
            lectores[b].bloque(ichunk * T, T, buf)
            g = buf
            if rfi is not None:
                g = rfi.aplicar(buf, ichunk * T, rng=rng)
            q = cz.por_canal(g, xp) if por_canal else cz.fija(g, xp)
            if mascara is not None:
                q[xp.asarray(mascara), :] = cz.FALTANTE
            p = empaquetar_int4(q)
            p = p.get() if xp is not np else p                                 # [nch, T/2]
            datos[slot, :, b] = p.reshape(nch, nmc, 128).transpose(1, 0, 2)
            so[slot, :, b] = so_base_t
        hechas.put((w, ichunk))


# ------------------------------------------------------------------------------------------------- principal
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", help="YAML de la prueba (configs/prueba_64b.yml)")
    ap.add_argument("--salida", required=True, type=Path)
    ap.add_argument("--receptores", help="lista host:puerto separada por comas")
    ap.add_argument("--conexiones", type=int, help="conexiones por receptor")
    ap.add_argument("--set-dir"); ap.add_argument("--limite", type=int)
    ap.add_argument("--beams", type=int); ap.add_argument("--beams-con-datos", type=int)
    ap.add_argument("--relleno", choices=["ruido", "ceros"])
    ap.add_argument("--cuantizacion", choices=["fija", "por_canal"])
    ap.add_argument("--mascara"); ap.add_argument("--rfi", action="store_true", help="agregar la RFI del sitio")
    ap.add_argument("--ritmo", choices=["tiempo-real", "ack", "libre"]); ap.add_argument("--velocidad", type=float)
    ap.add_argument("--procesos", type=int); ap.add_argument("--gpu", action="store_true"); ap.add_argument("--semilla", type=int)
    a = ap.parse_args()
    c = cargar_config(a)
    a.salida.mkdir(parents=True, exist_ok=True)
    log = open(a.salida / "emisor.log", "a")

    def info(msg):
        line = f"{time.strftime('%F %T')} emisor: {msg}"
        print(line, flush=True); log.write(line + "\n"); log.flush()

    nb, nch, T = c["beams"], c["canales"], c["T"]
    n_datos = nb if c["beams_con_datos"] is None else max(1, min(c["beams_con_datos"], nb))
    seq_por_muestra = int(round(c["muestra_ms"] * 1e6 / DT_NS_PER_SEQ))
    nmc = T // MUESTRAS_MINICHUNK
    if not c["set_dir"] or "$" in c["set_dir"] or not (Path(c["set_dir"]) / CARPETAS[c["sets"][0]] / "manifest.csv").exists():
        sys.exit(f"ERROR: no encuentro los datos en set_dir={c['set_dir']!r}. export CHARTS_SET_DIR=/carpeta/del/grupo3 "
                 "(con A_snr_fijo/, B_amplitud_fija/, ... y su manifest.csv), o --set-dir, o emisor.set_dir en el YAML")
    tramos, asign, total = armar_streams(c["set_dir"], c["sets"], n_datos, int(c["prepad_s"] * 1000),
                                         int(c["cola_s"] * 1000), c["semilla"], c["limite"], T)
    nchunks = total // T
    with open(a.salida / "asignacion.csv", "w", newline="") as f:
        import csv
        campos = list(dict.fromkeys(k for r in asign for k in r))
        w = csv.DictWriter(f, fieldnames=campos); w.writeheader(); w.writerows(asign)
    info(f"{len(asign)} archivos en {n_datos} de {nb} beams; {total / 1000:.0f} s por beam ({nchunks} bloques de {T}); "
         f"{nch} canales; cuantización {c['cuantizacion']}; RFI {'sí' if c['rfi'].get('activa') else 'no'}; "
         f"máscara {c['mascara'] or 'no'}; ritmo {c['ritmo']} ×{c['velocidad']}")

    # memoria compartida: 2 búferes de [beam, canal, T/2] y escalas/offsets
    # disposición de envío [slot, minichunk, beam, canal, ...]: el rango de canales de una conexión en un minichunk
    # queda como un trozo contiguo por beam
    tam_d, tam_s = 2 * nmc * nb * nch * 128, 2 * nmc * nb * nch * 2 * 2
    shm_d, shm_s = shared_memory.SharedMemory(create=True, size=tam_d), shared_memory.SharedMemory(create=True, size=tam_s)
    datos = np.ndarray((2, nmc, nb, nch, 128), np.uint8, buffer=shm_d.buf)
    so = np.ndarray((2, nmc, nb, nch, 2), np.float16, buffer=shm_s.buf)
    ctx = mp.get_context("spawn")
    P = max(1, min(c["procesos"], nb))
    tareas = [ctx.Queue() for _ in range(P)]; hechas = ctx.Queue()
    procs = [ctx.Process(target=preparar, daemon=True,
                         args=(w, list(range(w, nb, P)), c, tramos, n_datos, total, shm_d.name, shm_s.name, tareas[w], hechas))
             for w in range(P)]
    for p in procs:
        p.start()

    # conexiones: cada una manda un rango contiguo de canales (canal 0 = frecuencia más baja)
    nconn = len(c["receptores"]) * c["conexiones"]
    cortes = np.linspace(0, nch, nconn + 1).astype(int)

    def preparar_chunk(ichunk):
        for q in tareas:
            q.put((ichunk, ichunk % 2))

    def esperar_chunk(ichunk):
        n = 0
        while n < P:
            try:
                w_, ic = hechas.get(timeout=5)
            except queue.Empty:
                muertos = [w for w, p in enumerate(procs) if not p.is_alive()]
                if muertos:
                    raise SystemExit(f"ERROR: murieron los procesos de preparación {muertos} (ver el error arriba)")
                continue
            assert ic == ichunk, (ic, ichunk)
            n += 1

    # el primer bloque se prepara ANTES de fijar la hora del seq 0: si no, los datos quedan fechados antes de salir
    # (la lectura en frío de los primeros .fil tarda varios segundos) y la latencia medida en la búsqueda se infla
    t_a = time.monotonic()
    preparar_chunk(0)
    esperar_chunk(0)
    info(f"primer bloque listo en {time.monotonic() - t_a:.1f} s")
    unix0 = time.time_ns()
    t0 = [time.monotonic()]                          # reloj del ritmo tiempo-real: seq 0 sale en unix0
    conns = []
    for k in range(nconn):
        ch0, ch1 = int(cortes[k]), int(cortes[k + 1])
        meta = metadatos_yaml([nch], c["banda"], list(range(nb)), list(range(ch0, ch1)), unix0, DT_NS_PER_SEQ, seq_por_muestra)
        conns.append((Conexion(c["receptores"][k % len(c["receptores"])], meta, ack=c["ritmo"] == "ack"), ch0, ch1))
    info(f"{nconn} conexiones abiertas hacia {', '.join(c['receptores'])}")

    errores = []

    def enviar(k, ichunk, slot):
        con, ch0, ch1 = conns[k]
        try:
            for m in range(nmc):
                g = ichunk * nmc + m
                if c["ritmo"] == "tiempo-real":
                    espera = t0[0] + g * (MUESTRAS_MINICHUNK * c["muestra_ms"] / 1000) / c["velocidad"] - time.monotonic()
                    if espera > 0:
                        time.sleep(espera)
                seq = (ichunk * T + m * MUESTRAS_MINICHUNK) * seq_por_muestra
                con.enviar_minichunk(seq, so[slot, m, :, ch0:ch1], datos[slot, m, :, ch0:ch1])
        except Exception as ex:                     # noqa: BLE001
            errores.append(f"{con.direccion}: {ex!r}")

    t_ini = time.monotonic()
    t_prep = t_env = 0.0                             # espera a la preparación y tiempo de envío, acumulados
    try:
        for ichunk in range(nchunks):
            ta = time.monotonic()
            if ichunk > 0:
                esperar_chunk(ichunk)
            tb = time.monotonic(); t_prep += tb - ta
            if ichunk + 1 < nchunks:
                preparar_chunk(ichunk + 1)          # el siguiente se prepara mientras se envía este
            hilos = [threading.Thread(target=enviar, args=(k, ichunk, ichunk % 2)) for k in range(nconn)]
            for h in hilos:
                h.start()
            for h in hilos:
                h.join()
            t_env += time.monotonic() - tb
            if errores:
                raise RuntimeError("; ".join(errores[:3]))
            if (ichunk + 1) % 10 == 0 or ichunk + 1 == nchunks:
                dt = time.monotonic() - t_ini
                info(f"bloque {ichunk + 1}/{nchunks}, {dt:.0f} s, ritmo {(ichunk + 1) * T / 1000 / dt:.2f}× tiempo real; "
                     f"por bloque: esperando la preparación {t_prep / (ichunk + 1):.2f} s, enviando {t_env / (ichunk + 1):.2f} s")
    finally:
        for q in tareas:
            q.put(None)
        for con, _, _ in conns:
            con.cerrar()
        dt = time.monotonic() - t_ini
        resumen = dict(nbeams=nb, beams_con_datos=n_datos, nchan=nch, T=T, total_muestras=total, nchunks=nchunks,
                       archivos=len(asign), prepad=int(c["prepad_s"] * 1000), cola=int(c["cola_s"] * 1000),
                       dt_ns_per_seq=DT_NS_PER_SEQ, seq_per_frb_time_sample=seq_por_muestra, zone_freq_edges=c["banda"],
                       receptores=c["receptores"], conexiones=nconn, cuantizacion=c["cuantizacion"],
                       escala_sigma=0.5, offset_sigma=0.0 if c["cuantizacion"] == "por_canal" else 0.125,
                       rfi_sitio=c["rfi"] if c["rfi"].get("activa") else None, mascara=c["mascara"], ritmo=c["ritmo"],
                       velocidad=c["velocidad"], duracion_s=dt, ritmo_medido=nchunks * T / 1000 / max(dt, 1e-9),
                       minichunks_descartados_por_pirate=sum(x[0].descartados for x in conns) if c["ritmo"] == "ack" else None,
                       inicio_unix=unix0 / 1e9, errores=errores)
        json.dump(resumen, open(a.salida / "emisor.json", "w"), indent=1)
        info(f"fin: {nchunks} bloques en {dt:.0f} s ({resumen['ritmo_medido']:.2f}× tiempo real)")
        shm_d.close(); shm_s.close(); shm_d.unlink(); shm_s.unlink()


if __name__ == "__main__":
    main()
