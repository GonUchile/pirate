#!/usr/bin/env python
"""Grouper del testbench, versión 1: búsqueda de picos real en el mapa DM-tiempo de pirate.

Por chunk, por batch de beams y por árbol (todo en la GPU salvo la agrupación final, que es chica):
  1. Regiones: celdas de out_max con S/N >= --umbral-celda, unidas por vecindad de 8 (sin unir beams).
     De cada región se toma su celda máxima como candidato (cupyx.scipy.ndimage.label + lexsort).
  2. Se guardan los --max-cand candidatos de mayor S/N por beam y se decodifican con create_events
     (DM, llegada al borde inferior, ancho exactos).
  3. Agrupación por burst: en el mapa de pirate un burst es una diagonal larga, y todas sus celdas
     comparten la llegada a la frecuencia alta, t_top = t_evento - K*DM (K: retardo por unidad de DM
     entre el borde inferior y 501,6 MHz). Dentro de cada beam, de mayor a menor S/N, un candidato se
     descarta si un evento ya aceptado (de este chunk o del anterior, de cualquier árbol) tiene
     |t_top - t_top'| <= --tol-t y S/N mayor o igual. Así una diagonal partida en varias regiones queda como
     un solo evento; si el pico real llega en un chunk posterior a un trozo débil ya enviado, se envía igual
     (puede quedar un duplicado, pero con el DM correcto).
  4. Se mandan al sifter hasta --k eventos por beam con S/N >= --umbral (máx. --k-total por chunk).
     Con --log-todos ARCHIVO se escriben además TODOS los eventos agrupados con S/N >= --umbral, sin los topes
     --k y --k-total, en el formato del sifter.log (para contar todo lo que se detecta sin saturar el sifter).

Para comparar en igualdad de condiciones, sobre el MISMO mapa se calcula también el resultado del
grouper v0 (un máximo por chunk y beam, umbral --umbral) y se escribe en eventos_v0.log con el mismo
formato que el sifter.log del toy_sifter. Las ventanas del mapa se guardan como en el v0, para los
eventos del v1.

Uso (entorno de pirate):
  python grouper_v1.py 127.0.0.1:7000 -s 127.0.0.1:7500 -t 5 --salida DIR
"""
import argparse
import contextlib
import itertools
import json
import queue
import sys
import threading
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from grouper_testbench import VDM, PATCH, escritor    # noqa: E402

SEQ = 200                     # cuentas de fpga por muestra (1 ms)


def retardo_borde_a_top(cfg_yaml):
    """Segundos por unidad de DM entre el borde inferior de la banda y 501,6 MHz."""
    import pirate_frb as pf
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".yml", delete=False) as f:
        f.write(cfg_yaml)
    cfg = pf.DedispersionConfig.from_yaml(f.name)
    f_lo, f_hi = cfg.zone_freq_edges[0], cfg.zone_freq_edges[-1]
    s_por_dm_banda = 1e-3 / cfg.dm_per_unit_delay()
    return s_por_dm_banda * (f_lo ** -2 - 501.6 ** -2) / (f_lo ** -2 - f_hi ** -2)


def linea_sifter(ev, i):
    return (f"    beam_id={int(ev.beam_ids[i])}, dm={float(ev.dms[i]):.4g}, "
            f"fpga_timestamp={int(ev.fpga_timestamps[i])}, width={float(ev.widths_ms[i]):.4g} ms, "
            f"subband=[{float(ev.subband_freqs_lo_MHz[i]):.1f},{float(ev.subband_freqs_hi_MHz[i]):.1f}] MHz, "
            f"snr={float(ev.snrs[i]):.4g}, tree_index={int(ev.tree_index[i])}")


class Reloj:
    """Tiempo por etapa dentro de un chunk. Con perfil=True sincroniza la GPU en cada marca (medición exacta)."""

    ETAPAS = ("espera_pirate", "copias", "v0", "regiones", "v0_decod", "seleccion", "v1_decod", "agrupar",
              "enviar", "guardar", "cola_espera", "parches")

    def __init__(self, perfil, archivo):
        import cupy as cp
        self.cp, self.perfil = cp, perfil
        self.f = open(archivo, "w") if perfil else None
        if self.f:
            self.f.write("ichunk,total_ms," + ",".join(e + "_ms" for e in self.ETAPAS) + ",n_cand,n_v1\n")

    def nuevo(self):
        self.acum = dict.fromkeys(self.ETAPAS, 0.0)
        self.t0 = self.t = time.perf_counter()

    def marca(self, etapa):
        if self.perfil:
            self.cp.cuda.get_current_stream().synchronize()
        ahora = time.perf_counter()
        self.acum[etapa] += ahora - self.t
        self.t = ahora

    def fin(self, ichunk, n_cand, n_v1):
        total = time.perf_counter() - self.t0
        if self.f:
            self.f.write(f"{ichunk},{1e3 * total:.2f}," + ",".join(f"{1e3 * self.acum[e]:.2f}" for e in self.ETAPAS)
                         + f",{n_cand},{n_v1}\n")
            self.f.flush()
        return total


def correr(grouper, sifter, a):
    import cupy as cp
    import cupyx.scipy.ndimage as ndi
    from pirate_frb.rpc import FrbSifterEvents
    from pirate_frb.utils import atomic_print

    nbeams, nbatches = grouper.total_beams, grouper.nbatches
    bpb = nbeams // nbatches
    beam_set_id = grouper.xengine_metadata.beamset
    K = retardo_borde_a_top(grouper.dedispersion_config_yaml_string)
    if sifter is not None:
        sifter.send_configuration(
            pirate_yaml=grouper.dedispersion_config_yaml_string,
            xengine_yaml=grouper.xengine_metadata_yaml_string,
            dedispersion_plan_yaml=grouper.dedispersion_plan_yaml_string,
            grouper_yaml={"grouper_testbench": "v1", "umbral_celda": a.umbral_celda, "tol_t": a.tol_t, "k": a.k},
            search_ip_addr=grouper.search_ip_addr)

    a.salida.mkdir(parents=True, exist_ok=True)
    log_todos = open(a.log_todos, "w") if a.log_todos else None
    if log_todos:
        log_todos.write("eventos del grouper v1 sin topes --k/--k-total (todos los agrupados con S/N > umbral)\n")
    log_v0 = open(a.salida.parent / "eventos_v0.log", "w")
    log_v0.write("eventos del grouper v0 (máximo por chunk y beam) calculados sobre el mismo mapa que el v1\n")
    cola = queue.Queue(maxsize=64)
    hilo = threading.Thread(target=escritor, args=(cola, a.salida), daemon=True)
    hilo.start()
    estructura = np.zeros((3, 3, 3), bool)
    estructura[1] = True                          # vecindad de 8 en (DM, tiempo), sin unir beams
    estructura = cp.asarray(estructura)

    anillo = tokens = None
    fpga_inicio, pendientes = {}, None
    previos = {}                                  # beam -> lista de (t_top, snr) aceptados en el chunk anterior
    reloj = Reloj(a.perfil, a.salida.parent / "tiempos_grouper.csv")
    meta_info = {"nbeams": nbeams, "nbatches": nbatches, "VDM": VDM, "PATCH": PATCH, "version": "v1",
                 "full_steady_ichunk": int(grouper.full_steady_ichunk), "args": {k: str(v) for k, v in vars(a).items()}}

    def guardar(pend, ichunk_ev):
        if pend is None or not len(pend["snr"]):
            return
        ventanas, validos = [], []
        for k in range(len(pend["snr"])):
            it, b, idm = int(pend["tree"][k]), int(pend["ibeam"][k]), int(pend["idm"][k])
            ring = anillo[it]
            ndm, nt = ring.shape[2], ring.shape[3]
            orden = [(ichunk_ev + d) % 3 for d in (-1, 0, 1)]
            disp = [ichunk_ev - 1 >= 0, True, (ichunk_ev + 1) in fpga_inicio]
            lo, hi = max(0, idm - VDM), min(ndm, idm + VDM)
            v = cp.zeros((2 * VDM, 3 * nt), dtype=cp.float16)
            for j, (slot, ok) in enumerate(zip(orden, disp)):
                if ok:
                    v[lo - (idm - VDM): hi - (idm - VDM), j * nt:(j + 1) * nt] = ring[slot, b, lo:hi, :].astype(cp.float16)
            ventanas.append(v)
            validos.append((lo - (idm - VDM), hi - (idm - VDM), int(disp[0]), int(disp[2])))
        datos = {k: v for k, v in pend.items() if k != "tok"}
        datos["ventana"] = cp.asnumpy(cp.stack(ventanas))
        datos["tokens"] = pend["tok"]
        datos["valido"] = np.array(validos, dtype=np.int32)
        datos["chunk_fpga_start"] = np.int64(fpga_inicio[ichunk_ev])
        datos["meta"] = np.array(json.dumps(meta_info))
        reloj.marca("guardar")
        cola.put((f"chunk_{ichunk_ev:06d}.npz", datos))
        reloj.marca("cola_espera")

    ichunk = -1
    try:
        for ichunk in itertools.count():
            reloj.nuevo()
            # v0 sobre el mismo mapa
            v0_max = cp.full((nbeams,), -cp.inf, dtype=cp.float32)
            v0_tree = cp.full((nbeams,), -1, dtype=cp.int64)
            v0_idm = cp.full((nbeams,), -1, dtype=cp.int64)
            v0_it = cp.full((nbeams,), -1, dtype=cp.int64)
            v0_tok = cp.zeros((nbeams,), dtype=cp.uint32)
            cand = []                             # por (batch, árbol): (beam, árbol, idm, itime, snr, token)

            for ibatch in range(nbatches):
                beam0 = ibatch * bpb
                sl = slice(beam0, beam0 + bpb)
                reloj.marca("copias")                # (lo que quede de la vuelta anterior)
                with grouper.get_output(ichunk, ibatch) as outputs:
                    reloj.marca("espera_pirate")
                    if anillo is None:
                        anillo = [cp.zeros((3, nbeams) + t.shape[1:], dtype=cp.float32) for t in outputs.out_max]
                        tokens = [cp.zeros((nbeams,) + t.shape[1:], dtype=cp.uint32) for t in outputs.out_argmax]
                        meta_info["formas"] = [list(t.shape[1:]) for t in outputs.out_max]
                    for itree, tree_out in enumerate(outputs.out_max):
                        tok = outputs.out_argmax[itree]
                        if not a.sin_ventanas:
                            anillo[itree][ichunk % 3, sl] = tree_out
                            tokens[itree][sl] = tok
                        reloj.marca("copias")
                        b, ndm, nt = tree_out.shape
                        # --- v0 ---
                        flat = tree_out.reshape(b, ndm * nt)
                        bmax, barg = flat.max(axis=1), flat.argmax(axis=1)
                        btok = tok.reshape(b, ndm * nt)[cp.arange(b), barg]
                        upd = bmax > v0_max[sl]
                        v0_max[sl] = cp.where(upd, bmax, v0_max[sl])
                        v0_tree[sl] = cp.where(upd, itree, v0_tree[sl])
                        v0_idm[sl] = cp.where(upd, barg // nt, v0_idm[sl])
                        v0_it[sl] = cp.where(upd, barg % nt, v0_it[sl])
                        v0_tok[sl] = cp.where(upd, btok, v0_tok[sl])
                        reloj.marca("v0")
                        # --- v1: máximo de cada región conexa sobre el umbral ---
                        mask = tree_out >= a.umbral_celda
                        lab, nreg = ndi.label(mask, structure=estructura)
                        if int(nreg) == 0:
                            reloj.marca("regiones")
                            continue
                        idx = cp.flatnonzero(mask.ravel())
                        labs = lab.ravel()[idx]
                        vals = tree_out.ravel()[idx]
                        orden = cp.lexsort(cp.stack((-vals, labs)))
                        labs_o = labs[orden]
                        primero = cp.ones(labs_o.shape, dtype=bool)
                        primero[1:] = labs_o[1:] != labs_o[:-1]
                        g = idx[orden[primero]]                       # índice plano de la celda máxima de cada región
                        bb, rem = g // (ndm * nt), g % (ndm * nt)
                        cand.append((bb + beam0, cp.full(g.shape, itree), rem // nt, rem % nt,
                                     tree_out.ravel()[g], tok.ravel()[g]))
                        reloj.marca("regiones")

            # eventos v0 (a la bitácora de comparación)
            ib0 = cp.nonzero(v0_max > a.umbral)[0]
            ev0 = grouper.create_events(ichunk, v0_tree[ib0], ib0, v0_idm[ib0], v0_it[ib0], v0_max[ib0], v0_tok[ib0])
            fpga_inicio[ichunk] = int(ev0.chunk_fpga_start)
            for viejo in [c for c in fpga_inicio if c < ichunk - 3]:
                del fpga_inicio[viejo]
            lineas = [f"toy_sifter beamset={beam_set_id} fpga=[{ev0.chunk_fpga_start}:{ev0.chunk_fpga_end}] "
                      f"coarse_snr_max={float(v0_max.max()):.4g}"]
            lineas += [linea_sifter(ev0, i) for i in range(len(ev0))]
            log_v0.write("\n".join(lineas) + "\n")
            reloj.marca("v0_decod")
            n_cand = 0

            # candidatos v1: los max_cand de mayor S/N por beam
            if cand:
                cb, ct, cd, ci, cs, ck = (cp.concatenate([c[j] for c in cand]) for j in range(6))
                o = cp.lexsort(cp.stack((-cs, cb)))
                cb, ct, cd, ci, cs, ck = cb[o], ct[o], cd[o], ci[o], cs[o], ck[o]
                inicio = cp.ones(cb.shape, dtype=bool)
                inicio[1:] = cb[1:] != cb[:-1]
                grupo = cp.cumsum(inicio) - 1                     # índice del beam en el orden
                rango = cp.arange(cb.size) - cp.flatnonzero(inicio)[grupo]   # posición dentro de su beam
                sel = rango < a.max_cand
                cb, ct, cd, ci, cs, ck = cb[sel], ct[sel], cd[sel], ci[sel], cs[sel], ck[sel]
                n_cand = int(cb.size)
                reloj.marca("seleccion")
                evc = grouper.create_events(ichunk, ct, cb, cd, ci, cs, ck)
                hb, ht, hd, hi = (cp.asnumpy(x) for x in (cb, ct, cd, ci))
                snr = np.asarray(evc.snrs)
                t_top = np.asarray(evc.fpga_timestamps) / SEQ * 1e-3 - np.asarray(evc.dms) * K
                reloj.marca("v1_decod")
                # agrupación por burst (t_top) dentro de cada beam, contra lo ya aceptado
                acept = []
                todos = []
                nuevos_prev = {}
                for b in np.unique(hb):
                    ii = np.flatnonzero(hb == b)
                    ii = ii[np.argsort(-snr[ii])]
                    ya = list(previos.get(int(b), []))
                    nb = 0
                    for i in ii:
                        # se descarta solo si ya hay uno del mismo burst al menos igual de fuerte; si el nuevo
                        # es más fuerte (p. ej. el pico real tras un trozo débil de la diagonal en el chunk anterior), va igual
                        if any(abs(t_top[i] - tp) <= a.tol_t and sp >= snr[i] for tp, sp in ya):
                            continue
                        ya.append((t_top[i], snr[i]))
                        nuevos_prev.setdefault(int(b), []).append((t_top[i], snr[i]))
                        if snr[i] > a.umbral:
                            todos.append(i)
                        if snr[i] > a.umbral and nb < a.k:
                            acept.append(i)
                            nb += 1
                previos = nuevos_prev
                acept = np.array(sorted(acept, key=lambda i: -snr[i])[:a.k_total], dtype=int)
                if log_todos and todos:
                    log_todos.write(f"toy_sifter beamset={beam_set_id} fpga=[{ev0.chunk_fpga_start}:{ev0.chunk_fpga_end}] "
                                    f"n_todos={len(todos)}\n" + "\n".join(linea_sifter(evc, i) for i in todos) + "\n")
                    log_todos.flush()                      # si el grouper se mata con SIGKILL no se pierde lo escrito
            else:
                evc, acept, previos = None, np.zeros(0, int), {}

            reloj.marca("agrupar")
            if len(acept):
                ev1 = FrbSifterEvents(
                    beam_ids=np.asarray(evc.beam_ids)[acept], fpga_timestamps=np.asarray(evc.fpga_timestamps)[acept],
                    dms=np.asarray(evc.dms)[acept], snrs=snr[acept], rfi_probs=np.zeros(len(acept)),
                    widths_ms=np.asarray(evc.widths_ms)[acept],
                    subband_freqs_lo_MHz=np.asarray(evc.subband_freqs_lo_MHz)[acept],
                    subband_freqs_hi_MHz=np.asarray(evc.subband_freqs_hi_MHz)[acept],
                    tree_index=np.asarray(evc.tree_index)[acept],
                    chunk_fpga_start=ev0.chunk_fpga_start, chunk_fpga_end=ev0.chunk_fpga_end)
            else:
                ev1 = FrbSifterEvents(*(np.zeros(0) for _ in range(9)),
                                      chunk_fpga_start=ev0.chunk_fpga_start, chunk_fpga_end=ev0.chunk_fpga_end)
            if sifter is not None:
                sifter.send_events(beam_set_id=beam_set_id, events=ev1, coarsegrain_snr=v0_max)
            reloj.marca("enviar")

            if not a.sin_ventanas:
                guardar(pendientes, ichunk - 1)
            pendientes = None
            if len(acept) and not a.sin_ventanas:
                parches = []
                for i in acept:
                    tk = tokens[int(ht[i])]
                    ndm, nt = tk.shape[1], tk.shape[2]
                    p = cp.zeros((2 * PATCH, 2 * PATCH), dtype=cp.uint32)
                    a0, a1 = max(0, hd[i] - PATCH), min(ndm, hd[i] + PATCH)
                    b0, b1 = max(0, hi[i] - PATCH), min(nt, hi[i] + PATCH)
                    p[a0 - (hd[i] - PATCH):a1 - (hd[i] - PATCH), b0 - (hi[i] - PATCH):b1 - (hi[i] - PATCH)] = \
                        tk[hb[i], a0:a1, b0:b1]
                    parches.append(p)
                pendientes = {
                    "beam": np.asarray(evc.beam_ids)[acept], "fpga_timestamp": np.asarray(evc.fpga_timestamps)[acept],
                    "dm": np.asarray(evc.dms)[acept], "snr": snr[acept], "width_ms": np.asarray(evc.widths_ms)[acept],
                    "tree": ht[acept].astype(np.int32), "ibeam": hb[acept], "idm": hd[acept], "itime": hi[acept],
                    "ichunk": np.full(len(acept), ichunk, dtype=np.int64),
                    "estable": np.full(len(acept), ichunk >= grouper.full_steady_ichunk),
                    "tok": cp.asnumpy(cp.stack(parches)),
                }
            reloj.marca("parches")
            total = reloj.fin(ichunk, n_cand, len(acept))
            atomic_print(f"grouper_v1: ichunk={ichunk}, v0={len(ev0)}, v1={len(acept)}, "
                         f"t={1e3 * total:.0f} ms")
    finally:
        if anillo is not None and pendientes is not None:
            guardar(pendientes, ichunk)
        log_v0.close()
        if log_todos:
            log_todos.close()
        cola.put(None)
        hilo.join()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("grouper_addr")
    ap.add_argument("-s", "--sifter", default=None)
    ap.add_argument("-t", "--umbral", type=float, default=5.0, help="S/N mínimo de un evento enviado al sifter")
    ap.add_argument("--umbral-celda", type=float, default=5.0, help="S/N mínimo de las celdas que forman regiones")
    ap.add_argument("--tol-t", type=float, default=0.1, help="tolerancia en t_top (s) para unir candidatos de un mismo burst")
    ap.add_argument("--max-cand", type=int, default=32, help="candidatos por beam y chunk que se decodifican")
    ap.add_argument("--k", type=int, default=4, help="eventos por beam y chunk")
    ap.add_argument("--k-total", type=int, default=256, help="eventos por chunk")
    ap.add_argument("--salida", type=Path, required=True)
    ap.add_argument("--log-todos", type=Path, default=None,
                    help="escribe todos los eventos agrupados (sin topes --k/--k-total) en este archivo")
    ap.add_argument("--perfil", action="store_true", help="tiempo por etapa en tiempos_grouper.csv (sincroniza la GPU)")
    ap.add_argument("--sin-ventanas", action="store_true", help="no guarda ventanas (costo de operación)")
    a = ap.parse_args()

    from pirate_frb.rpc import FrbGrouper, FrbSifterClient
    from pirate_frb.utils import atomic_print
    sifter_cm = FrbSifterClient(a.sifter) if a.sifter else contextlib.nullcontext()
    with sifter_cm as sifter, FrbGrouper(a.grouper_addr, restore_cuda_device=False) as grouper:
        try:
            correr(grouper, sifter, a)
        except KeyboardInterrupt:
            atomic_print("grouper_v1: interrumpido")
        except RuntimeError as e:
            if grouper.is_stopped:
                atomic_print(f"grouper_v1: pirate se desconectó ({e}); saliendo")
            else:
                raise


if __name__ == "__main__":
    main()
