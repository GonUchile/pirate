#!/usr/bin/env python
"""Grouper del testbench, versión 0: el toy grouper de pirate + guardar ventanas del mapa DM-tiempo.

Comportamiento hacia el sifter: IDÉNTICO a pirate_frb/run_toy_grouper.py (un evento por chunk y por
beam con el máximo S/N sobre árboles, DM y tiempo, si supera el umbral). Así la evaluación y las
etiquetas de la prueba de 64 beams (evaluar_prueba_64b.py, ml/01_catalogo_candidatos.py) siguen valiendo.

Lo nuevo: por cada evento con S/N >= --umbral-guardar se guarda, un chunk más tarde, la ventana del mapa
del árbol del evento en resolución nativa (sin redimensionar):
  - out_max: DM [idm-128, idm+128) x tiempo [chunk anterior, chunk del evento, chunk siguiente]
    -> (256, 3*nt), float16; fuera del mapa se rellena con 0 (y se anota el rango válido);
  - out_argmax: parche 32 x 32 de tokens centrado en el pico (chunk del evento), uint32.
Se guarda un .npz por chunk con todos los eventos de ese chunk, en un hilo aparte (no frena a pirate).

Copias en la GPU (para tener los chunks vecinos): 3 chunks de out_max de todos los beams por árbol
(árbol 0, 64 beams: 3 x 64 MB) y los tokens del chunk actual (64 MB).

Uso (entorno de pirate), igual que toy_grouper:
  python grouper_testbench.py 127.0.0.1:7000 -s 127.0.0.1:7500 -t 5 --salida DIR [--umbral-guardar 5]
"""
import argparse
import contextlib
import itertools
import json
import queue
import threading
import time
from pathlib import Path

import numpy as np

VDM, PATCH = 128, 16          # media ventana en DM; medio parche de tokens


def escritor(cola, salida):
    """Hilo que escribe los .npz (uno por chunk con eventos)."""
    while True:
        item = cola.get()
        if item is None:
            return
        nombre, datos = item
        np.savez(salida / nombre, **datos)


def correr(grouper, sifter, umbral, umbral_guardar, salida):
    import cupy as cp
    from pirate_frb.utils import atomic_print

    nbeams = grouper.total_beams
    nbatches = grouper.nbatches
    bpb = nbeams // nbatches
    beam_set_id = grouper.xengine_metadata.beamset
    if sifter is not None:
        sifter.send_configuration(
            pirate_yaml=grouper.dedispersion_config_yaml_string,
            xengine_yaml=grouper.xengine_metadata_yaml_string,
            dedispersion_plan_yaml=grouper.dedispersion_plan_yaml_string,
            grouper_yaml={"toy_grouper": True, "grouper_testbench": "v0"},
            search_ip_addr=grouper.search_ip_addr)

    salida.mkdir(parents=True, exist_ok=True)
    cola = queue.Queue(maxsize=64)
    hilo = threading.Thread(target=escritor, args=(cola, salida), daemon=True)
    hilo.start()

    anillo = None            # por árbol: (3, nbeams, ndm, nt) float32
    tokens = None            # por árbol: (nbeams, ndm, nt) uint32, chunk actual
    fpga_inicio = {}         # ichunk -> chunk_fpga_start
    pendientes = None        # eventos del chunk anterior, esperando el chunk siguiente
    meta_info = {"nbeams": nbeams, "nbatches": nbatches, "VDM": VDM, "PATCH": PATCH,
                 "full_steady_ichunk": int(grouper.full_steady_ichunk)}

    def guardar(pend, ichunk_ev):
        """Arma las ventanas de los eventos del chunk ichunk_ev (usa los chunks ichunk_ev-1, ichunk_ev, ichunk_ev+1)."""
        if pend is None or not len(pend["snr"]):
            return
        ventanas, validos = [], []
        for k in range(len(pend["snr"])):
            it, b, idm = int(pend["tree"][k]), int(pend["ibeam"][k]), int(pend["idm"][k])
            ring = anillo[it]
            ndm, nt = ring.shape[2], ring.shape[3]
            orden = [(ichunk_ev + d) % 3 for d in (-1, 0, 1)]
            disponibles = [ichunk_ev - 1 >= 0, True, (ichunk_ev + 1) in fpga_inicio]
            lo, hi = max(0, idm - VDM), min(ndm, idm + VDM)
            v = cp.zeros((2 * VDM, 3 * nt), dtype=cp.float16)
            for j, (slot, ok) in enumerate(zip(orden, disponibles)):
                if ok:
                    v[lo - (idm - VDM): hi - (idm - VDM), j * nt:(j + 1) * nt] = ring[slot, b, lo:hi, :].astype(cp.float16)
            ventanas.append(v)
            validos.append((lo - (idm - VDM), hi - (idm - VDM), int(disponibles[0]), int(disponibles[2])))
        datos = {k: v for k, v in pend.items() if k != "tok"}
        datos["ventana"] = cp.asnumpy(cp.stack(ventanas))
        datos["tokens"] = pend["tok"]
        datos["valido"] = np.array(validos, dtype=np.int32)    # fila inicio, fila fin, hay chunk previo, hay chunk siguiente
        datos["chunk_fpga_start"] = np.int64(fpga_inicio[ichunk_ev])
        datos["meta"] = np.array(json.dumps(meta_info))
        cola.put((f"chunk_{ichunk_ev:06d}.npz", datos))

    ichunk = -1
    try:
        for ichunk in itertools.count():
            t0 = time.time()
            per_max = cp.full((nbeams,), -cp.inf, dtype=cp.float32)
            per_tree = cp.full((nbeams,), -1, dtype=cp.int64)
            per_idm = cp.full((nbeams,), -1, dtype=cp.int64)
            per_it = cp.full((nbeams,), -1, dtype=cp.int64)
            per_tok = cp.zeros((nbeams,), dtype=cp.uint32)

            for ibatch in range(nbatches):
                beam0 = ibatch * bpb
                with grouper.get_output(ichunk, ibatch) as outputs:
                    if anillo is None:
                        anillo = [cp.zeros((3, nbeams) + t.shape[1:], dtype=cp.float32) for t in outputs.out_max]
                        tokens = [cp.zeros((nbeams,) + t.shape[1:], dtype=cp.uint32) for t in outputs.out_argmax]
                        meta_info["formas"] = [list(t.shape[1:]) for t in outputs.out_max]
                    for itree, tree_out in enumerate(outputs.out_max):
                        sl = slice(beam0, beam0 + bpb)
                        # copias para las ventanas (dentro del context manager: después la memoria se reusa)
                        anillo[itree][ichunk % 3, sl] = tree_out
                        tokens[itree][sl] = outputs.out_argmax[itree]
                        # lo mismo que el toy grouper
                        b, ndm, nt = tree_out.shape
                        flat = tree_out.reshape(b, ndm * nt)
                        bmax, barg = flat.max(axis=1), flat.argmax(axis=1)
                        btok = outputs.out_argmax[itree].reshape(b, ndm * nt)[cp.arange(b), barg]
                        upd = bmax > per_max[sl]
                        per_max[sl] = cp.where(upd, bmax, per_max[sl])
                        per_tree[sl] = cp.where(upd, itree, per_tree[sl])
                        per_idm[sl] = cp.where(upd, barg // nt, per_idm[sl])
                        per_it[sl] = cp.where(upd, barg % nt, per_it[sl])
                        per_tok[sl] = cp.where(upd, btok, per_tok[sl])

            ibeam = cp.nonzero(per_max > umbral)[0]
            events = grouper.create_events(ichunk, per_tree[ibeam], ibeam, per_idm[ibeam], per_it[ibeam],
                                           per_max[ibeam], per_tok[ibeam])
            fpga_inicio[ichunk] = int(events.chunk_fpga_start)
            for viejo in [c for c in fpga_inicio if c < ichunk - 3]:
                del fpga_inicio[viejo]
            if sifter is not None:
                sifter.send_events(beam_set_id=beam_set_id, events=events, coarsegrain_snr=per_max)

            # eventos del chunk anterior: ya está el chunk siguiente en el anillo
            guardar(pendientes, ichunk - 1)

            # eventos de este chunk que se guardarán con el próximo chunk
            sel = cp.nonzero(per_max > umbral_guardar)[0]
            pendientes = None
            if len(sel):
                ev = grouper.create_events(ichunk, per_tree[sel], sel, per_idm[sel], per_it[sel],
                                           per_max[sel], per_tok[sel])
                tr, bm, idm, itm = (cp.asnumpy(a[sel]) for a in (per_tree, cp.arange(nbeams), per_idm, per_it))
                parches = []
                for k in range(len(sel)):
                    tk = tokens[int(tr[k])]
                    ndm, nt = tk.shape[1], tk.shape[2]
                    p = cp.zeros((2 * PATCH, 2 * PATCH), dtype=cp.uint32)
                    a0, a1 = max(0, idm[k] - PATCH), min(ndm, idm[k] + PATCH)
                    b0, b1 = max(0, itm[k] - PATCH), min(nt, itm[k] + PATCH)
                    p[a0 - (idm[k] - PATCH):a1 - (idm[k] - PATCH), b0 - (itm[k] - PATCH):b1 - (itm[k] - PATCH)] = tk[bm[k], a0:a1, b0:b1]
                    parches.append(p)
                pendientes = {
                    "beam": ev.beam_ids, "fpga_timestamp": ev.fpga_timestamps, "dm": ev.dms, "snr": ev.snrs,
                    "width_ms": ev.widths_ms, "tree": tr.astype(np.int32), "ibeam": bm, "idm": idm, "itime": itm,
                    "ichunk": np.full(len(sel), ichunk, dtype=np.int64),
                    "estable": np.full(len(sel), ichunk >= grouper.full_steady_ichunk),
                    "tok": cp.asnumpy(cp.stack(parches)),
                }
            atomic_print(f"grouper_testbench: ichunk={ichunk}, nevents={len(events)}, "
                         f"guardados={0 if pendientes is None else len(pendientes['snr'])}, "
                         f"coarse_snr_max={float(per_max.max()):.4g}, t={1e3 * (time.time() - t0):.0f} ms")
    finally:
        # al desconectarse pirate: guardar los eventos del último chunk (sin chunk siguiente) y vaciar la cola
        if anillo is not None and pendientes is not None:
            guardar(pendientes, ichunk)
        cola.put(None)
        hilo.join()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("grouper_addr")
    ap.add_argument("-s", "--sifter", default=None)
    ap.add_argument("-t", "--umbral", type=float, default=10.0)
    ap.add_argument("--umbral-guardar", type=float, default=5.0)
    ap.add_argument("--salida", type=Path, required=True)
    a = ap.parse_args()

    from pirate_frb.rpc import FrbGrouper, FrbSifterClient
    from pirate_frb.utils import atomic_print
    sifter_cm = FrbSifterClient(a.sifter) if a.sifter else contextlib.nullcontext()
    with sifter_cm as sifter, FrbGrouper(a.grouper_addr, restore_cuda_device=False) as grouper:
        try:
            correr(grouper, sifter, a.umbral, a.umbral_guardar, a.salida)
        except KeyboardInterrupt:
            atomic_print("grouper_testbench: interrumpido")
        except RuntimeError as e:
            if grouper.is_stopped:
                atomic_print(f"grouper_testbench: pirate se desconectó ({e}); saliendo")
            else:
                raise


if __name__ == "__main__":
    main()
