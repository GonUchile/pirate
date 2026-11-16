#!/usr/bin/env python3
"""Sifter con alertas: última etapa de la cadena (reemplaza al toy_sifter de pirate).

Es un servidor gRPC con el mismo servicio FrbSifter de pirate (`frb_sifter.proto`): el grouper le manda, una vez
por bloque de 2,048 s, los eventos de todos los beams. Hace dos cosas:

  1. Registra todos los eventos en la salida estándar con el mismo formato que el toy_sifter (el lanzador la
     guarda en sifter.log y la evaluación la lee igual que antes).
  2. Decide qué eventos son una ALERTA DE FRB y las escribe en <salida>/alertas.jsonl (una línea JSON por alerta)
     y en la salida estándar ("ALERTA FRB ..."). Opcionalmente ejecuta un comando por alerta (--accion), con el
     JSON en la entrada estándar: avisar, volcar datos, mandar un VOEvent, etc.

Regla de alerta (todo configurable):
  - S/N >= --umbral (7,6 por defecto: ~10 falsas alarmas por hora en el sistema de 64 beams, con la RFI del
    sitio y la máscara m1; ver docs/RESULTADOS_64beams.md)
  - --dm-min <= DM <= --dm-max (la RFI terrestre tiene DM ~ 0)
  - veto de RFI: si hay eventos en más de --max-beams beams con la misma llegada (± --ventana-s, a 501,6 MHz), se
    descarta (una señal del cielo cae en uno o pocos beams vecinos; la RFI entra en todos)
  - los eventos de la misma llegada (± --ventana-s) en bloques seguidos y en beams coincidentes forman un grupo; el
    grupo se emite cuando pasan --espera-bloques bloques sin que su S/N mejore, con su mejor evento (un pulso
    brillante aparece antes, con menos S/N, en DM más bajos). Si después mejora > 20 %, sale una "actualizacion"

Tiempos: la hora UTC sale de unix_ns_at_seq_0 del X-engine (lo manda el emisor). `utc_llegada` es la llegada a
501,6 MHz; `latencia_s` es lo que pasó desde que terminó de llegar el pulso (borde inferior de la banda) hasta
que se emitió la alerta.

Necesita el entorno de pirate (grpc y pirate_frb). Uso:
  python servidor/sifter/sifter_alertas.py 127.0.0.1:7500 --salida resultados/x [--umbral 7.6] [--accion CMD]
"""
import argparse
import json
import signal
import subprocess
import sys
import tempfile
import threading
import time
from concurrent import futures
from datetime import datetime, timezone
from pathlib import Path

import grpc
import yaml
from pirate_frb.rpc.grpc import frb_sifter_pb2, frb_sifter_pb2_grpc

F_TOP_MHZ = 501.6


def retardo_borde_a_top(dd_yaml):
    """Segundos por unidad de DM entre el borde inferior de la banda (donde pirate fecha el evento) y 501,6 MHz."""
    import pirate_frb as pf
    with tempfile.NamedTemporaryFile("w", suffix=".yml", delete=False) as f:
        f.write(dd_yaml)
    cfg = pf.DedispersionConfig.from_yaml(f.name)
    Path(f.name).unlink()
    f_lo, f_hi = cfg.zone_freq_edges[0], cfg.zone_freq_edges[-1]
    return (1e-3 / cfg.dm_per_unit_delay()) * (f_lo ** -2 - F_TOP_MHZ ** -2) / (f_lo ** -2 - f_hi ** -2)


def utc(ns):
    return datetime.fromtimestamp(ns / 1e9, tz=timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def imprimir(texto):
    sys.stdout.write(texto + "\n")
    sys.stdout.flush()


def ejecutar_accion(comando, alerta):
    """Ejecuta el comando con la alerta (JSON) en la entrada estándar; un error no detiene al sifter."""
    try:
        r = subprocess.run(comando, shell=True, input=json.dumps(alerta), text=True, capture_output=True, timeout=60)
        if r.returncode:
            imprimir(f"accion: rc={r.returncode} {r.stderr.strip()[:200]}")
    except Exception as e:                                # noqa: BLE001
        imprimir(f"accion: falló ({e})")


class Grupo:
    """Eventos con la misma llegada a 501,6 MHz (± ventana) en un beam o en beams coincidentes."""

    def __init__(self, ev, t, beams, bloque):
        self.mejor, self.t, self.beams, self.bloque_mejor = ev, t, set(beams), bloque
        self.emitido = None                           # id de la alerta, "veto", o None mientras está abierto
        self.snr_emitido = 0.0
        self.vetado = False
        self.n = 0


class SifterAlertas(frb_sifter_pb2_grpc.FrbSifterServicer):
    def __init__(self, a):
        self.a = a
        self.lock = threading.Lock()
        self.k_top, self.unix0, self.dt_ns = 0.0, 0, 5000
        self.grupos = []
        self.n_alertas = self.n_vetadas = 0
        a.salida.mkdir(parents=True, exist_ok=True)
        self.archivo = open(a.salida / "alertas.jsonl", "w")

    def CheckConfiguration(self, request, context):
        if request.protocol_version != frb_sifter_pb2.PROTOCOL_VERSION_CURRENT:
            msg = f"versión de protocolo {request.protocol_version}, se requiere {frb_sifter_pb2.PROTOCOL_VERSION_CURRENT}"
            imprimir(f"CheckConfiguration RECHAZADA ({msg})")
            context.abort(grpc.StatusCode.FAILED_PRECONDITION, msg)
        with self.lock:
            if request.pirate_yaml:
                self.k_top = retardo_borde_a_top(request.pirate_yaml)
            x = yaml.safe_load(request.xengine_yaml) if request.xengine_yaml else {}
            self.unix0 = int(x.get("unix_ns_at_seq_0", 0))
            self.dt_ns = int(x.get("dt_ns_per_seq", 5000))
        imprimir(f"CheckConfiguration(search_ip_addr={request.search_ip_addr!r}) unix_ns_at_seq_0={self.unix0} "
                 f"dt_ns_per_seq={self.dt_ns} retardo_a_top={self.k_top:.5f} s/DM")
        return frb_sifter_pb2.ConfigReply(ok=True)

    def FrbEvents(self, request, context):
        cg = request.coarsegrain_snr
        lineas = [f"toy_sifter beamset={request.beam_set_id} fpga=[{request.chunk_fpga_start}:{request.chunk_fpga_end}] "
                  f"coarse_snr_max={max(cg) if len(cg) else 0.0:.4g}"]
        for ev in request.events:
            lineas.append(f"    beam_id={ev.beam_id}, dm={ev.dm:.4g}, fpga_timestamp={ev.fpga_timestamp}, "
                          f"width={ev.width_ms:.4g} ms, subband=[{ev.subband_freq_lo_MHz:.1f},{ev.subband_freq_hi_MHz:.1f}] MHz, "
                          f"snr={ev.snr:.4g}, tree_index={ev.tree_index}")
        imprimir("\n".join(lineas))
        with self.lock:
            self.decidir(list(request.events), request.chunk_fpga_start, request.chunk_fpga_end)
        return frb_sifter_pb2.FrbEventsReply(ok=True, message="")

    def decidir(self, eventos, c0, c1):
        """Agrega los eventos del bloque [c0, c1) a sus grupos y emite los grupos cuyo S/N dejó de crecer.

        Un pulso brillante aparece primero en DM más bajos (se completan antes) con S/N creciente, y el máximo llega
        con el DM correcto algunos bloques después. Por eso un grupo queda abierto hasta que pasan --espera-bloques
        bloques sin que su S/N mejore, y la alerta lleva el mejor evento."""
        a = self.a
        s_seq = self.dt_ns * 1e-9
        t_top = [ev.fpga_timestamp * s_seq - ev.dm * self.k_top for ev in eventos]    # s desde seq 0, a 501,6 MHz
        cand = sorted((i for i, ev in enumerate(eventos)
                       if ev.snr >= a.umbral and ev.dm >= a.dm_min and (a.dm_max is None or ev.dm <= a.dm_max)),
                      key=lambda i: -eventos[i].snr)
        for i in cand:
            ev, t = eventos[i], t_top[i]
            beams = {eventos[j].beam_id for j in range(len(eventos)) if abs(t_top[j] - t) <= a.ventana_s}
            g = next((g for g in self.grupos if abs(t - g.t) <= a.ventana_s and ev.beam_id in g.beams), None)
            if g is None:
                g = Grupo(ev, t, beams, c1)
                self.grupos.append(g)
            g.beams |= beams
            g.n += 1
            if len(beams) > a.max_beams:
                g.vetado = True
            if ev.snr > g.mejor.snr:
                g.mejor, g.t, g.bloque_mejor = ev, t, c1
                if g.emitido not in (None, "veto") and ev.snr > 1.2 * g.snr_emitido:
                    self.emitir(g, "actualizacion")
        largo = max(c1 - c0, 1)
        for g in self.grupos:
            if g.emitido is None and c1 - g.bloque_mejor >= a.espera_bloques * largo:
                self.cerrar(g)
        t_ahora = c1 * s_seq
        self.grupos = [g for g in self.grupos if g.emitido is None or t_ahora - g.t < a.memoria_s]

    def cerrar(self, g):
        if g.vetado or len(g.beams) > self.a.max_beams:
            g.emitido = "veto"
            self.n_vetadas += 1
            imprimir(f"VETO RFI beam={g.mejor.beam_id} DM={g.mejor.dm:.1f} S/N={g.mejor.snr:.2f}: "
                     f"{len(g.beams)} beams con la misma llegada")
        else:
            self.emitir(g, "alerta")

    def cerrar_todos(self):
        with self.lock:
            for g in self.grupos:
                if g.emitido is None:
                    self.cerrar(g)

    def emitir(self, g, tipo):
        a, ev, ahora = self.a, g.mejor, time.time_ns()
        ns_top = self.unix0 + int(round(g.t * 1e9))
        ns_fin = self.unix0 + ev.fpga_timestamp * self.dt_ns     # llegada al borde inferior: el pulso ya llegó entero
        if tipo == "alerta":
            self.n_alertas += 1
            g.emitido = (f"CHARTS-{datetime.fromtimestamp(ns_top / 1e9, tz=timezone.utc):%Y%m%dT%H%M%S.%f}"[:-3]
                         + f"-b{ev.beam_id}")
        g.snr_emitido = float(ev.snr)
        al = {"id": g.emitido, "tipo": tipo, "utc_llegada": utc(ns_top), "f_ref_mhz": F_TOP_MHZ, "beam": int(ev.beam_id),
              "dm": round(float(ev.dm), 2), "snr": round(float(ev.snr), 2), "ancho_ms": float(ev.width_ms),
              "subbanda_mhz": [round(ev.subband_freq_lo_MHz, 2), round(ev.subband_freq_hi_MHz, 2)],
              "beams_coincidentes": sorted(int(b) for b in g.beams), "eventos_en_grupo": g.n,
              "t_stream_s": round(g.t, 4), "fpga_timestamp": int(ev.fpga_timestamp),
              "utc_emitida": utc(ahora), "latencia_s": round((ahora - ns_fin) / 1e9, 3), "n": self.n_alertas}
        self.archivo.write(json.dumps(al) + "\n")
        self.archivo.flush()
        imprimir(f"{'ALERTA FRB' if tipo == 'alerta' else 'ACTUALIZACIÓN'} {al['id']} beam={al['beam']} DM={al['dm']} "
                 f"S/N={al['snr']} ancho={al['ancho_ms']} ms llegada={al['utc_llegada']} latencia={al['latencia_s']} s")
        if a.accion:
            hilo = threading.Thread(target=ejecutar_accion, args=(a.accion, al), daemon=True)
            hilo.start()
            if a.accion_espera:
                hilo.join()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("direccion", help="ip:puerto donde escucha (el grouper se conecta aquí)")
    ap.add_argument("--salida", type=Path, required=True, help="carpeta donde se escribe alertas.jsonl")
    ap.add_argument("--umbral", type=float, default=7.6, help="S/N mínimo de una alerta")
    ap.add_argument("--dm-min", type=float, default=20.0)
    ap.add_argument("--dm-max", type=float, default=None)
    ap.add_argument("--max-beams", type=int, default=8, help="más beams que esto con la misma llegada = RFI")
    ap.add_argument("--ventana-s", type=float, default=0.5, help="tolerancia de llegada (s) para agrupar y vetar")
    ap.add_argument("--espera-bloques", type=int, default=1, help="bloques sin mejora del S/N antes de emitir la alerta")
    ap.add_argument("--memoria-s", type=float, default=60.0, help="segundos que se recuerda un grupo emitido")
    ap.add_argument("--accion", default=None, help="comando que se ejecuta por alerta, con el JSON en la entrada estándar")
    ap.add_argument("--accion-espera", action="store_true", help="esperar a que termine el comando (por defecto no)")
    a = ap.parse_args()

    signal.signal(signal.SIGTERM, lambda *x: (_ for _ in ()).throw(KeyboardInterrupt()))
    servicio = SifterAlertas(a)
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1), options=[("grpc.so_reuseport", 0)])
    frb_sifter_pb2_grpc.add_FrbSifterServicer_to_server(servicio, server)
    if server.add_insecure_port(a.direccion) == 0:
        sys.exit(f"ERROR: no se pudo escuchar en {a.direccion} (¿ocupado?)")
    server.start()
    imprimir(f"sifter_alertas: esperando al grouper en {a.direccion}; umbral S/N {a.umbral}, DM {a.dm_min}–{a.dm_max}, "
             f"veto > {a.max_beams} beams; alertas en {a.salida / 'alertas.jsonl'}")
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        pass
    finally:
        servicio.cerrar_todos()
        imprimir(f"sifter_alertas: {servicio.n_alertas} alertas, {servicio.n_vetadas} vetadas por RFI")
        server.stop(grace=1.0).wait()


if __name__ == "__main__":
    main()
