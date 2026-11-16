#!/usr/bin/env python3
"""Lanzador de la prueba: arma las configs de pirate desde UN archivo YAML y maneja los procesos.

Subcomandos:
  servidor  En el PC de búsqueda: sifter con alertas + grouper v1 + servidor de pirate. Queda escuchando en los
            receptores del YAML hasta Ctrl-C, `--duracion` o que exista el archivo <salida>/DETENER.
  emisor    Corre emisor/emisor.py con el mismo YAML (puede ser en otro PC: solo necesita numpy y PyYAML).
  local     Las dos cosas en esta máquina, una detrás de la otra, y al final evalúa la corrida.

Uso:
  python servidor/lanzador.py local    --config configs/prueba_64b.yml --salida resultados/corta --limite 20
  python servidor/lanzador.py servidor --config configs/prueba_64b.yml --salida resultados/remota
  python servidor/lanzador.py emisor   --config configs/prueba_64b.yml --salida resultados/remota --receptores 10.0.0.2:5000
Las opciones que no son del lanzador (--limite, --rfi, --mascara, --ritmo, ...) pasan al emisor.

En <salida>: config_dedispersion.yml y config_servidor.yml (generados), server.log, grouper.log, sifter.log,
alertas.jsonl (alertas de FRB, también se muestran en la consola), eventos_todos.log, tiempos_grouper.csv, gpu.csv, vmstat.log y lo que escribe el emisor.
"""
import argparse
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import yaml

RAIZ = Path(__file__).resolve().parent.parent


# ------------------------------------------------------------------------------------------- configs de pirate
def configs_pirate(cfg, salida):
    """Escribe las configs de dedispersión y del servidor de pirate a partir del YAML de la prueba."""
    p, red, s = cfg["prueba"], cfg["red"], cfg.get("servidor", {})
    d = dict(s.get("dedispersion", {}))
    dd = {
        "zone_freq_edges": [float(x) for x in p["banda_mhz"]],
        "zone_nfreq": [int(p["canales"])],
        "time_sample_ms": float(p.get("muestra_ms", 1.0)),
        "beams_per_gpu": int(p["beams"]),
        "dtype": "float16",
        "time_samples_per_chunk": int(p.get("bloque_muestras", 2048)),
        "frequency_subband_counts": d.get("frequency_subband_counts", [0, 0, 0, 0, 1]),
        "max_gpu_clag": int(d.get("max_gpu_clag", 5)),
        "beams_per_batch": int(d.get("beams_per_batch", 2)),
        "num_active_batches": int(d.get("num_active_batches", 2)),
        "toplevel_tree_rank": int(d.get("rank", 15)),
        "primary_trees": d.get("arboles", [{"num_early_triggers": 0, "max_width": 32,
                                            "wt_dm_downsampling": 64, "wt_time_downsampling": 64}]),
        "future_write_max_samples": int(d.get("future_write_max_samples", 5000)),
    }
    sv = {
        "num_servers": 1,
        "server_cpus": s.get("cpus", [0]),
        "host_memory_per_server": f"{s.get('memoria_ram_gib', 150)} GiB",
        "gpu_memory_per_server": f"{s.get('memoria_gpu_gib', 17)} GiB",
        "use_hugepages": bool(s.get("hugepages", False)),
        "ringbuf_nchunks": int(s.get("anillo_bloques", 64)),
        "data_ip_addrs": [list(red["receptores"])],
        "rpc_ip_addrs": [red.get("rpc", "127.0.0.1:6000")],
        "grouper_ip_addrs": [red.get("grouper", "127.0.0.1:7000")],
        "min_data_mtu": 1500, "min_rpc_mtu": 1500,
        "ssd_dirs": [s.get("ssd_dir", "/dev/shm/pirate_ssd")],
        "ssd_threads_per_server": 1,
        "nfs_dir": str(Path(s.get("volcados_dir", salida / "volcados")).resolve()),
        "nfs_threads_per_server": 1,
        "check_mountpoints": [],
    }
    (salida / "config_dedispersion.yml").write_text(yaml.safe_dump(dd, sort_keys=False, default_flow_style=None))
    (salida / "config_servidor.yml").write_text(yaml.safe_dump(sv, sort_keys=False, default_flow_style=None))
    Path(sv["nfs_dir"]).mkdir(parents=True, exist_ok=True)
    return salida / "config_dedispersion.yml", salida / "config_servidor.yml"


def puerto_libre(direccion):
    host, puerto = direccion.rsplit(":", 1)
    with socket.socket() as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind((host, int(puerto)))
            return True
        except OSError:
            return False


# ------------------------------------------------------------------------------------------------- procesos
class Procesos:
    def __init__(self, salida):
        self.salida, self.lista, self.monitores = salida, [], []

    def lanzar(self, nombre, cmd):
        log = open(self.salida / f"{nombre}.log", "w")
        p = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
        self.lista.append((nombre, p))
        self.registro(f"{nombre} pid {p.pid}: {' '.join(map(str, cmd))}")
        return p

    def monitor(self, cmd, archivo):
        if shutil.which(cmd[0]):
            self.monitores.append(subprocess.Popen(cmd, stdout=open(self.salida / archivo, "w"), stderr=subprocess.DEVNULL))

    def registro(self, msg):
        line = f"{time.strftime('%F %T')} {msg}"
        print(line, flush=True)
        with open(self.salida / "lanzador.log", "a") as f:
            f.write(line + "\n")

    def apagar(self):
        self.registro("apagando")
        for _, p in reversed(self.lista):
            if p.poll() is None:
                try:
                    os.killpg(p.pid, signal.SIGINT)
                except ProcessLookupError:
                    pass
        fin = time.time() + 60
        for _, p in self.lista:
            try:
                p.wait(timeout=max(0.1, fin - time.time()))
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGKILL)
        for m in self.monitores:
            m.terminate()


def esperar_texto(archivo, texto, timeout_s, proceso):
    fin = time.time() + timeout_s
    while time.time() < fin:
        if archivo.exists() and texto in archivo.read_text(errors="replace"):
            return True
        if proceso.poll() is not None:
            return False
        time.sleep(1)
    return False


def entorno_pirate(cfg):
    """Busca pirate_frb: $CHARTS_PIRATE_ENV, entornos.pirate del YAML, el entorno de este Python o el PATH."""
    candidatos = [os.environ.get("CHARTS_PIRATE_ENV"), (cfg.get("entornos") or {}).get("pirate")]
    for c in candidatos:
        if c:
            c = Path(os.path.expandvars(os.path.expanduser(str(c))))
            if (c / "bin" / "pirate_frb").exists():
                return str(c / "bin" / "python"), str(c / "bin" / "pirate_frb")
            raise SystemExit(f"ERROR: no está {c}/bin/pirate_frb (revisar CHARTS_PIRATE_ENV o entornos.pirate)")
    aqui = Path(sys.executable).parent / "pirate_frb"
    if aqui.exists():
        return sys.executable, str(aqui)
    enpath = shutil.which("pirate_frb")
    if enpath:
        return str(Path(enpath).parent / "python"), enpath
    raise SystemExit("ERROR: no encuentro pirate_frb. Activar su entorno (conda activate ...) o export CHARTS_PIRATE_ENV=/ruta/al/entorno. "
                     "Instalación: INSTALL.md")


def mostrar_alertas(archivo, pr):
    """Muestra en la consola (y en lanzador.log) cada alerta que escribe el sifter."""
    while not archivo.exists():
        time.sleep(1)
    with open(archivo) as f:
        while True:
            linea = f.readline()
            if not linea:
                time.sleep(0.5)
                continue
            try:
                x = json.loads(linea)
                pr.registro(f"*** ALERTA FRB {x['id']}: beam {x['beam']}, DM {x['dm']}, S/N {x['snr']}, "
                            f"llegada {x['utc_llegada']}, latencia {x['latencia_s']} s")
            except (ValueError, KeyError):
                pass


def arrancar_servidor(cfg, salida, pr):
    red, s = cfg["red"], cfg.get("servidor", {})
    py, pirate = entorno_pirate(cfg)
    pr.registro(f"pirate: {pirate}")
    ocupados = [d for d in list(red["receptores"]) + [red.get("rpc", "127.0.0.1:6000"), red.get("grouper", "127.0.0.1:7000"),
                                                     red.get("sifter", "127.0.0.1:7500")] if not puerto_libre(d)]
    if ocupados:
        raise SystemExit(f"ERROR: puertos ocupados {ocupados}: ¿otra corrida activa?")
    dd, sv = configs_pirate(cfg, salida)
    pr.monitor(["nvidia-smi", "--query-gpu=timestamp,utilization.gpu,memory.used,power.draw", "--format=csv", "-l", "5"], "gpu.csv")
    pr.monitor(["vmstat", "-t", "5"], "vmstat.log")
    al = s.get("alertas") or {}
    args_s = [py, str(RAIZ / "servidor" / "sifter" / "sifter_alertas.py"), red.get("sifter", "127.0.0.1:7500"),
              "--salida", str(salida), "--umbral", str(al.get("umbral", 7.6)), "--dm-min", str(al.get("dm_min", 20)),
              "--max-beams", str(al.get("max_beams", 8)), "--ventana-s", str(al.get("ventana_s", 0.25))]
    args_s += ["--dm-max", str(al["dm_max"])] if al.get("dm_max") is not None else []
    args_s += ["--accion", str(al["accion"])] if al.get("accion") else []
    (salida / "alertas.jsonl").unlink(missing_ok=True)
    pr.lanzar("sifter", args_s)
    threading.Thread(target=mostrar_alertas, args=(salida / "alertas.jsonl", pr), daemon=True).start()
    time.sleep(3)
    g = s.get("grouper", {})
    args_g = [py, str(RAIZ / "servidor" / "grouper" / "grouper_v1.py"), red.get("grouper", "127.0.0.1:7000"),
              "-s", red.get("sifter", "127.0.0.1:7500"), "-t", str(g.get("umbral", 6)), "--salida", str(salida / "ventanas")]
    args_g += ["--sin-ventanas"] if not g.get("ventanas", False) else []
    args_g += ["--log-todos", str(salida / "eventos_todos.log")] if g.get("registro_completo", True) else []
    args_g += ["--perfil"] if g.get("perfil", False) else []
    args_g += [str(x) for x in g.get("extra", [])]
    pr.lanzar("grouper", args_g)
    time.sleep(3)
    srv = pr.lanzar("server", [pirate, "run", "server", str(sv), str(dd)])
    if not esperar_texto(salida / "server.log", "server(s) started", 900, srv):
        raise SystemExit("ERROR: el servidor de pirate no arrancó (ver server.log)")
    pr.registro(f"servidor listo; receptores {', '.join(red['receptores'])}")


def cmd_emisor(cfg_path, salida, extra):
    return [sys.executable, str(RAIZ / "emisor" / "emisor.py"), "--config", str(cfg_path), "--salida", str(salida)] + extra


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("modo", choices=["servidor", "emisor", "local"])
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--salida", required=True, type=Path)
    ap.add_argument("--duracion", type=float, help="segundos que el servidor queda escuchando (modo servidor)")
    ap.add_argument("--sin-evaluar", action="store_true")
    a, extra = ap.parse_known_args()
    cfg = yaml.safe_load(open(a.config))
    a.salida = a.salida.resolve()
    a.salida.mkdir(parents=True, exist_ok=True)
    shutil.copy(a.config, a.salida / "prueba.yml")

    if a.modo == "emisor":
        sys.exit(subprocess.call(cmd_emisor(a.config, a.salida, extra)))

    if a.modo == "local":
        sd = os.path.expandvars(str((cfg.get("emisor") or {}).get("set_dir") or "$CHARTS_SET_DIR"))
        if "--set-dir" not in extra and ("$" in sd or not Path(sd).is_dir()):
            raise SystemExit(f"ERROR: no está la carpeta de datos {sd!r}: export CHARTS_SET_DIR=/carpeta/del/grupo3")
    pr = Procesos(a.salida)
    signal.signal(signal.SIGTERM, lambda *x: (_ for _ in ()).throw(KeyboardInterrupt()))
    try:
        arrancar_servidor(cfg, a.salida, pr)
        if a.modo == "servidor":
            fin = time.time() + a.duracion if a.duracion else None
            while (fin is None or time.time() < fin) and not (a.salida / "DETENER").exists():
                if any(p.poll() is not None for _, p in pr.lista):
                    pr.registro("un proceso terminó solo; ver los logs"); break
                time.sleep(2)
        else:
            rc = subprocess.call(cmd_emisor(a.config, a.salida, extra))
            pr.registro(f"emisor terminó (rc={rc}); esperando 30 s a que pirate procese la cola")
            time.sleep(30)
    except KeyboardInterrupt:
        pr.registro("interrumpido")
    finally:
        pr.apagar()
    if a.modo == "local" and not a.sin_evaluar:
        py, _ = entorno_pirate(cfg)
        subprocess.call([py, str(RAIZ / "evaluacion" / "evaluar_prueba_64b.py"), str(a.salida)])
        pr.registro(f"evaluación en {a.salida / 'resultados.md'}")
        subprocess.call([py, str(RAIZ / "evaluacion" / "verificar_alertas.py"), str(a.salida)])


if __name__ == "__main__":
    main()
