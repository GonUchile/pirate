"""Protocolo de red X-engine → pirate (versión 2), en Python puro.

Implementa lo que describe pirate en notes/network_protocol.md, sin depender de pirate:

  Por conexión TCP (una por par emisor/receptor), al abrirla:
    uint32  0xf4bf4b02                (magic; el 02 es la versión)
    uint32  flags                     (FLAG_ACK = 0x1: el receptor responde 1 byte por minichunk)
    uint32  largo del texto, incluidos ≥ 1 bytes de relleno en cero
    texto   metadatos YAML (formato configs/xengine_metadata.yml de pirate), terminado en cero

  Después, una secuencia de minichunks de 256 muestras de tiempo, sin relleno entre campos:
    uint32  0xf4bf4b02
    uint64  número de secuencia (seq) del inicio del minichunk
    float16 [nbeams, nfreq, 2]        escala y offset por beam y canal (dato = escala·q + offset)
    int4    [nbeams, nfreq, 256]      intensidad; −8 = muestra faltante; dos por byte: (x[1] << 4) | x[0]

  nfreq es el número de canales que manda ESTA conexión (los de `freq_channels` en sus metadatos).
  Todo es little-endian. Los seq de minichunks consecutivos difieren en 256 · seq_per_frb_time_sample,
  y el primero debe ser múltiplo de ese valor.

Si FLAG_ACK está activo, tras cada minichunk el receptor devuelve un byte: 1 si lo copió al anillo
de pirate y 0 si lo recibió pero lo descartó.
"""
import socket
import struct

import numpy as np

MAGIC = 0xF4BF4B02
FLAG_ACK = 0x1
MUESTRAS_MINICHUNK = 256

# Valores de relleno de la geometría del telescopio: pirate los exige y los compara entre emisores,
# pero la búsqueda no los usa (son los de XEngineMetadata::make_fiducial de pirate).
TELESCOPIO_FIDUCIAL = {
    "tel_origin_itrs_lat_deg": 49.32075144444,
    "tel_origin_itrs_lon_deg": -119.62081125,
    "tel_grid_x_axis": [0.999974342398359362, -0.000037539331442772, -0.007163318767675494],
    "tel_grid_y_axis": [0.000065403387739210, 0.999992433220348809, 0.003889630373557614],
    "tel_dish_elev_axis": [0.99999999838132391, -0.000056897733584327, 0.0],
    "tel_dish_vert_axis": [0.0, 0.0, 1.0],
    "tel_dish_coelev_deg": 0.0,
    "tel_dish_separation_x_m": 6.300156854906823,
    "tel_dish_separation_y_m": 8.500057809796308,
}


def _lista(v):
    return "[" + ", ".join(repr(float(x)) if isinstance(x, float) else str(int(x)) for x in v) + "]"


def metadatos_yaml(zone_nfreq, zone_freq_edges, beam_ids, freq_channels, unix_ns_at_seq_0, dt_ns_per_seq,
                   seq_per_frb_time_sample, beamset=0, beam_positions_x=None, beam_positions_y=None,
                   telescopio=None, noise_variance=1.0):
    """Texto YAML de metadatos de una conexión. Todos los emisores hacia un mismo receptor deben mandar lo mismo,
    salvo freq_channels (los canales de cada uno)."""
    nb = len(beam_ids)
    bx = beam_positions_x if beam_positions_x is not None else [0.01 * (i % 16) for i in range(nb)]
    by = beam_positions_y if beam_positions_y is not None else [0.01 * (i // 16) for i in range(nb)]
    tel = dict(TELESCOPIO_FIDUCIAL, **(telescopio or {}))
    lineas = [
        "version: 2",
        f"zone_nfreq: {_lista(zone_nfreq)}",
        f"zone_freq_edges: {_lista([float(x) for x in zone_freq_edges])}",
        f"freq_channels: {_lista(freq_channels)}",
        f"beamset: {int(beamset)}",
        f"beam_ids: {_lista(beam_ids)}",
        f"beam_positions_x: {_lista([float(x) for x in bx])}",
        f"beam_positions_y: {_lista([float(x) for x in by])}",
        f"unix_ns_at_seq_0: {int(unix_ns_at_seq_0)}",
        f"dt_ns_per_seq: {int(dt_ns_per_seq)}",
        f"seq_per_frb_time_sample: {int(seq_per_frb_time_sample)}",
    ]
    for k, v in tel.items():
        lineas.append(f"{k}: {_lista(v) if isinstance(v, (list, tuple)) else repr(float(v))}")
    lineas.append(f"noise_variance: {float(noise_variance)!r}")
    return "\n".join(lineas) + "\n"


def cabecera_conexion(texto_yaml, flags=0):
    """Bytes que se mandan al abrir la conexión: magic, flags, largo y el YAML con ceros al final."""
    b = texto_yaml.encode("ascii")
    relleno = 8 - (len(b) % 8)                      # al menos un cero; alinea a 8 bytes
    return struct.pack("<III", MAGIC, flags, len(b) + relleno) + b + b"\0" * relleno


def empaquetar_int4(codigos):
    """codigos: uint8 con valores 0..15 (int4 en complemento a 2) y último eje de largo par → dos por byte."""
    c = codigos.astype(np.uint8, copy=False)
    return (c[..., 0::2] & 0x0F) | ((c[..., 1::2] & 0x0F) << 4)


class Conexion:
    """Una conexión TCP hacia un receptor de pirate, que manda un subconjunto fijo de canales."""

    def __init__(self, direccion, texto_yaml, ack=False, timeout_s=60.0):
        host, puerto = direccion.rsplit(":", 1)
        self.direccion = direccion
        self.ack = ack
        self.sock = socket.create_connection((host, int(puerto)), timeout=timeout_s)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.sock.sendall(cabecera_conexion(texto_yaml, FLAG_ACK if ack else 0))
        self.enviados = self.ensamblados = self.descartados = 0

    def enviar_minichunk(self, seq, escalas_offsets, datos):
        """escalas_offsets: float16 [nbeams, nfreq, 2]; datos: uint8 empaquetado [nbeams, nfreq, 128].
        Devuelve el byte de ACK (1 ensamblado, 0 descartado) o None si no hay ACK."""
        so = np.ascontiguousarray(escalas_offsets, dtype=np.float16)
        d = np.ascontiguousarray(datos, dtype=np.uint8)
        self.sock.sendall(struct.pack("<IQ", MAGIC, int(seq)))
        self.sock.sendall(so.tobytes())
        self.sock.sendall(memoryview(d).cast("B"))
        self.enviados += 1
        if not self.ack:
            return None
        r = self.sock.recv(1)
        if not r:
            raise ConnectionError(f"{self.direccion}: el receptor cerró la conexión")
        if r[0]:
            self.ensamblados += 1
        else:
            self.descartados += 1
        return r[0]

    def cerrar(self):
        try:
            self.sock.close()
        except OSError:
            pass
