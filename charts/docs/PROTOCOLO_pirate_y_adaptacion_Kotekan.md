# Cómo mandar datos a pirate: protocolo de red y adaptación a Kotekan

Este documento explica qué tiene que mandar el X-engine (o cualquier emisor) para que pirate busque FRB, y cómo
llevar el emisor de referencia de este repositorio (`emisor/`, Python) a un stage de **Kotekan** (C++).

## 1. Qué recibe pirate

pirate recibe **intensidad ya formada en beams**: un cubo `(beam, canal de frecuencia, tiempo)`.

| | Prueba de 64 beams | Lo fija |
|---|---|---|
| Beams por nodo de búsqueda | 64 | `prueba.beams` |
| Canales | 21 504 (672 gruesos × 32), 299,85–501,75 MHz | `prueba.canales`, `prueba.banda_mhz` |
| Muestra de tiempo | 1 ms | `prueba.muestra_ms` |
| Bits por muestra | 4 (int4) + escala y offset en float16 | protocolo |
| Tasa por nodo de búsqueda | 64 × 21 504 × 4 bits × 1000/s = 5,5 Gbit/s + ~3 % de escalas | — |

El cálculo de beams, la canalización ×32 y la integración a 1 ms los hace el X-engine; pirate empieza en la
recuantización a 4 bits.

## 2. El protocolo (versión 2)

Referencia: `notes/network_protocol.md` de pirate. Implementación de referencia: `emisor/protocolo.py`.

**Una conexión TCP persistente por par emisor → receptor.** Cada emisor (nodo X-engine) manda **un subconjunto de
canales para todos los beams** del receptor. El receptor junta los canales de todos sus emisores.

Al abrir la conexión (todo little-endian):

| Campo | Tipo | Valor |
|---|---|---|
| magic | uint32 | `0xf4bf4b02` (el `02` es la versión) |
| flags | uint32 | `0x1` = FLAG_ACK (el receptor confirma cada minichunk; solo para pruebas) |
| largo | uint32 | bytes del texto que sigue, incluido ≥ 1 cero de relleno |
| metadatos | texto ASCII terminado en cero | YAML con el formato de `configs/xengine_metadata.yml` de pirate |

Metadatos obligatorios (ver `metadatos_yaml()` en `emisor/protocolo.py`):

| Campo | Qué es | ¿Igual en todos los emisores? |
|---|---|---|
| `version` | 2 | sí |
| `zone_nfreq`, `zone_freq_edges` | canales y bordes de banda en MHz (una o más zonas) | sí |
| `freq_channels` | **los canales que manda esta conexión** (índices, 0 = frecuencia más baja) | no |
| `beamset`, `beam_ids`, `beam_positions_x/y` | identificación de los beams del receptor | sí |
| `unix_ns_at_seq_0`, `dt_ns_per_seq`, `seq_per_frb_time_sample` | reloj: época, ns por cuenta de seq, cuentas por muestra de 1 ms | sí |
| `tel_*` | geometría del telescopio (pirate la exige; la búsqueda no la usa) | sí |
| `noise_variance` | varianza del ruido por zona (1,0 si los datos ya están normalizados) | sí |

Luego, **minichunks** de 256 muestras de tiempo, uno tras otro, sin relleno:

| Campo | Tipo | Tamaño |
|---|---|---|
| magic | uint32 | 4 bytes |
| seq | uint64 | 8 bytes: seq del inicio del minichunk |
| escalas y offsets | float16 `[nbeams, nfreq, 2]` | nbeams × nfreq × 4 bytes |
| datos | int4 `[nbeams, nfreq, 256]` | nbeams × nfreq × 128 bytes |

- `nfreq` = largo de `freq_channels` de esa conexión. El tiempo es el eje más rápido.
- Dos muestras por byte: `(x[1] << 4) | x[0]`. El valor −8 marca una muestra faltante (pirate la suma como 0).
- pirate reconstruye `dato = escala × q + offset` y supone **media 0 y varianza 1** por canal: la escala y el offset
  tienen que dejar los datos así (o varianza = `noise_variance`).
- Los seq de minichunks consecutivos difieren en `256 × seq_per_frb_time_sample`. El primero debe ser múltiplo de
  ese valor. Saltar seq = datos faltantes.
- Con FLAG_ACK, tras cada minichunk el receptor responde 1 byte: 1 = copiado al anillo, 0 = descartado.

## 3. Recuantización a 4 bits (lo más importante para la sensibilidad)

Medido en las pruebas (`docs/ANALISIS_pirate_cuantizacion_y_grupo3_21504_2026-10-09.md`, `docs/RESULTADOS_64beams.md`):

- **2 niveles por σ** (escala = σ/2), redondeo centrado y recorte a ±7 conserva ~99 % del S/N.
- **La escala y el offset tienen que ser por canal.** Con un valor fijo para todos los canales, los canales con
  RFI fuerte quedan pegados en +7 y suman un offset a todos los DM (todos los bloques daban S/N ≈ 118).
  `emisor/cuantizacion.py::por_canal`: mediana y σ = (p84 − p16)/2 por canal y bloque.
- **Canales malos como −8** (máscara de RFI conocida, canales saturados) **y corregir la escala de los útiles** por
  √(N/N_útiles). pirate calcula el S/N suponiendo varianza 1 en todos los canales: sin la corrección, todo el S/N
  sale escalado por √(1 − fracción enmascarada).

## 4. El emisor de referencia (`emisor/`)

```
fuentes.py       de dónde salen los datos (archivos .fil, ruido, relleno)   ← en Kotekan: el buffer de entrada
cuantizacion.py  8 bits → 4 bits + escala/offset por minichunk, máscara     ← en Kotekan: el cálculo del stage
protocolo.py     cabecera, metadatos YAML, minichunks, empaquetado int4     ← en Kotekan: el envío por socket
emisor.py        procesos de preparación + 1 hilo por conexión, ritmo       ← en Kotekan: main_thread()
```

Para probar otra fuente de datos sin tocar el resto: una clase con `bloque(t0, T, out)` que llene `out [T, nchan]`
uint8 (ver `fuentes.py`).

## 5. Adaptación a Kotekan

Kotekan organiza el procesamiento en **stages** (clases C++ que heredan de `kotekan::Stage`) conectados por
**buffers** de frames con metadatos, configurados en YAML.

### 5.1 Stages de referencia que ya existen

| Stage | Qué tomar de él |
|---|---|
| `lib/stages/bufferSend.{hpp,cpp}` | **El modelo principal:** conexión TCP persistente, reconexión (`reconnect_time`), `send()` con `MSG_NOSIGNAL` en bucle, timeout (`SO_SNDTIMEO`), descarte de frames si el socket va lento (`drop_frames`) |
| `lib/stages/frbNetworkProcess.{hpp,cpp}` | El envío de FRB de CHIME (UDP, varios destinos, pings de vida, endpoints REST para activar/desactivar destinos) |
| `lib/stages/UpchannelizationSchedule.*`, `chartsBasebandReadout.*` | Stages propios de CHARTS: cómo se ve el buffer de intensidad y sus metadatos en nuestra cadena |

### 5.2 Stage propuesto: `pirateSender`

```
[buffer de intensidad: beam × canal × tiempo, float/uint8]  ──►  pirateSender  ──TCP──►  receptores de pirate
```

Configuración YAML sugerida (mismas claves que `configs/prueba_64b.yml`):

```yaml
pirate_sender:
  kotekan_stage: pirateSender
  in_buf: frb_intensity_buffer
  receivers: ["10.0.0.2:5000"]      # uno por nodo de búsqueda (o varios receptores por nodo)
  freq_channels: [0, 5376]          # rango de canales que manda ESTE nodo X-engine
  beam_ids: [0, 63]
  zone_nfreq: [21504]
  zone_freq_edges: [299.8497764530551, 501.7502235469449]
  samples_per_minichunk: 256
  requantization: per_channel       # mediana y σ por canal (ver sección 3)
  mask_file: mascara_m1.bin         # canales a −8 (opcional)
  ack: false
  reconnect_time: 5
```

Pasos del `main_thread()`:

1. **Arranque:** construir el texto YAML de metadatos (igual que `metadatos_yaml()`), con `unix_ns_at_seq_0` =
   `time0_ns` de Kotekan, `dt_ns_per_seq` = largo de la FFT del F-engine / frecuencia de muestreo, y
   `seq_per_frb_time_sample` = factor de canalización × factor de integración (así lo propone pirate en
   `configs/xengine_metadata.yml`). Abrir una conexión por receptor y mandar la cabecera.
2. **Por frame del buffer:** `wait_for_full_frame()`; tomar el seq del frame de sus metadatos.
3. **Recuantizar** cada beam y canal a int4 con escala/offset por minichunk (sección 3). Es el paso de cómputo:
   en CPU con SIMD o en GPU si el buffer ya está ahí (en las pruebas, ~0,2 s por beam y bloque de 2 s en un hilo de
   numpy; en C++ vectorizado debería ser mucho menos).
4. **Empaquetar** con la disposición del protocolo: para cada minichunk, `[magic][seq][float16 nbeams×nfreq×2]`
   `[uint8 nbeams×nfreq×128]`. Conviene guardar el frame ya en el orden `[minichunk, beam, canal, 128 bytes]`
   (como `emisor.py`), para que cada envío sea contiguo.
5. **Enviar** con `send()` en bucle (como `bufferSend`); si se activa `ack`, leer 1 byte por minichunk.
6. `mark_frame_empty()`.

Recomendaciones:
- **Seq continuos:** si un frame se pierde, saltar su seq (pirate lo trata como faltante); no reenviar.
- **Reconexión:** al reconectar hay que repetir la cabecera; el primer seq debe ser múltiplo de `256 × seq_per_frb_time_sample`.
- **Varios receptores por nodo de búsqueda** si la tasa por receptor pasa de ~11–13 Gbit/s (medido: un receptor de
  pirate llegó a ~13,5 Gbit/s y dos a ~16 con 256 beams).
- **Validación:** el mismo cubo enviado con `emisor/emisor.py` y con el stage tiene que dar en pirate los mismos
  eventos (mismo DM, tiempo y S/N). Para comparar byte a byte, `protocolo.py` arma exactamente los mismos bytes.

### 5.3 Lo que falta definir con el equipo del X-engine

- Formato y metadatos del buffer de intensidad en Kotekan (dtype, orden de ejes, dónde va el seq).
- Si la recuantización a 4 bits se hace en el X-engine (este stage) o en un conversor en el nodo de búsqueda.
- Cómo se reparten canales entre nodos X-engine y beams entre nodos de búsqueda (`freq_channels`, `beam_ids`).
- La geometría real del telescopio (`tel_*`) y las posiciones de los beams.
