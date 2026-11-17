# pirate: estructura de trabajo, nomenclatura y salidas de datos

Referencia para CHARTS: 21 504 canales, 1 ms por muestra, 300–501,6 MHz, 64 beams, datos de 4 bits.
Verificado contra el código de pirate instalado en `sources/pirate` (2026-10-23).
Figuras generadas con `figuras_estructura_pirate.py` (en esta misma carpeta).
Versión navegable con los mismos dibujos: https://claude.ai/artifact/UyukWK3oJgRc8uKz1hRPaC

---

## 1. Nomenclatura

Los nombres son los del autor de pirate. "Bloque" es solo la traducción de "chunk" que usamos en español.

| Término | Qué es | Tamaño en CHARTS | Dónde está definido |
|---|---|---|---|
| muestra | Un instante de tiempo | 1 ms | — |
| **minichunk** | 256 muestras. Así envía los datos cada x-node, y cada uno trae su escala y offset de 4 bits por canal | 256 ms | `notes/developer.md:43` |
| **chunk** (bloque) | Rango de 2048 muestras de **todos los beams**. Es la unidad de proceso: pirate dedispersa y entrega el mapa una vez por chunk | 2,048 s · 64 frames · ~1,45 GB | `notes/developer.md:42` |
| **batch** (lote) | Grupo de beams que la GPU procesa a la vez dentro de un chunk | `beams_per_batch` | `notes/developer.md:44` |
| **frame** | Un par **(chunk, beam)**: un chunk de un solo beam. Es la unidad de almacenamiento | ~22,7 MB (datos crudos) | `notes/developer.md:45`, `include/pirate/MegaRingbuf.hpp:48` |
| **frame_ringbuf** | Anillo de **datos crudos** (4 bits) en RAM del servidor | ~20 chunks (40 s) × 64 beams ≈ 28 GB | `FrbServer::frame_ringbuf` |
| **MegaRingbuf** | Anillo de **sumas parciales** de la dedispersión, repartido en zonas de GPU y de RAM | depende de `max_gpu_clag` | `include/pirate/MegaRingbuf.hpp:18` |
| **segment** | Trozo mínimo del MegaRingbuf (una línea de caché de la GPU) | 128 bytes | `notes/developer.md:46` |
| **clag** (chunk lag) | Retardo de un dato del MegaRingbuf, en chunks: cuántos chunks hacia atrás guarda cada fila del árbol | 0 a 16 en el árbol 0 | `MegaRingbuf.hpp:55-57` |
| **flag** (frame lag) | El mismo retardo, contado en frames (`flag = clag × num_batches`) | — | `MegaRingbuf.hpp:57` |
| **zones** | Tramos del MegaRingbuf con distinto largo, cada uno en GPU o en RAM | — | `MegaRingbuf.hpp:59` |
| **max_gpu_clag** | El clag más grande que queda en la VRAM; lo demás va a RAM | 5 en la prueba de 64 beams | `include/pirate/DedispersionConfig.hpp:101-103` |

**Lo más fácil de confundir:**
- **Chunk vs. frame:** un chunk son los frames de todos los beams en el mismo intervalo de 2,048 s. Con 64 beams, 1 chunk = 64 frames. pirate **procesa** de a un chunk y **guarda** de a un frame.
- **Clag:** no es un dato sino un **número que tiene cada fila del árbol** (aproximadamente, cada DM). Las sumas parciales guardadas son la historia, en el MegaRingbuf.
- **Dos anillos con contenido distinto:**
  - frame_ringbuf: datos crudos, para volcados y para clasificadores sobre datos crudos;
  - MegaRingbuf: sumas a medio terminar, internas de la dedispersión.

  pirate usa la palabra "frame" en los dos, siempre con el sentido de (chunk, beam).

---

## 2. Figuras

### Dentro de un frame
![Un frame: 21 504 canales × 2048 muestras de 4 bits en 8 minichunks, más una escala y un offset por canal y minichunk](fig1_frame.png)

- `data`: int4, forma (21 504, 2048), 22,0 MB. El valor −8 marca una muestra faltante.
- `scales_offsets`: float16, forma (21 504, 8, 2), 0,69 MB.
- Metadatos: `beam_id`, `time_chunk_index` y los metadatos del X-engine.
- `intensidad = escala × valor_4bits + offset`, con la escala y el offset de su canal y su minichunk. En la GPU lo hace `GpuDequantizationKernel`.
- Definición: `include/pirate/AssembledFrame.hpp`.

### De frame a chunk
![Un chunk son los 64 frames de un mismo intervalo de 2,048 s](fig2_frame_chunk.png)

### La grilla beams × chunks
![Grilla de beams por chunks: un frame es una casilla, un chunk una columna, un batch un grupo de beams de una columna, y el anillo de frames la ventana de las últimas columnas](fig3_grilla.png)

### Dónde vive cada cosa
![Anillo de frames en RAM, MegaRingbuf repartido entre RAM y VRAM, árboles de dedispersión y mapa DM-tiempo hacia el grouper](fig4_almacenes.png)

### El clag
![Cada fila del árbol guarda clag chunks hacia atrás; las de clag hasta 5 en VRAM y el resto en RAM](fig5_clag.png)

Con DM ≈ 1105 las frecuencias bajas llegan ~33 s después que las altas, por eso esa fila guarda 16 chunks.

| `max_gpu_clag` | Dónde queda la historia | Costo por beam (árbol 0) |
|---|---|---|
| 16 | Toda en la GPU | 0,56 GB de VRAM, sin tráfico de historia por PCIe |
| 5 (la prueba de 64 beams) | 38 % en la GPU, 62 % en RAM | 0,22 GB de VRAM + 30 MB/s de PCIe |
| 1 | Casi toda en RAM | 0,045 GB de VRAM + 56 MB/s de PCIe |

### La cadena completa
![X-engines, servidor pirate, grouper, sifter y trigger: qué existe y qué falta](fig6_cadena.png)

---

## 3. Cómo salen los datos de pirate

pirate trabaja en un bucle: por cada **chunk**, por cada **batch** de beams y por cada **árbol** de dedispersión, entrega un resultado al grouper.

### 3.1 El mapa DM-tiempo (pirate → grouper, en la GPU)

Por cada árbol, dos arreglos de forma **(beams_per_batch, coarse_ndm, coarse_ntime)**:
- `out_max`: S/N máximo en cada celda, sobre las posiciones DM-tiempo de la celda y sobre los anchos;
- `out_argmax`: un token que codifica la posición exacta dentro de la celda y el ancho de ese máximo.

`FrbGrouper.create_events()` lo convierte en DM, tiempo, ancho y sub-banda.

Con la configuración de 3 árboles (`configs/charts_rt_r15t3.yml`):

| Árbol | Rango de DM | DMs probados | Mapa por chunk (DM × tiempo) | Celda en DM | Celda en tiempo | Anchos |
|---|---|---|---|---|---|---|
| 0 (1 ms, rank 15) | 0 – 1107 | 32 768 | 2048 × 128 | 0,54 | 16 ms | 16 |
| 1 (2 ms, rank 14) | 1107 – 2213 | 16 384 | 1024 × 64 | 1,08 | 32 ms | 13 |
| 2 (4 ms, rank 14) | 2213 – 4427 | 16 384 | 1024 × 32 | 2,16 | 64 ms | 10 |

En total pirate prueba 65 536 DMs: los árboles extra son de rank 14 y solo cubren la mitad superior de su rango.

**Condiciones para usar el mapa** (`notes/grouper_interface.md`):
- Comparte memoria de GPU con pirate (memoria compartida por IPC); los metadatos van por gRPC por loopback.
- Solo es válido **dentro del context manager de cada batch** (`grouper.get_output(ichunk, ibatch)`). Después pirate reusa esa memoria. Para guardar chunks anteriores hay que copiarlos (~67 MB por chunk para el árbol 0 con 64 beams).
- El grouper tiene que terminar cada chunk en menos de 2,048 s, o frena a pirate.
- No hay que tocar la GPU fuera del context manager de `FrbGrouper`.

### 3.2 La lista de candidatos (grouper → sifter, por gRPC)

Una vez por chunk, un `FrbEventsMessage` (`grpc/frb_sifter.proto`):

| Campo | Contenido |
|---|---|
| `chunk_fpga_start`, `chunk_fpga_end` | Rango de tiempo del chunk |
| `coarsegrain_snr` | S/N máximo de cada beam en ese chunk (todos los beams) |
| `events[]` | Lista de `FrbEvent` |

Cada `FrbEvent`:

| Campo | Contenido |
|---|---|
| `beam_id` | Beam |
| `fpga_timestamp` | Llegada al **borde inferior de la banda**; puede caer fuera del chunk |
| `dm`, `snr`, `width_ms` | Decodificados del token: posición exacta, no la de la celda |
| `subband_freq_lo_MHz`, `subband_freq_hi_MHz` | Banda donde se detectó (early triggers) |
| `tree_index` | Árbol de origen |
| `rfi_prob` | **Reservado para el clasificador**; hoy siempre vale 0 |

Antes de los eventos, el grouper manda al sifter un `ConfigMessage` con tres YAML: la configuración de pirate, los metadatos del X-engine y el plan de dedispersión.

### 3.3 Los datos crudos (frame_ringbuf → archivos, por pedido)

- No se leen directamente: se piden con el RPC **`WriteFiles`** (`grpc/frb_search.proto`), indicando beams y un rango de `fpga_seq`.
- pirate escribe **un archivo por frame** en `{nfs_dir}/{acqdir}/frame_b(BEAM)_t(CHUNK).asdf`, con los datos de 4 bits, las escalas y offsets y los metadatos.
- El rango puede incluir chunks **futuros**: se escriben a medida que llegan, y `SubscribeFiles` avisa cuando cada archivo está listo.
- Si el rango ya salió del anillo, esos archivos simplemente no aparecen.
- Clientes en Python: `pirate_frb/rpc/FrbSearchClient.py` y `FileSubscriber.py`.
- **Costo:** un burst a DM ~1000 necesita ~17 chunks (~35 s) por beam, unos 370 MB escritos antes de poder clasificarlo.

---

## 4. Qué existe y qué hay que construir

| Pieza | Estado | Lenguaje | Archivo |
|---|---|---|---|
| Recepción, frames, anillo de frames | existe | C++ | `src_lib/`, `include/pirate/Receiver.hpp`, `AssembledFrame.hpp` |
| Dedispersión en árbol + peak-finding por celdas | existe | C++/CUDA (kernels generados desde Python) | `include/pirate/Dedisperser.hpp`, `pirate_frb/cuda_generator/` |
| MegaRingbuf (reparto GPU/RAM) | existe | C++/CUDA | `include/pirate/MegaRingbuf.hpp` |
| Interfaz para leer el mapa y decodificar eventos | existe | Python sobre C++ | `pirate_frb/rpc/FrbGrouper.py` |
| Volcado de frames (`WriteFiles`, `SubscribeFiles`) | existe; nadie lo llama | gRPC + Python | `pirate_frb/rpc/FrbSearchClient.py` |
| Grouper | **solo de juguete**: máximo por beam y chunk, sin agrupar ni clasificar | Python/cupy | `pirate_frb/run_toy_grouper.py` |
| Sifter | **solo de juguete**: imprime los eventos | Python | `pirate_frb/run_toy_sifter.py` |
| Clasificador | **no existe** (solo el campo `rfi_prob`) | — | — |
| Coincidencia entre beams, decisión, trigger a los x-nodes | **no existe** | — | — |

---

## 5. Dónde encaja un clasificador

| Etapa | Entrada | Dónde corre | Restricción |
|---|---|---|---|
| 1 · sobre el mapa | Ventana de `out_max` (+ `out_argmax`) de cada árbol alrededor del evento | Dentro del grouper, en la GPU, cada chunk | < 2,048 s por chunk para todos los beams; sin redimensionar el mapa (recorte a resolución nativa) |
| 2 · sobre datos crudos | Frames de 4 bits pedidos con `WriteFiles` → imágenes frecuencia-tiempo y DM-tiempo | Después del grouper o en el sifter, solo para los sobrevivientes | La lectura del anillo cuesta ~370 MB por candidato a DM ~1000 |

Presupuesto de tiempo real:
- buffer de los x-nodes: ~40 s;
- latencia de pirate a DM ~1000: ~33 s;
- quedan **~5 s** para grouper + clasificador + trigger.

Ver `docs/PLAN_testbench_clasificador_ML_FETCH_en_pirate_2026-10-22.md`.

---

## Archivos relacionados

- `docs/GUIDE_pirate_installation_usage_benchmark_CHARTS_2026-10-08.md`: instalación, uso y benchmark.
- `docs/RESUMEN_pirate_CHARTS_paso_a_paso_2026-10-11.md`: memoria por beam, capacidad y prueba de 64 beams.
- `sources/pirate/notes/grouper_interface.md`: interfaz pirate ↔ grouper (del autor).
- `sources/pirate/notes/developer.md`: definiciones de chunk, batch, frame y segment (del autor).

## Anexo · Grouper v1 del testbench, paso a paso

`grouper/grouper_v1.py`, explicado con figuras en `grouper_v1_paso_a_paso.html` (página: https://claude.ai/artifact/KyMfyPwgJxUrBrmAzBJirm). Las figuras se generan con `figuras_grouper_v1.py`.

![Entrada: mapa out_max](fig_g1_mapa.png)
![Paso 1: regiones conexas sobre S/N 5](fig_g2_regiones.png)
![Paso 2: máximo por región](fig_g3_maximos.png)
![Paso 3: agrupación por t_top](fig_g4_ttop.png)
![Paso 3b: caso entre chunks](fig_g6_entre_chunks.png)
![Paso 4: salida al sifter, v1 frente a v0](fig_g5_salida.png)
