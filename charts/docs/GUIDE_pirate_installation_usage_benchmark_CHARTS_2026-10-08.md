# pirate (pirate_frb): instalación, uso y primer benchmark para CHARTS

**Fecha:** 2026-10-08 · **Máquina:** RTX 4090 (sm_89), CUDA 13.1, gcc 13.3, 314 GiB RAM
**Docs upstream:** https://kmsmith137.github.io/pirate/ (notas TeX: `dedispersion.pdf`, `detrending.pdf`, `variance_map.pdf`, `inpainting.pdf`)

pirate es la búsqueda FRB en tiempo real en GPU que Kendrick Smith está escribiendo para CHORD.
Es un dedispersor en **árbol** (tipo bonsai, pero en GPU) en C++/CUDA con interfaz Python (`pirate_frb`).
Incluye un servidor que recibe datos del X-engine por TCP, un ring buffer, escritura de datos por RPC (gRPC) y peak-finding.

---

## 1. Qué quedó instalado y dónde

| Elemento | Ruta |
|---|---|
| Entorno conda (Python 3.13) | `<entorno de pirate>` |
| Fuente ksgpu (dependencia) | `<fuentes>/ksgpu` (pip editable) |
| Fuente pirate | `<fuentes>/pirate` (pip editable, v1.5.0) |
| Configs y logs CHARTS | `<carpeta de trabajo>` |

Activar el entorno:

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate <entorno de pirate>
pirate_frb --help
```

**Verificación:** `pirate_frb test -n 1` pasa completo (~1 min), sin fallos.

### Problemas encontrados en la instalación (para reinstalar)

1. **`pip` apunta a un pip de Python 2.7** (`~/.local/bin/pip` gana en el PATH). Dentro del entorno hay que usar siempre `python -m pip ...`.
2. **Falta `clang-format`.** El generador de kernels lo necesita, pero el `environment_*.yml` no lo declara. Se instaló con `conda install -c conda-forge clang-format`.
3. **Carrera en `make -j`.** `asdf-cxx/include/asdf/config.hxx` no se genera a tiempo. Basta con correr `make asdf-cxx/include/asdf/config.hxx` antes de `make -j 32`.
4. **Arquitectura.** Se compiló solo para sm_89 con un `config.mk` (gitignored) en cada repo: `NVCC_ARCH = -gencode arch=compute_89,code=sm_89`. pirate tarda ~2 min con 32 hilos.

Para actualizar: `git pull && make -j 32` en `ksgpu` y luego en `pirate`, con el entorno activo. Por ser instalación editable, no hace falta repetir el pip.

---

## 2. Conceptos clave

- **Dedispersión en árbol:** tiene costo ~O(N_f · N_t · log N_DM), no O(N_f · N_t · N_DM) como la fuerza bruta (AstroAccelerate, dedisp). Por eso el throughput es mucho mayor. A cambio, la cuadrícula de DM y los retardos son aproximados; el detalle está en `dedispersion.pdf`.
- **`toplevel_tree_rank` (R):** fija el número de DMs, 2^R, y el retardo máximo, 2^R muestras. Con 1 ms y 300–501,6 MHz:
  - R = 15 → DM 0–1107
  - R = 16 → DM 0–2213
- **`primary_trees`:** son árboles adicionales que cubren DMs más altos. Cada uno se aplica sobre datos re-muestreados ×2 en tiempo.
- **`frequency_subband_counts`:** cada árbol busca además en subbandas, lo que da sensibilidad a FRBs de banda angosta. Se calculan con `pirate_frb dev make_subbands FMIN FMAX 0.1 -r 4`.
- **Peak-finding:** se hace en el mismo kernel (boxcars hasta `max_width`). La salida es una matriz gruesa de (DM, tiempo) con el **S/N máximo** (`out_max`) y un token `out_argmax` que codifica DM fino, tiempo, ancho y subbanda.
  - El S/N se normaliza con pesos analíticos a partir de las varianzas por canal.
- **Formato de entrada:** **int4 con escala/offset float16** por (beam, canal, bloque de 256 muestras). El valor −8 significa muestra enmascarada.
  - Llega por TCP (protocolo en `notes/network_protocol.md`) o, en modo offline, desde archivos ASDF (`frame_b{BEAM}_t{CHUNK}.asdf`).
- **Kernels pre-generados:** solo existen kernels para combinaciones fijas de (dtype, dd_rank, max_width, subbandas, early triggers). Están listadas en `makefile_helper.py`, bajo `production_kernels`.
  - Una config nueva con otras subbandas requiere agregar filas ahí y recompilar.
  - En cambio, número de canales, banda, `time_sample_ms` y número de beams se pueden cambiar sin recompilar.

---

## 3. Comandos principales

```bash
# Inspeccionar una config: plan de dedispersión, rango de DM por árbol y memoria
pirate_frb show dedisperser -c -t CONFIG.yml

# Benchmark de la GPU: subida H2G de int4, decuantización, dedispersión y peak-finding
pirate_frb time_dedisperser -n 30 -b NBEAMS -H -g CLAG CONFIG.yml
#   -b  número de beams que se asignan de verdad en memoria
#   -H  sin hugepages (necesario aquí: no hay hugepages reservadas)
#   -g  max_gpu_clag: menos ring buffer en VRAM y más en RAM

# Tests unitarios
pirate_frb test -n 1

# Búsqueda "toy" completa en loopback, 4 terminales en este orden
# (comandos exactos en notes/quick_start.md):
pirate_frb run toy_sifter 127.0.0.1:7500
pirate_frb run toy_grouper ...
pirate_frb run server configs/frb_server/toy.yml configs/dedispersion/toy.yml
pirate_frb run fake_xengine -f -g 30 ...      # inyecta FRBs simulados cada 30 s
pirate_frb rpc status | start_stream | show_streams | rand_write ...

# Dedispersión offline de datos guardados (1 beam; peak-finding provisorio: S/N máx por chunk)
pirate_frb run offline_dedisperser ACQDIR CONFIG.yml
```

Desde Python, `pirate_frb.OfflineDedisperser(config)` procesa un `AssembledFrame` por chunk. Dentro del `with od.dedisperse(frame) as out:` se lee `out.out_max[itree]`, con forma (beams, ndm_grueso, nt_grueso).

---

## 4. Primer benchmark con parámetros CHARTS

**Config:** `projects/pirate_charts/configs/charts_sb0_r15.yml`

- 300–501,6 MHz, **21 504 canales**, 1 ms, fp16
- Rank 15: un árbol, **DM 0–1107**, boxcar ≤ 32 ms
- Sin subbandas: reutiliza los kernels ya compilados (estilo `chord_sb0`)
- Logs en `projects/pirate_charts/time_r15_*.log`

"Capacidad" es el número de beams que la GPU procesaría en tiempo real al ritmo medido (`real-time beams`). Si los beams asignados son menos que la capacidad, la configuración entra en tiempo real.

| Beams asignados | clag | VRAM | RAM pinned | Capacidad (beams t.r.) | Carga | ¿Tiempo real? |
|---:|---:|---:|---:|---:|---:|:---:|
| 16 | 5 | — | — | 765 | 2 % | ✅ |
| 64 | 5 | 15,6 GB | 21,5 GB | 761 | 8 % | ✅ |
| 96 | 2 | 9,5 GB | 45 GB | 481 | 20 % | ✅ |
| 128 | 1 | 7,6 GB | 65 GB | 396 | 32 % | ✅ |
| 192 | 2 | 17,2 GB | 91 GB | 481 | 40 % | ✅ |
| **256** | **1** | **13,4 GB** | **130 GB** | **405** | **63 %** | ✅ |
| 320 | 1 | 16,2 GB | 162 GB | 392 | 82 % | ✅ (poco margen) |
| 384 | 0 | 10,0 GB | 204 GB | 335 | 115 % | ❌ |

Lectura:

- **El cómputo no es el límite.** Si todo cabe en VRAM, la GPU da para ~760 beams.
- **El límite es la memoria.** El ring buffer de la dedispersión necesita ~0,24 GB de VRAM más ~0,34 GB de RAM por beam con clag 5.
  - Al bajar `max_gpu_clag`, el ring buffer pasa a RAM y el cuello de botella pasa a ser el PCIe: H2G ≈ 22,5 GB/s, prácticamente saturado en PCIe 4.0 x16.
- **Techo práctico:** ~250–300 beams de 21 504 canales por 4090 (`-g 1`, ~130–160 GB de RAM), con un margen de 20–40 %.
  - Como referencia, la estimación para s-engine/AstroAccelerate en el plan del 2026-09-25 era de ~70–100 beams.
- Con clag 2, la capacidad sube a ~480 beams, pero la VRAM alcanza solo para ~200 beams.

**Ojo: esto mide solo la parte GPU.** No incluye:

- recepción por red ni armado de frames (CPU);
- **enmascarado de RFI**, que todavía no está implementado en pirate (es la próxima prioridad del autor);
- grouper y sifter, que hoy son versiones "toy";
- conversión de 8 bits a int4.

---

## 5. Diferencias con CHARTS y trabajo necesario

1. **Cuantización a int4.** CHARTS entrega 8 bits y pirate solo acepta int4 con escala/offset cada 256 muestras.
   - Hay que re-cuantizar y medir la pérdida de S/N con el set de comparación (`outputs_comparacion`). Se espera que sea pequeña, pero no está medida.
2. **Sin RFI.** Con datos reales, el S/N no es confiable hasta que exista el enmascarado.
   - Mientras tanto, se pueden marcar muestras como −8 (masked) con el mismo masking que usamos en los otros pipelines.
3. **Subbandas CHARTS.** `make_subbands 300 501.6 0.1 -r 4` da counts `(0,1,6,3,1)`, combinación que no tiene kernels compilados.
   - Para usarla hay que agregar 4 filas `('fp16', 8|7, W, 1, (0,1,6,3,1), et)` en `makefile_helper.py` y recompilar. Puede costar más throughput.
4. **Ingreso de datos.**
   - **Offline**, para evaluar Recall y FP con nuestros `.fil`: escribir un conversor `.fil → frames ASDF` o alimentar `OfflineDedisperser` desde Python.
     - El peak-finding offline es provisorio. Hay que extraer candidatos (DM, t, ancho, S/N) desde `out_max`/`out_argmax` y agruparlos nosotros.
   - **Tiempo real:** implementar un emisor del protocolo X→FRB v2 (TCP, minichunks de 256 muestras) desde kotekan/CHARTS. El servidor se configura solo con la metadata que recibe.
5. **API inestable.** El proyecto es "experimental" y está en desarrollo activo (1.5.0), así que formatos y protocolo pueden cambiar.

### Siguientes pasos sugeridos

1. Correr la búsqueda toy (4 procesos, FRBs inyectados) para ver la cadena completa con candidatos. Todavía no se ha corrido.
2. Escribir un conversor `.fil → ASDF` y pasar el set de comparación por `offline_dedisperser`. Primero medir el Recall frente a los otros pipelines, en el formato estándar Recall + FP.
3. Agregar los kernels de subbandas para CHARTS y repetir el benchmark.
4. Medir el costo de CPU de un servidor real con N beams usando `fake_xengine` con la metadata de CHARTS (más o menos 256 beams).

---

## 6. Búsqueda offline sobre .fil y comparación con las otras pipelines (2026-10-08, segunda parte)

### Código

- `projects/pirate_charts/pirate_fil_search.py`: busca FRBs con pirate en un `.fil` de 8 bits.
  - **No pasa por int4 ni por ASDF.** Normaliza por canal y escribe float16 directo en el buffer de entrada del `GpuDedisperser` (`get_input()`).
  - Decodifica `out_argmax` con `plan.decode_argmax/decode_argmax2` para obtener DM, t, ancho y subbanda.
  - Agrupa: máximos locales 3×3 + supresión de no-máximos (0,1 s y 10 % en DM).
  - `--int4 N` simula la re-cuantización del protocolo de red, con N niveles por σ.
  - El orden de frecuencias está verificado: el canal 0 de pirate es la frecuencia **más baja**, así que el `.fil` (foff < 0) se invierte.
- `projects/pirate_charts/configs/charts_search.yml`: rank 16 con dos árboles. Cubre DM 0–~2200 con boxcar ≤ 32 muestras y DM ~2200–4400 con boxcar de 32 ms. Sin subbandas.
- `bench/pipelines/pirate.py`: pirate como quinta pipeline de `run_benchmark.py --pipelines pirate`.
- `resultados_pipelines_clasicas/comparar_pirate.py`: compara las 5 pipelines con el formato Recall + FP.
- `projects/pirate_charts/calibrate_snr.py`: inyecta pulsos boxcar de S/N conocido en ruido gaussiano.

### Calibración con pulsos inyectados (S/N verdadero 20, sin smearing intracanal)

| Canales | W = 1 ms | W = 4 ms | W = 16 ms |
|---|---|---|---|
| 21 504 | 0,63–0,72 | 0,86–0,91 | 0,92–1,09 |
| 672 | 0,32–0,42 (no detecta a DM ≥ 1000) | 0,31–0,79 | 0,60–0,90 |

- pirate supone canales angostos frente al salto de retardo entre canales, como en CHORD o CHARTS con 21 504 canales. Con 672 canales y pulsos angostos pierde S/N.
- Error de DM < 0,1 % y error de tiempo ≤ 1,4 ms con 21 504 canales.
- **int4 frente a float16** (misma realización de ruido): pérdida de 0–1 % con 2 niveles/σ y ~2 % con 3 niveles/σ. Es despreciable.

### Comparación: 300 archivos de `outputs_comparacion` (672 canales, 100 de A + 100 de B + 100 de NEG), boxcar ≤ 32 ms

Las cuatro pipelines clásicas salen de la campaña bc32 `20260925_042706`, sobre los mismos archivos. Salida en `resultados_pipelines_clasicas/pirate_prueba_20261008/comparacion.txt`.

Con S/N ≥ 5:

| | Recall A | Recall B | FP ruido/h | FP no-FRB (% NEG) |
|---|---|---|---|---|
| Heimdall | 87 | 58 | 444 | 45 |
| AA | 100 | 85 | 979 | 95 |
| CASM-Hella | 95 | 63 | 865 | 66 |
| PRESTO | 99 | 66 | 3233 | 63 |
| **pirate** | **100** | **79** | 2999 | 88 |

A igual tasa de FP por ruido, **10 eventos/h**:

| | umbral | Recall A | Recall B | FP no-FRB (% NEG) |
|---|---|---|---|---|
| Heimdall | 5,80 | 73 | 54 | 32 |
| AA | 6,14 | 98 | 63 | 56 |
| CASM-Hella | 6,80 | 49 | 43 | 27 |
| PRESTO | 6,12 | 77 | 54 | 35 |
| **pirate** | 6,16 | **100** | **68** | 67 |

- A igual tasa de FP por ruido, pirate tiene el mejor Recall en A y en B.
- Su FP no-FRB es alto, igual que el de AA: también detecta los negativos, que son señales dispersadas con un índice de DM no físico.
- Con 100 archivos por set, cada porcentaje tiene un error de ±4–5 %.
- Tiempo de búsqueda: **0,76 s por archivo de 119 s** (672 canales, 1 beam), contando solo la búsqueda.

---

## 7. Prueba de tiempo real con ring buffer y máximo de beams (2026-10-08, tercera parte)

### Cadena probada (todo en esta máquina, por loopback)

`fake_xengine` (TCP, int4) → `FrbServer` (recepción → ring buffer de frames en RAM → H2G → dedispersión + peak-finding en la GPU) → `toy_grouper` (umbral 10) → `toy_sifter`.

- El fake X-engine se mantiene a lo más 4 chunks por delante del servidor. Por eso la tasa `beams=X/N` que imprime el servidor es la capacidad real de la cadena completa.
- El ring buffer de frames ocupa toda la RAM que sobra de `host_memory_per_server` después de la dedispersión.

Archivos en `projects/pirate_charts/`:
- `correr_tiempo_real.sh ETIQUETA CONFIG MINUTOS [opciones fake_xengine]` (con `SERVER_CFG=...` para elegir la config del servidor).
- `configs/server_charts*.yml` y `configs/charts_rt_r15t3_b*.yml`.
- `tiempo_real/summarize.sh` y `tiempo_real/cruzar_sifter.py`, para resumir corridas y cruzar inyectados con detectados.

### Config de dedispersión elegida: `charts_rt_r15t3`

Rank 15 con 3 árboles, sin subbandas:

| Árbol | Resolución | DM |
|---|---|---|
| 0 | 1 ms | 0–1107 |
| 1 | 2 ms | 1107–2213 |
| 2 | 4 ms | 2213–4427 |

Boxcar máximo de 32 ms en los tres árboles.

Rank 16 (un solo árbol a 1 ms hasta DM ~2200) cuesta ~3,7 GB de RAM por beam y alcanza para solo ~64 beams. Rank 15 con 3 árboles cuesta ~1,3 GB por beam.

`time_dedisperser` (solo GPU, 21 504 canales):

| Beams | clag | VRAM | RAM | Capacidad (beams en tiempo real) |
|---:|---:|---:|---:|---:|
| 128 | 1 | 11,3 GB | 167 GB | 260 |
| 160 | 2 | 19,6 GB | 202 GB | 298 |
| 192 | 1 | 15,7 GB | 250 GB | 264 |
| 128 | 3 | 22,2 GB | 156 GB | 341 |

### Resultados de la cadena completa

| Corrida | Beams | Receptores | Datos | Ring buffer de frames | Capacidad (beams equiv.) | Tiempo real | GPU | CPU ociosa |
|---|---:|---:|---|---|---:|---|---:|---:|
| b128_junk | 128 | 1 | ceros | ~31 chunks = ~64 s | 151,5 | 1,18× | 52 % | 87 % |
| b160_2rx_junk | 160 | 2 | ceros | ~12 chunks = ~25 s | 222 | **1,39×** | 89 % | 81 % |
| b160_2rx_frb | 160 | 2 | gaussiano + FRBs S/N 20 | ~25 s | 200 | **1,25×** | 87 % | 43 % |

- **Con un receptor**, la entrada TCP limita a ~13,5 Gbps. Con dos receptores (`data_ip_addrs: [[':5000', ':5001']]`) llega a ~19,7 Gbps y el límite pasa a ser la GPU.
- **FRBs inyectados con 160 beams:** se detectaron 1557 de 1557 evaluables (100 %) en los tres árboles y en todos los anchos. S/N medido: mediana 18,8 sobre 20 inyectado (p10 16,2). Con umbral 10 no hubo eventos falsos.
- **El máximo de esta máquina es ~160 beams, y lo fija la RAM.**
  - Con 160 beams sobran ~2–4 GB de RAM, y no queda memoria ni para arrancar un proceso Python nuevo (por eso `rpc status` falla con un error de MKL).
  - Más beams achican el ring buffer por debajo de ~25 s. El barrido completo a DM 1000 en 300–501,6 MHz dura ~30 s, así que un volcado por trigger necesita al menos eso.
  - Hacia 176 beams la GPU también quedaría al ~98 %.

### Salvedades

- El fake X-engine corre en la misma máquina. Con datos gaussianos consume ~45 % de la CPU y le baja algo de margen al servidor (1,25× en vez de 1,39×).
- Los FRBs inyectados son del simulador de pirate: pulsos ideales, sin RFI ni smearing intracanal. El grouper es el "toy" (un evento por chunk y beam sobre el umbral).
- Cosas que no se probaron:
  - volcados del ring buffer por RPC (`write_files`, `start_stream`);
  - una tarjeta de red real, porque la prueba fue por loopback;
  - corridas largas (horas).
