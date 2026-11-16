# CHARTS · búsqueda de FRB en tiempo real con pirate (64 beams)

Código, configuraciones y resultados de la prueba de búsqueda de FRB en tiempo real para **CHARTS** con
[pirate](https://github.com/kmsmith137/pirate) (dedispersión en árbol en GPU): **64 beams × 21 504 canales**
(300–502 MHz, 9,4 kHz, 1 ms), DM ≤ 1105, en una RTX 4090.

El proyecto está separado en piezas independientes que se comunican solo por la red, con el protocolo de pirate:

```
  PC que produce los datos (X-engine o prueba)           PC de búsqueda (GPU)
  ┌──────────────────────────────────┐                 ┌───────────────────────────────────────────────┐
  │ emisor/  (Python + numpy)        │   TCP, 4 bits   │ servidor/lanzador.py                          │
  │  lee .fil u otra fuente,         │ ──────────────► │  pirate (externo): anillo en RAM + árbol GPU  │
  │  RFI y máscara opcionales,       │  + escala/offset│  grouper v1: eventos en el mapa DM-tiempo     │
  │  4 bits + escala/offset          │                 │  sifter: decide y emite ALERTAS DE FRB        │
  └──────────────────────────────────┘                 └───────────────────────────────────────────────┘
                                                        alertas.jsonl → evaluacion/: alertas verificadas,
                                                                        completitud y falsas alarmas
```

Los datos de prueba (4000 FRB simulados, `.fil` de 21 504 canales) se generan con `simulador/`.
El emisor no depende de pirate: puede correr en otro PC. Para llevarlo a **Kotekan**, ver
[`docs/PROTOCOLO_pirate_y_adaptacion_Kotekan.md`](docs/PROTOCOLO_pirate_y_adaptacion_Kotekan.md).
Para entender qué hace cada archivo, ver [`docs/ESTRUCTURA.md`](docs/ESTRUCTURA.md).

## Rutas: lo único que hay que indicar

No hay rutas de una máquina en particular en el código. Para la prueba hacen falta solo dos cosas:

| Qué | Cómo | Si falta |
|---|---|---|
| Dónde está pirate | `conda activate <entorno de pirate>` (se autodetecta), o `export CHARTS_PIRATE_ENV=/ruta/al/entorno` | el lanzador dice que no encuentra `pirate_frb` |
| Dónde están los datos (.fil del grupo 3) | `export CHARTS_SET_DIR=/ruta/al/grupo3` | el lanzador y el emisor lo dicen antes de empezar |

Las demás variables (tabla al final) son solo para scripts opcionales, y cada uno avisa si le falta la suya.
Todo lo demás se busca relativo al repositorio.

## Resultados principales

Detalle en [`docs/RESULTADOS_64beams.md`](docs/RESULTADOS_64beams.md). Corridas completas de 4000 archivos
(297 A, 297 B y 318 cortados con DM ≤ 1000), grouper v1, máscara de canales m1:

| | Sin RFI + máscara | RFI del sitio sin mitigar* | RFI del sitio + máscara m1 |
|---|---|---|---|
| Falsas alarmas a S/N ≥ 6 (por beam·hora) | 674 | ~66 000 | 2 463 |
| Umbral S/N para 10 FA por beam·hora | 6,72 | no se alcanza | 7,04 |
| Completitud A / B / cortados a 10 FA/h | 89 / 83 / 63 % | ~8 % | 77 / 82 / 59 % |
| Falsos positivos en negativos | 41 % | 100 % | 39 % |
| Detección por bloque de 2,048 s (pirate + grouper, sin emisor) | 0,49 s | — | 0,59 s |

\* Prueba corta de 80 archivos; completitud corregida por coincidencias al azar.
Set A: S/N esperado 8 · set B: flujo fijo 617 Jy · cortados: burst parcialmente fuera del archivo · negativos: señales
dispersadas con índice ≠ −2 (no-FRB).

Esos resultados se obtuvieron con una cadena anterior (emisor con el `FakeXEngine` de pirate, en el repositorio
`charts-pirate-64beams`, carpeta `legado/`). La cadena de esta carpeta (`emisor/` + `servidor/lanzador.py`) se validó
contra ella: mismas detecciones en la prueba de 8 archivos.

## Alertas de FRB

La última etapa, `servidor/sifter/sifter_alertas.py`, decide qué evento es un FRB y emite una alerta: una línea JSON en
`<salida>/alertas.jsonl` (id, hora UTC de llegada a 501,6 MHz, beam, DM, S/N, ancho, beams coincidentes, latencia), un
aviso en la consola del lanzador y, si se configura, un comando por alerta con el JSON en la entrada estándar
(`servidor.alertas.accion`: avisar, volcar datos, VOEvent...). Regla por defecto (`servidor.alertas` en el YAML):

- S/N ≥ 7,6 (~10 falsas alarmas por hora del sistema de 64 beams con la RFI del sitio + máscara m1) y 20 ≤ DM;
- veto de RFI: más de 8 beams con la misma llegada;
- un grupo (misma llegada en bloques seguidos o en beams coincidentes) se emite cuando pasa un bloque sin que su S/N
  mejore, con su mejor evento: un pulso brillante aparece antes en DM más bajos y con menos S/N.

`evaluacion/verificar_alertas.py` cruza cada alerta con los FRB inyectados → `alertas.md` (lo corre el modo `local`).
Prueba de 8 archivos sin RFI (2026-11-15): 2 alertas, las 2 FRB verdaderos con el DM correcto, 0 falsas alarmas,
latencia 4,6–5,8 s desde que termina de llegar el pulso (≤ 2 s esperando el fin del bloque + 2 s del bloque sin
mejora + ~2,4 s de pirate y grouper).

## Instalación

Ver **[`INSTALL.md`](INSTALL.md)**: qué instalar en cada PC, la dirección y el commit de cada repositorio (pirate, ksgpu), y el script `instalar_pirate.sh`.

## Uso

```bash
conda activate /ruta/al/entorno/de/pirate          # o export CHARTS_PIRATE_ENV=/ruta/al/entorno/de/pirate
export CHARTS_SET_DIR=/ruta/al/grupo3              # .fil del simulador (A_snr_fijo, B_amplitud_fija, negativos, parcial)

# todo en este PC: servidor + emisor + evaluación (80 archivos)
python servidor/lanzador.py local --config configs/prueba_64b.yml --salida resultados/corta --limite 20 --procesos 16

# con la RFI del sitio y la máscara m1
python servidor/lanzador.py local --config configs/prueba_64b.yml --salida resultados/corta_rfi --limite 20 \
    --rfi --mascara rfi_sitio/mascara_m1.npy

# en dos PC: en el de búsqueda (con red.receptores = IP de este PC en el YAML) ...
python servidor/lanzador.py servidor --config configs/prueba_64b.yml --salida resultados/remota
# ... y en el que produce los datos (solo necesita numpy y PyYAML)
python emisor/emisor.py --config configs/prueba_64b.yml --salida resultados/remota --receptores 10.0.0.2:5000
# después, en el PC de búsqueda
python evaluacion/evaluar_prueba_64b.py resultados/remota
```

Todo se configura en **un archivo** (`configs/prueba_64b.yml`): beams, canales, banda, red (receptores, conexiones),
memoria del servidor, parámetros de dedispersión y del grouper, y opciones del emisor. Las opciones de línea de
comando del emisor (`python emisor/emisor.py --help`) pisan las del YAML.

Antes de lanzar: la GPU libre y los puertos de `red` desocupados (el lanzador lo verifica).

## Requisitos

| Pieza | Necesita |
|---|---|
| `emisor/` | Python ≥ 3.9, numpy, PyYAML. Opcional: cupy (`gpu: true`) y, para `--rfi`, el modelo `rfi_sitio/modelo_rfi_sitio.npz` (177 MB, no va en git; se regenera con `rfi_sitio/01_` y `02_`) |
| `servidor/` | pirate compilado desde [kmsmith137/pirate](https://github.com/kmsmith137/pirate), commit `255edd26` (1.5.0), `sm_89`; su entorno (Python 3.13, numpy 2.5, cupy 14.2); GPU NVIDIA (probado en RTX 4090, driver 590, CUDA 13.1); ~150 GB de RAM con la config por defecto. Instalación: [`docs/GUIA_pirate_instalacion_uso_y_benchmark_CHARTS_2026-10-08.md`](docs/GUIA_pirate_instalacion_uso_y_benchmark_CHARTS_2026-10-08.md). No se modificó pirate |
| `evaluacion/` | el entorno de pirate (usa su `DedispersionConfig`) |
| `rfi_sitio/` 00–07 | numpy, scipy, h5py, matplotlib; el baseband del sitio para regenerar el modelo |
| `simulador/` | Python 3.12, numpy, scipy, fitburst (CHIME/FRB, commit `3c76da8`): genera los 4000 .fil del grupo 3 (~4,3 TB; no van en git) |

Rendimiento del emisor en Python (64 beams, 16 procesos, CPU de 16 núcleos): ~1,07× el tiempo real.

## Variables de entorno

| Variable | Quién la usa | Obligatoria |
|---|---|---|
| `CHARTS_SET_DIR` | emisor (si `emisor.set_dir` del YAML es `null`), `cuantizacion/` | sí, para la prueba |
| `CHARTS_PIRATE_ENV` | lanzador y evaluación (si `entornos.pirate` es `null`). Sin ella se busca `pirate_frb` en el entorno activo y en el PATH | no, si el entorno de pirate está activado |
| `CHARTS_RFI_MODELO` | `--rfi`: `modelo_rfi_sitio.npz`, si no está en `rfi_sitio/` | no |
| `CHARTS_BASEBAND_DIR` | `rfi_sitio/00` y `01` (baseband del sitio) | solo para regenerar la RFI |
| `CHARTS_RESULTADOS_RFI` | `analisis_rfi/` (carpeta con varias corridas) | solo para esos scripts |
| `CHARTS_RESULTADOS_SUBBANDAS` | `evaluacion/subbandas/comparar_pareado.py` | solo para ese script |
| `CHARTS_CORRIDA`, `CHARTS_SALIDA_ANALISIS` | `evaluacion/estadistica_mapa/` (o pasar la corrida como argumento) | no |

## Limitaciones conocidas

- La RFI es una sola captura de 4,1 s (promedio de 8 antenas) que se repite; no incluye impulsos de banda ancha ni las variaciones de nivel comunes a toda la banda; todos los beams la ven igual.
- La máscara sale de la misma captura que se inyecta (conocimiento perfecto de la RFI).
- El grouper (v1) y el sifter de alertas son propios y simples; no hay clasificador (RFI vs. FRB) ni coordinación entre varios nodos de búsqueda.
- 10 FA/h del sistema de 64 beams (0,16 por beam·hora) no se puede medir con estas corridas: el tiempo sin señal es ~3 beam·hora.
- En el emisor, el ruido de los huecos entre archivos se toma de 8 bloques precalculados que se repiten (`ruido_precalculado: false` para generarlo nuevo, unas 5 veces más lento).

Proyecto CHARTS, Departamento de Astronomía, Universidad de Chile.
