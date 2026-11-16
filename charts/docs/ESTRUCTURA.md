# Estructura del repositorio

## Qué es de pirate y qué es nuestro

**pirate** (Kendrick Smith, CHORD) es la carpeta de arriba de `charts/` (este repositorio es el fork de CHARTS) y **no se modificó**. Hace dos cosas:
recibe los datos por red (y los guarda en un anillo en RAM) y dedispersa en la GPU con el árbol, entregando un mapa
de S/N por DM y tiempo. **Todo lo de este repositorio es nuestro:** alimentar a pirate, interpretar su salida y medir.

## La cadena y qué archivo hace cada parte

```
 1. SIMULADOR      2. EMISOR             3. PIRATE           4. GROUPER            5. SIFTER          6. EVALUACIÓN
 simulador/     ─► emisor/emisor.py ──► (externo)       ──► servidor/grouper/ ──► servidor/sifter/ ──► evaluacion/
 genera los .fil  simula el X-engine   dedispersión        grouper_v1.py         sifter_alertas.py   evaluar_prueba_64b.py
                                                                                 ALERTAS DE FRB      verificar_alertas.py
                   + RFI + máscara      en GPU              eventos en el mapa
                          └──────────── todo lo arranca servidor/lanzador.py ────────────────┘
```

## Carpetas

| Carpeta | Para qué | ¿Hace falta para correr la prueba? |
|---|---|---|
| `simulador/` | Genera los 4000 `.fil` de prueba (FRB simulados con formas del catálogo de CHIME): `12_generar_set_comparacion_21504.py` y sus dependencias, catálogos y sorteos | sí, una vez (los datos no van en git) |
| `cuantizacion/` | Escalamiento de 8 a 4 bits: teoría (`teoria_cuantizacion.py`), medición del S/N en el set (`cuantizacion_21504.py`), pérdida de rango (`figura_rango_cuantizacion.py`), validación de archivos y figuras. Respalda `docs/ANALISIS_pirate_cuantizacion_…` | no |
| `configs/` | `prueba_64b.yml`: **la única config** de la prueba (beams, canales, red, memoria, dedispersión, grouper, emisor). Las demás son las configs de pirate de las corridas reportadas | sí |
| `emisor/` | El emisor independiente (sin pirate): `protocolo.py` (bytes del protocolo de red), `cuantizacion.py` (8 → 4 bits, máscara), `fuentes.py` (de dónde salen los datos), `emisor.py` (procesos + conexiones) | sí |
| `servidor/` | `lanzador.py` (arma las configs de pirate desde el YAML y arranca sifter, grouper y pirate; modos `servidor`, `emisor`, `local`), `grouper/` (grouper v1 y v0) y `sifter/sifter_alertas.py` (decide qué es un FRB y emite las alertas; servicio gRPC FrbSifter de pirate) | sí |
| `evaluacion/` | Análisis de las corridas (ver la tabla de abajo) | sí (`evaluar_prueba_64b.py`) |
| `rfi_sitio/` | RFI medida en el sitio: canalización ×32, modelo de RFI, máscaras m1/m2, inyección (`rfi_sitio.py`, la usa el emisor con `--rfi`), lectura de cabeceras .fil (`sigproc.py`) | solo con RFI |
| `busqueda_offline/` | `pirate_fil_search.py`: buscar un solo .fil con pirate, sin red | no |
| `analisis_rfi/` | Análisis de las corridas con RFI (conteo sin topes, coincidencias al azar, completas con máscara) y monitor de recursos por proceso | no |
| `escalamiento/` | Memoria y capacidad de pirate según DM máximo, tamaño de bloque y `max_gpu_clag` | no |
| `docs/` | Resultados, protocolo y adaptación a Kotekan, instalación y funcionamiento de pirate, cuantización | para leer |
| `figuras/` | Ejemplos: FRB con y sin RFI, canalización, conteos | para leer |

## Para adaptar a otra arquitectura

- **Otra fuente de datos** (en vivo, otro formato): una clase con `bloque(t0, T, out)` en `emisor/fuentes.py`.
- **Otro emisor** (Kotekan, FPGA): implementar `emisor/protocolo.py` en ese lenguaje; ver
  `docs/PROTOCOLO_pirate_y_adaptacion_Kotekan.md`.
- **Otro PC de búsqueda:** cambiar `red.receptores` (IP del PC de búsqueda) y la memoria en el YAML; pirate se autodetecta (entorno activo, `CHARTS_PIRATE_ENV` o `entornos.pirate`).
- **Otro número de beams o canales:** `prueba.beams` / `prueba.canales` en el YAML; el lanzador arma las configs de pirate.

## Análisis de las corridas (`evaluacion/` y `analisis_rfi/`)

Todos leen una carpeta de corrida (la que deja el lanzador) y escriben sus tablas en ella o en `<corrida>/analisis/`.

| Script | Qué responde |
|---|---|
| `evaluacion/evaluar_prueba_64b.py` | **El principal:** completitud por set, falsas alarmas por beam·hora y FP en negativos, a umbrales fijos y a igual tasa de FA → `resultados.md` |
| `evaluacion/verificar_alertas.py` | Cada alerta de `alertas.jsonl` contra los FRB inyectados: verdaderas, no-FRB, falsas alarmas, FRB sin alerta y latencia → `alertas.md` |
| `evaluacion/tasas_estandar.py` | Completitud a 10 FA por beam·hora y a 10 FA/h del sistema (`--log eventos_todos.log` para el registro completo) |
| `evaluacion/perfil_grouper.py` | Tiempo de detección por bloque: grouper trabajando vs esperando a pirate, por etapa (requiere `perfil: true`) |
| `evaluacion/resumen_recursos.py` | GPU, RAM y CPU por proceso (de `analisis_rfi/monitor_procesos.py`) |
| `evaluacion/resumir.sh` | Resumen rápido del `server.log`: bloques, beams equivalentes, Gbit/s |
| `evaluacion/evaluar_al_terminar.sh` | Espera a que termine una corrida y la evalúa |
| `evaluacion/analisis_perdidas_64b.py`, `eventos_cercanos_64b.py` | FRB perdidos y eventos cercanos a cada archivo |
| `evaluacion/ver_ventanas.py` | Muestra ventanas del mapa DM-tiempo guardadas por el grouper |
| `evaluacion/estadistica_mapa/` | Estadística del mapa: ruido vs señal, S/N medido vs esperado (`estadistica_mapa.py`) y por qué se pierde cada FRB (`perdidos.py`) |
| `evaluacion/subbandas/` | Banda completa vs subbandas `[0,0,1,2,1]` archivo por archivo (`comparar_pareado.py`; config `configs/charts_64b_dm1000_sb1.yml`) |
| `analisis_rfi/analizar_todos.py` | Conteo de todos los candidatos con y sin RFI (sin topes del grouper) |
| `analisis_rfi/comparar_diag.py` | Variantes del grouper y máscaras, con corrección por aciertos al azar |
| `analisis_rfi/analizar_completas.py` | Corridas completas con máscara: Recall, FA y aciertos al azar |
| `analisis_rfi/monitor_procesos.py` | Registra recursos por proceso cada 5 s durante una corrida |

Variables: `CHARTS_CORRIDA` (corrida por defecto de `estadistica_mapa/`), `CHARTS_SALIDA_ANALISIS` (dónde escriben),
`CHARTS_RESULTADOS_SUBBANDAS` y `CHARTS_RESULTADOS_RFI` (carpetas con varias corridas para comparar). Los scripts de
`analisis_rfi/` se corren con el Python del entorno de pirate. Si falta una variable, el script lo dice y termina.
