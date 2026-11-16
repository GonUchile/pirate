# RFI medida en el sitio para las simulaciones de 21 504 canales

Creado el 2026-10-29. Reemplaza, para el grupo 3, al modelo RFI v2 (`motor/rfi_potencia.py`, `lineas10`), que venía de
un espectrómetro de 1 antena a 100 ms. Aquí la RFI sale de voltajes reales del sitio, procesados a la misma
resolución que las simulaciones (21 504 canales de 9,375 kHz, 1 ms).

## Datos de entrada

`260902T161623Z_CHARTS_hdf5/` (25 GB; `export CHARTS_BASEBAND_DIR=...` para 00 y 01). Captura de baseband del 2026-09-02.

| | |
|---|---|
| Forma | 32 entradas × 672 canales de 0,3 MHz (300,0–501,6 MHz) × 1 228 800 muestras |
| Muestreo | 3,333 µs (= 1/0,3 MHz); duración 4,096 s |
| Formato | `uint8` = una muestra compleja: 4 bits reales (nibble alto) + 4 imaginarios (bajo), complemento a 2 |
| Archivos | 8 grupos de 84 canales × 7 chunks; `baseband_virtual.h5` los une (32, 672, 1228800) |

**Solo las entradas 0–7 tienen señal** (ruido gaussiano, curtosis 2,0, σ de 0,7 a 1,5 niveles por componente).
Las 8–31 están cuantizadas casi solo en 0 y ±1 (σ ≈ 0,4 niveles, 74 % de ceros): no tienen señal útil.
Ver `diag_entradas.png`.

Otra medición en el mismo disco, no usada aquí: `corr_10_24s/` (visibilidades cada 10,24 s durante 24 h desde el
2026-10-01). Sirve para ver cómo varían las líneas persistentes a lo largo del día.

## Scripts (en orden)

| Script | Qué hace | Salida |
|---|---|---|
| `00_diagnostico_entradas.py` | Potencia, curtosis y cuantización por entrada | `diag_entradas.png` |
| `01_espectro_dinamico.py` | FFT de 32 por canal grueso → 21 504 canales; \|x\|²; bins de 1 ms exactos (9,375 cuadros de 106,67 µs, reparto fraccional). Entradas 0–7 por separado | `espectro_dinamico.npy` (8 × 4096 × 21504, float32, 2,8 GB), `frecuencias_mhz.npy` |
| `02_modelo_rfi.py` | Piso por entrada, r(t,f) del beam, RFI persistente y rápida, estadísticas | `modelo_rfi_sitio.npz`, `resumen_rfi.json`, `modelo_rfi_sitio.png` |
| `rfi_sitio.py` | Librería `RfiSitio` (numpy o cupy) que agrega la RFI a un bloque del .fil | — |
| `sigproc.py` | Lee la cabecera de un .fil (la usan 03 y 06) | — |
| `03_figura_antes_despues.py` | Figura sin/con RFI de un archivo | `B_277_sin_vs_con_rfi_sitio.png` |
| `05_figura_upcanalizacion.py` | Antes (672 canales de 0,3 MHz) y después (21 504 de 9,375 kHz) de canalizar, ambos a 1 ms, una figura por antena | `figuras_upcanalizacion/antena{0..7}.png` |
| `04_escribir_fil.py` | El baseband canalizado ×32 e integrado a 1 ms, como .fil: por entrada (antena); float32 sin normalizar y 8 bits calibrado (piso 32, σ = 4 por canal) | `fil/sitio_260902_entrada{0..7}_{f32,8bit}.fil` |

Entorno: numpy, scipy, h5py y matplotlib (`requirements.txt`) para 00–07; el del emisor (cupy opcional) para inyectar en tiempo real.

Validación del procesamiento: en canales limpios σ/media = 0,327 a 1 ms, igual a 1/√(9,375 kHz × 1 ms).

## Modelo

Por entrada a: `piso_a(f) = B_a(c)·S_a(k)` (B: nivel del canal grueso sin RFI, mediana corrida iterativa;
S: scalloping de la PFB dentro del canal grueso, 0,63 en los bordes a 1,15 al centro) y
`r_a(t,f) = P_a(t,f)/piso_a(f) − 1`, la potencia de RFI en unidades de la potencia del sistema.

Beam: `r = promedio de r_a`. Para un beam fuera del lóbulo principal (lóbulos laterales con fases aleatorias) la
razón RFI/sistema es en promedio la de una antena, así que la atenuación por defecto es 0 dB.

- **Persistente** `r_p(f)`: mediana en los 4,096 s.
- **Rápida** `e(t,f)`: exceso sobre r_p mayor que 5 σ (σ = MAD temporal por canal), más los vecinos conectados > 3 σ.

### Resultados (`resumen_rfi.json`)

| | |
|---|---|
| Canales con líneas > 10 dB | 2,1 % (457 canales) |
| Canales con r_p > 1 (> 3 dB) | 12,2 % |
| Zonas peores | 410–432 MHz (S/N retenido 0,46 tras calibrar) y 300–340 MHz (0,68) |
| S/N retenido para espectro plano (calibración) | 0,79 |
| Píxeles con RFI rápida | 0,49 % |
| Eventos rápidos | ~22 000 /s en toda la banda; 1 ms de mediana (p99 8 ms); ~19 kHz de ancho (p99 0,12 MHz) |
| Impulsos de banda ancha (> 10 % de la banda en 1 ms) | ninguno en 4,1 s |
| RFI rápida externa (coincide en otra entrada) | 40–70 % en las entradas 0, 4, 5, 6, 7; ~1 % en 1 y 2 (local) |
| Saturados en 255 tras inyectar | 0,68 % |

Se ven además un tren de pulsos periódicos en ~452 MHz (cada ~85 ms), una línea que aparece a los 2,4 s en
400 MHz y modulación en 362 MHz.

## Cómo se agrega a un .fil (`rfi_sitio.py`)

Con s = x − 32 (señal + ruido de σ0 = 4 cuentas), P_sys = 4·√9,375 = 12,25 cuentas, g = 10^(−atenuación/10),
rp = g·r_p(f), re = g·e(t,f):

- Canal **visible** (línea con 10 log10(1 + rp) > `umbral_lineas_db`, 10 dB por defecto como en RFI v2):
  `y = 32 + s + P_sys·(rp + re) + σ0·√((1 + rp + re)² − 1)·N(0,1)`
- Canal **calibrado** (el offset/escala por canal del 8 bits normaliza la potencia persistente):
  `y = 32 + [s + P_sys·re + σ0·√((1 + rp + re)² − 1)·N(0,1)] / (1 + rp)`

Se redondea y se recorta a 0–255. Es la misma física que RFI v2 (`10_agregar_rfi_v2.py`).

Archivos más largos que 4,096 s: bootstrap de bloques contiguos de 4,096 s con desfase aleatorio dentro de la
captura (determinista por semilla y bloque).

## Uso

**Tiempo real, sin escribir a disco** (`emisor/emisor.py --rfi`):
```
python servidor/lanzador.py local --config configs/prueba_64b.yml --salida resultados/rfi --rfi --mascara rfi_sitio/mascara_m1.npy
```
Todos los beams usan el mismo reloj de RFI (llega a todos a la vez por los lóbulos laterales).
`--rfi-ganancia-beam-db` da a cada beam una ganancia distinta. En GPU cuesta ~10 ms por bloque y beam
(0,6 s por chunk de 2,048 s con 64 beams) y 176 MB.

## Limitaciones

- **Una sola captura de 4,1 s.** En archivos largos la RFI se repite por bloques; no hay variación en escalas de
  minutos u horas, ni impulsos de banda ancha (no aparecieron en 4,1 s). Hace falta más baseband, o usar
  `corr_10_24s/` para modular las líneas persistentes en el día.
- **4+4 bits con σ ≈ 1 nivel:** las líneas más fuertes (> ~15 dB) se saturan en ±7 y su potencia queda subestimada.
- **8 antenas sin formar beam:** se asume que el beam ve la RFI como el promedio de las antenas (lóbulo lateral
  promedio). Beams con un nulo o un lóbulo hacia la fuente verían menos o más; se puede variar con
  `atenuacion_db` y `ganancia_beam_db`.
- **Grilla de frecuencias:** la captura usa 0,3 MHz por canal grueso y las simulaciones 0,30045 MHz. Cada canal
  simulado toma el canal medido más cercano en frecuencia, y se asume que 300,0 MHz es el centro del canal grueso 0.

## Primeras pruebas (2026-10-29)

**Offline, B_277 con `pirate_fil_search.py`** (normaliza cada canal por chunk):

| | 8 bits | 4 bits (n = 2) |
|---|---|---|
| S/N del FRB sin → con RFI | 30,7 → 27,9 | 26,3 → 23,4 |
| Candidatos S/N ≥ 5 en 18 s, sin → con RFI | 155 → 1763 | 158 → 1087 |

La RFI genera candidatos a DM 44–121 con S/N 11–14.

**Tiempo real, 64 beams, prueba corta (80 archivos, `projects/pirate_charts/tiempo_real/`):**

| Corrida | A 10 FP/h: umbral | Recall A / B / cortados | Máximo típico por chunk |
|---|---|---|---|
| `prueba_corta_orig_cache` (sin RFI) | 6,54 | 100 / 83 / 50 % | S/N ~5,8 |
| `prueba_corta_norm` (sin RFI, normalización por canal) | 6,68 | 100 / 83 / 50 % | S/N ~5,9 |
| `prueba_corta_rfi_sitio` (RFI + normalización) | no se alcanza (1116 FP/h a S/N 5–8) | 0 / 33 / 0 % | S/N ~20, 48 ms, DM ~78 |

- La cuantización fija del emisor no sirve con RFI: los canales saturados suman un offset y todo chunk da S/N ≈ 118.
  Por eso `--rfi-sitio` activa `--normalizar-canales` (mediana y σ por canal y chunk; saturados → −8).
- Aun normalizando, la RFI que varía dentro del chunk da S/N ~20 en boxcars anchos a DM bajo, y el toy grouper
  (un evento por chunk y beam) entrega siempre ese máximo. Falta mitigación de RFI (enmascarar canales por
  varianza/curtosis, o el clasificador de la etapa 1) y un grouper con varios eventos por chunk.

## Variaciones de nivel en toda la banda (no incluidas en el modelo de inyección)

En el .fil de 8 bits se ven saltos de nivel en toda la banda a la vez (~1,5, 2,1, 2,7 y 3,5 s). En los 6457 canales
limpios (r_p < 0,05), el promedio de banda a 1 ms fluctúa más que el ruido (σ esperado 0,0124 por canal):
×2,3–3,1 en las entradas 0, 1, 2, 5 y 6; ×4,7–7,5 en las 3, 4 y 7.
Son variaciones comunes a todos los canales, de ~0,03–0,09 σ por canal: cada píxel queda bajo el umbral de 5σ, así
que `e(t,f)` no las captura y la inyección actual no las reproduce. Sumadas sobre la banda pesan mucho a DM ≈ 0 y
DM bajo con boxcars anchos. Pendiente: agregar un término común r_bb(t) al modelo.
