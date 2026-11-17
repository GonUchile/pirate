# Resultados de la prueba de 64 beams en tiempo real

Período: 2026-10-08 a 2026-11-05. Máquina: RTX 4090 (24 GB), Xeon w5-3435X (16 núcleos), 314 GB de RAM.

## 1. Configuración común

- Datos: grupo 3, 4000 .fil de 21 504 canales (9,389 kHz, 300–502 MHz), 1 ms, 8 bits (piso 32, σ = 4 cuentas), sin RFI propia. Sets de 1000: A (S/N esperado 8), B (617 Jy), negativos (índice de dispersión ≠ −2) y cortados.
- Reparto en 64 streams (uno por beam), 10 s de ruido antes y 60 s después; 3510 s de stream por beam.
- pirate: rank 15, un árbol, DM ≤ 1105, boxcar ≤ 32 ms (`configs/charts_64b_dm1000.yml`), servidor `configs/server_64b.yml`.
- Acierto: DM ±10 % y tiempo ±máx(50 ms, boxcar). Falsas alarmas (FA): eventos fuera de las ventanas de señal, por beam·hora de tiempo sin señal. FP en negativos: no-FRB con algún evento a ±100 ms. Métricas con DM ≤ 1000.

## 2. Referencia sin RFI con el toy grouper (2026-10-10)

`resultados/completa_sin_rfi_toy_grouper.md`: a 10 FA por beam·hora (umbral 6,54), A 91,9 %, B 82,8 %, cortados 64,5 %,
FP en negativos 41,1 %. Cadena a 1,98× el tiempo real.

## 3. RFI del sitio (T3d, 2026-11-03/04)

- Modelo: baseband del sitio (2026-09-02, 8 antenas, 4,1 s) canalizado ×32 a 21 504 canales × 1 ms (`site_rfi/`). 2,1 % de los canales con líneas > 10 dB; RFI rápida angosta en el 0,5 % de los píxeles; sin impulsos de banda ancha.
- Inyección al vuelo en el emisor (`--rfi-sitio`), con el mismo reloj en los 64 beams y normalización por canal y chunk.
- **Sin mitigar** (prueba corta, 80 archivos): ~110× más candidatos (47 por beam y chunk), 99 % presentes en ≥ 32 beams; no hay umbral que llegue a 10 FA/h; completitud real ~8 % (21 % de los "aciertos" eran coincidencias al azar con RFI). Quitar el tope de 4 eventos por beam del grouper no recupera los FRB (`resultados/conteo_todos.md`).
- Diagnóstico del grouper (`resultados/diag_agrupacion.md`): el descarte por t_top no es la causa; la unión de regiones con S/N ≥ 5 absorbe parte de los FRB (con umbral de celda 7 la detección real sube de ~8 a ~28 %), pero el problema de fondo es la RFI en el mapa de pirate.

## 4. Máscara estática de canales (T3e, 2026-11-04/05)

- m1: líneas > 3 dB o RFI rápida > 1 % del tiempo = 17,5 % de la banda (quita el 98 % de la RFI rápida). m2: > 1 dB o > 0,2 % = 39 %.
- Los canales enmascarados van como dato faltante (−8). pirate calcula el S/N suponiendo varianza 1 en todos los canales, así que la escala de los útiles se multiplica por √(N/N_útiles); sin esta corrección todo el S/N queda escalado por √(1 − f).
- Pruebas cortas (`resultados/mascaras.md`): con RFI + m1 la detección real sube a 78 % (control 83 %); m2 pierde más S/N sin mejorar.

### Corridas completas con m1 (`resultados/completas_m1.md`)

| | Sin RFI + m1 | RFI + m1 |
|---|---|---|
| Candidatos (S/N > 6, sin topes) | 49 496 (0,45 por beam y chunk) | 185 502 (1,7) |
| FA a S/N ≥ 6 | 674 por beam·hora | 2 463 |
| Umbral para 10 FA por beam·hora | 6,72 | 7,04 |
| Completitud A / B / cortados | 88,9 / 82,5 / 63,2 % | 76,8 / 81,8 / 58,8 % |
| FP en negativos | 40,7 % | 39,1 % |
| Primer umbral sin FA (≤ 0,3 por beam·hora) | 7,24: 77,1 / 81,1 / 58,8 % | 7,60: 64,3 / 80,5 / 52,8 % |
| Aciertos al azar (S/N ≥ 6) | ≤ 0,7 % | ≤ 2 % |

## 5. Tiempos y recursos

- Detección por bloque de 2,048 s (sin el emisor): pirate 0,17 s (`time_dedisperser`, 764 beams a tiempo real con la GPU dedicada) + grouper v1 0,32 s sin RFI / 0,42 s con RFI + m1 (`--perfil`) = **0,49 / 0,59 s**. La cadena completa con el emisor de prueba tarda 0,96 / 1,44 s por bloque: la frena el emisor.
- Pendiente: medir la recepción por red y el armado de bloques dentro del servidor (el 0,17 s es del benchmark aislado).
- Recursos (prueba corta con RFI + m1, por proceso): servidor de pirate 17,4 GiB de GPU, 150 GiB de RAM (todo el pool asignado; necesita ~20 GiB de árbol + el anillo), ~0,3 núcleos de media; grouper 0,6 GiB de GPU, ~0,15 núcleos.
- Escalamiento (`scaling/`): la memoria del árbol crece ×4 por cada duplicación del DM máximo (rank +1); rank 16 es el máximo de pirate; árboles extra con muestras de 2 y 4 ms son mucho más baratos; con el árbol en RAM, el PCIe (~22 GB/s) limita la capacidad.

## 6. Limitaciones

Ver el README principal. En particular: RFI de una sola captura de 4,1 s repetida (promedio de 8 antenas, más optimista que la antena 4), máscara armada con la misma captura que se inyecta, sin variaciones de nivel comunes a toda la banda y RFI idéntica en todos los beams.
