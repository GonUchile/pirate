# pirate en CHARTS, set de 21 504 canales (grupo 3) y cuantización 8 → 4 bits

**Fechas:** 2026-10-08 y 2026-10-09
**Máquina:** RTX 4090 (24 GB), Xeon w5-3435X (32 hilos), 314 GiB de RAM, NVMe
**Documento relacionado:** `GUIDE_pirate_installation_usage_benchmark_CHARTS_2026-10-08.md` (instalación y comandos de pirate)

---

## 0. Resumen

| Tema | Resultado |
|---|---|
| **pirate instalado** | `environments/pirate` + `sources/{pirate,ksgpu}`. Los tests unitarios pasan. |
| **Capacidad, solo GPU** | 21 504 canales, 1 ms, DM ≤ 1107: ~760 beams en tiempo real si todo cabe en VRAM; ~250–300 con el ring buffer en RAM. El límite es memoria + PCIe, no cómputo. |
| **Búsqueda offline** (672 canales, 300 archivos) | A igual tasa de FP por ruido (10/h), pirate tiene el mejor Recall: A 100 %, B 68 %. Su FP no-FRB es alto (67 %), igual que AA. |
| **Tiempo real con ring buffer** (cadena completa) | Máximo de esta máquina: **160 beams** de 21 504 canales, DM ≤ 4427; 1,25–1,39× más rápido que el tiempo real. **100 % de los 1557 FRBs inyectados detectados.** El límite es la RAM. |
| **RFI** | El modelo `lineas10` (0 dB) no deja detectar nada (P8). La prueba de 64 beams se hace **sin RFI**; vienen mediciones nuevas de RFI. |
| **Grupo 3** (21 504 canales) | Generador `12_generate_comparison_set_21504.py` escrito y validado con 40 archivos. Generación completa **en curso** desde el 2026-10-09. |
| **Cuantización 8 → 4 bits** | Con **n = 2 niveles/σ y centro 32,5** se conserva 98,97 % del S/N en bursts débiles, con tope ±3,5σ (46 cuentas). Los bursts que superan ~14 cuentas por píxel sobre el piso se recortan: se detectan igual, pero con el S/N subestimado. |

---

## 1. pirate: qué es y cómo quedó instalado

pirate (`pirate_frb`, K. Smith, CHORD) es una búsqueda de FRBs en tiempo real para GPU. Usa **dedispersión en árbol**, con un costo ~O(N_f·N_t·log N_DM), en vez de fuerza bruta. Tiene un servidor que recibe int4 por TCP desde el X-engine, un ring buffer en RAM, peak-finding en la GPU y un grouper/sifter conectados por gRPC.

- **Rutas:**
  - entorno: `<entorno de pirate>`
  - fuentes: `sources/pirate` y `sources/ksgpu`
  - trabajo CHARTS: `projects/pirate_charts/`
- **Problemas de instalación resueltos:**
  - `pip` apunta a un Python 2.7: usar `python -m pip`.
  - Faltaba `clang-format`: se instaló en el entorno.
  - Carrera en `make -j` con `asdf/config.hxx`.
  - `config.mk` fija la compilación a sm_89.
- **Orden de frecuencias:** el canal 0 de pirate es la frecuencia **más baja**. El `.fil` (foff < 0) hay que invertirlo.

---

## 2. Capacidad de la GPU (`pirate_frb time_dedisperser`)

Mide subida H2G de int4 + decuantización + dedispersión + peak-finding. No incluye red, RFI ni grouper.

### DM ≤ 1107 (rank 15, un árbol, sin subbandas)

| Beams asignados | max_gpu_clag | VRAM | RAM | Capacidad (beams en tiempo real) |
|---:|---:|---:|---:|---:|
| 64 | 5 | 15,6 GB | 21,5 GB | 761 |
| 128 | 1 | 7,6 GB | 65 GB | 396 |
| 256 | 1 | 13,4 GB | 130 GB | 405 |
| 320 | 1 | 16,2 GB | 162 GB | 392 |
| 384 | 0 | 10,0 GB | 204 GB | 335 ❌ |

### DM ≤ 4427 (rank 15, tres árboles: DM 0–1107 a 1 ms, 1107–2213 a 2 ms, 2213–4427 a 4 ms)

| Beams | clag | VRAM | RAM | Capacidad |
|---:|---:|---:|---:|---:|
| 64 | 1 | 6,9 GB | 84 GB | 255 |
| 128 | 2 | 16,2 GB | 162 GB | 299 |
| 192 | 1 | 15,7 GB | 250 GB | 264 |

Rank 16 (un solo árbol a 1 ms hasta DM ~2200) cuesta ~3,7 GB de RAM por beam, lo que da un máximo de ~64 beams. **El límite es la RAM y el PCIe (~22 GB/s), no el cómputo.**

---

## 3. Búsqueda offline sobre `.fil` y comparación con las otras pipelines

- `projects/pirate_charts/pirate_fil_search.py` lee el `.fil` de 8 bits, normaliza por canal y escribe float16 **directo** en el buffer de la GPU. No pasa por int4 ni por ASDF.
- Decodifica los candidatos con `decode_argmax`, agrupa con máximos locales + supresión de no-máximos, y entrega `(dm, t_top, snr, ancho)`.
- Integrado como 5.ª pipeline del benchmark: `bench/pipelines/pirate.py`.
- Comparación: `resultados_pipelines_clasicas/comparar_pirate.py`.

**300 archivos del grupo 2** (672 canales: 100 de A, 100 de B, 100 de NEG), boxcar ≤ 32 ms:

| | Recall A (S/N ≥ 5) | Recall B | FP ruido/h | FP no-FRB | **Recall A a 10 FP/h** | **Recall B a 10 FP/h** | FP no-FRB a 10 FP/h |
|---|---|---|---|---|---|---|---|
| Heimdall | 87 | 58 | 444 | 45 % | 73 | 54 | 32 % |
| AA | 100 | 85 | 979 | 95 % | 98 | 63 | 56 % |
| CASM-Hella | 95 | 63 | 865 | 66 % | 49 | 43 | 27 % |
| PRESTO | 99 | 66 | 3233 | 63 % | 77 | 54 | 35 % |
| **pirate** | **100** | 79 | 2999 | 88 % | **100** | **68** | 67 % |

- Cada porcentaje tiene un error de ±4–5 %, porque son 100 archivos por set.
- Búsqueda de pirate: 0,76 s por archivo de 119 s, dominado por el sobrecosto de Python.
- Con S/N ≥ 5, pirate genera muchos FP por ruido porque prueba 65 536 DMs. El umbral práctico es ~6–7.

### Calibración del S/N (pulsos boxcar inyectados, S/N verdadero 20)

| Canales | W = 1 ms | W = 4 ms | W = 16 ms |
|---|---|---|---|
| 21 504 | 63–72 % | 86–91 % | 92–109 % |
| 672 | 32–42 % (no detecta a DM ≥ 1000) | 31–79 % | 60–90 % |

pirate supone canales angostos frente al retardo entre canales. A 21 504 canales, que es el caso de CHARTS, funciona bien. DM y tiempo salen con error < 0,1 % y ≤ 1,4 ms.

---

## 4. Tiempo real con ring buffer (cadena completa en esta máquina)

`fake_xengine` (TCP, int4) → `FrbServer` (recepción → ring buffer de frames en RAM → GPU) → `toy_grouper` → `toy_sifter`. Todo por loopback.

- Script: `projects/pirate_charts/correr_tiempo_real.sh`.
- Resumen de una corrida: `tiempo_real/summarize.sh`.
- Cruce de inyectados con detectados: `tiempo_real/cruzar_sifter.py`.

| Corrida | Beams | Receptores | Datos | Ring buffer | Velocidad vs. tiempo real | GPU |
|---|---:|---:|---|---|---|---:|
| b128_junk | 128 | 1 | ceros | ~64 s | 1,18× | 52 % |
| b160_2rx_junk | 160 | 2 | ceros | ~25 s | **1,39×** | 89 % |
| b160_2rx_frb | 160 | 2 | gaussiano + FRBs de S/N 20 | ~25 s | **1,25×** | 87 % |

- **160 beams con FRBs inyectados:** 1557 de 1557 detectados (100 %), en los 3 árboles y en todos los anchos. S/N medido: mediana 18,8 sobre 20. Con umbral 10, ningún evento falso.
- **Con un receptor TCP** la entrada se estanca en 13,5 Gbps. Con dos llega a 19,7 Gbps.
- **El límite es la RAM:** con 160 beams sobran 2–4 GB, y ni siquiera arranca un proceso Python nuevo (`rpc status` falla con un error de MKL).
- **Lecciones operativas:**
  - No editar un `.sh` mientras corre: bash lo relee.
  - `pgrep -f` se encuentra a sí mismo en la línea de comandos del wrapper; hay que anclar el patrón al Python del entorno.

---

## 5. RFI: decisión

- **El RFI de las pruebas anteriores es `lineas10`:** modelo v2 (`10_agregar_rfi_v2.py --umbral-lineas-db 10`), 0 dB, k = 5, impulsos de 1/s, semilla 42.
- **P8 (diagnóstico de 2026-10-01):** con ese modelo, ningún burst de A se detecta (0 de 5) y en B solo los de S/N ≳ 80.
  - La causa es el RFI rápido medido con **una antena a 100 ms**, aplicado al beam con 0 dB y mantenido 100 muestras en todos los canales. El ruido de la serie dedispersada sube ×7–14.
  - Barrido de atenuación: S/N 0,7 (0 dB), 1,6 (10 dB), 1,9 (20 dB) y 3,0 (30 dB), contra 7,3 sin RFI.
- **Decisión del equipo CHARTS (2026-10-09):** la prueba de 64 beams se hace **sin RFI**. Van a llegar mediciones nuevas para modelar el RFI de forma más realista.

---

## 6. Grupo 3: set de comparación a 21 504 canales

### Definición

Es el plan de `simulador_FRB/INVENTARIO_DATASETS.md`, implementado en `simulador_FRB/12_generate_comparison_set_21504.py`.

| Parámetro | Valor |
|---|---|
| Sorteos | **los mismos 1000 + 1000 del grupo 2** (`outputs_comparacion/parametros_sorteados.csv`), semilla 20260923 |
| Física | `modelo()` de `07_`: smearing dentro del canal, espectro normalizado al canal más brillante, scattering que conserva la energía, sobremuestreo ×4 |
| Grilla | **21 504 canales** = cada canal de 672 (300,45 kHz) partido en 32 (**9,389 kHz**); centros de 299,855 a 501,745 MHz |
| Ruido | σ = SEFD / √(Δν·τ) = **768,5 Jy** por canal y muestra (135,9 Jy a 672 canales) |
| A | S/N 8, **recalculado** con el modelo de 21 504 canales |
| B | **S0 = 616,87 Jy**, el mismo que en el grupo 2 |
| Negativos | par de A_i (su `dm_index_negativo`, escala de A_i) |
| Cortados | burst completo a S/N 15 (recalculado), mismo borde y fracción objetivo |
| Llegada | t_top (a 501,6 MHz) sorteado entre 3 y 5 s, con semilla por id |
| Largo | **variable**: ⌈t_top + barrido + cola del pulso + 1 s⌉; en la prueba piloto, 10–77 s |
| Formato | `.fil` de **8 bits**: piso 32, **1σ = 4 cuentas**, misma cuantización que el grupo 2 |
| Salida | `simulador_FRB/outputs_comparacion_21504/` (NVMe), `manifest.csv` por set, `config.json`, `_hechos.jsonl` (permite reanudar) |

### Validación (prueba piloto de 40 archivos, `outputs_comparacion_21504_prueba/`)

Script: `validate_21504.py`.

- **Ruido:** media 32,000 y σ 4,010 cuentas, exactamente como se diseñó. El ruido de la serie dedispersada sale en 1,00 ± 0,02.
- **Set A:** S/N medido − esperado = **+0,19 ± 0,76**, lo esperable con una sola realización de ruido por archivo.
- **B_001** (DM 727, mismo FRB que A_001): el barrido sigue la curva teórica y el S/N medido es 154,9 frente a 153,4 esperado.
- **Ganancia de la grilla fina:** A_000 (DM 1605) necesita 86 Jy para S/N 8, contra 287 Jy a 672 canales (×3,3). Coincide con el estudio de población.
- **Reproducibilidad:** la reconstrucción en float con la misma semilla reproduce los `.fil` **byte a byte**.
- **Figuras:** `plots_21504/validacion_A_001.png` y `plots_21504/validacion_B_001.png`.

### Generación completa

- Lanzada el 2026-10-09 con 24 procesos. Log: `simulador_FRB/outputs_comparacion_21504.log`. ETA ~2,5–3 h, ~4,4 TB.
- Si se corta, se relanza el mismo comando y continúa donde quedó:
  ```
  cd <carpeta de trabajo>/simulador
  .venv/bin/python 12_generate_comparison_set_21504.py --workers 24 --out outputs_comparacion_21504
  ```

---

## 7. Cuantización: de los 8 bits del set a los 4 bits de pirate

### 7.1 Conceptos

- **σ** es la desviación estándar del ruido de **un canal en una muestra de 1 ms**.
  - **Físicamente** sale de la ecuación del radiómetro: 768,5 Jy a 21 504 canales.
  - **En el `.fil`** se fijó la escala para que el piso quede en la cuenta 32 y **1σ = 4 cuentas**.
  - **En operación** se estimaría por canal con una ventana móvil de varios segundos (desviación estándar o 1,4826·MAD). La ventana tiene que ser mucho más larga que un pulso, para no restarlo.
- **Por qué 4 bits:** pirate recibe int4 con escala y offset float16. Es un requisito de su protocolo de red, no del instrumento. CHARTS entrega intensidad en 8 bits.
- **Por qué hay valores negativos:** el código de 4 bits (−7…+7) guarda la **fluctuación respecto al promedio**. El offset guarda el piso y se reconstruye intensidad = offset + escala·q, que sigue siendo positiva. El −8 está reservado para muestras enmascaradas.
- **"n niveles por σ"** = cuántos códigos de 4 bits ocupa 1σ:
  - escalón Δ = σ/n
  - q = redondeo((x − μ)/Δ), recortado a ±7
  - tope = 7σ/n
  - Ejemplos: n = 1 → ±7σ; **n = 2 → ±3,5σ**; n = 3 → ±2,33σ.

### 7.2 Teoría (señal débil en ruido gaussiano)

Fracción de S/N que conserva un cuantizador Q, con tₖ los umbrales, Jₖ el salto de Q en cada umbral y φ la densidad gaussiana:

  η = Σₖ Jₖ·φ(tₖ) / √Var[Q(x)]

Script: `simulador_FRB/quantization_theory.py`.

**Desde el dato continuo:**

| n | Escalón | Tope | S/N conservado | Muestras de ruido recortadas |
|---|---|---|---|---|
| 1 | 1σ | ±7σ | 96,1 % | 0 |
| 1,5 | 0,67σ | ±4,7σ | 98,2 % | 0 |
| 2 | 0,5σ | ±3,5σ | 99,0 % | 0,02 % |
| 2,83 | 0,35σ | ±2,5σ | **99,35 % (óptimo)** | 0,8 % |
| 3 | 0,33σ | ±2,3σ | 99,3 % | 1,2 % |
| 4 | 0,25σ | ±1,75σ | 98,7 % | 6 % |

**Desde los 8 bits del set** (la cadena real, `.fil` → emisor):

| Conversión | S/N conservado |
|---|---|
| 8 bits (el `.fil`) | 99,74 % |
| n = 2 con centro 32 (`q = rint((x−32)/2)`) | 98,22 % (los empates en ,5 arman intervalos desiguales) |
| **n = 2 con centro 32,5** (`q = rint((x−32,5)/2)`) | **98,97 %**; cada código agrupa exactamente 2 cuentas |
| n = 3 (`q = rint((x−32)·3/4)`) | 99,03 % |
| n = 1 | 95,35 % |

### 7.3 Medición con el mismo ruido y la misma señal

Script: `quantization_21504.py`. Reconstruye el float exacto de cada archivo de la prueba piloto y lo cuantiza de todas las formas.

| Cuantización | Teoría | Medido en los B brillantes (S/N 57–156) |
|---|---|---|
| 8 bits | −0,26 % | −0,2 a −0,4 % |
| 4 bits, 3 niveles/σ | −1,0 % | −1,0 a −1,6 % |
| 4 bits, 2 niveles/σ (centro 32) | −1,8 % | −2,2 a −2,8 % |
| 4 bits, 1 nivel/σ | −4,7 % | −4,9 a −5,8 % |
| 4 bits estilo pirate (escala/offset por bloque de 256 muestras) | −1,8 % + ~ancho/256 | −1,8 a −6 %; **A_007 (scattering 26 ms): −22 %** |

- En los bursts de S/N 8 los puntos se dispersan ±5 %: el error de cuantización suma algo de ruido nuevo. Los bursts brillantes miden la pérdida limpia.
- **El offset por bloque de pirate es un problema de implementación:** la media del bloque incluye el pulso y lo resta. Hay que usar un offset de largo plazo.

### 7.4 Rango: qué pasa con los bursts fuertes

Script: `fig_quantization_range.py`. Usa el mismo pulso de B_001 escalado, con el mismo ruido.

| Escala | Pico por píxel | Cuentas sobre el piso | S/N real | 8 bits | 4b n = 2 | 4b n = 3 | 4b n = 1 |
|---|---|---|---|---|---|---|---|
| ×1 | 0,8σ | 3 | 153 | 100 % | 99 % | 98 % | 96 % |
| ×3 | 2,4σ | 10 | 460 | 100 % | 98 % | 92 % | 96 % |
| ×5 | 4σ | 16 | 767 | 100 % | 93 % | 79 % | 96 % |
| ×10 | 8σ | 32 | 1534 | 100 % | 70 % | 53 % | 94 % |
| ×30 | 24σ | 96 | 4603 | 100 % | 32 % | 23 % | 53 % |
| ×100 | 80σ | 320 (satura 255) | 15 343 | 92 % | 12 % | 9 % | 21 % |

(La columna n = 2 se midió con el redondeo de centro 32; en estos casos manda el recorte, así que el centro 32,5 casi no la cambia.)

- **El S/N es el de la suma.** Un píxel de B_001 está apenas ~3 cuentas sobre el piso. El S/N 153 sale de sumar ~6900 canales × ~28 ms ≈ 190 000 píxeles: S/N ≈ (altura media ~0,35σ) × √190 000 ≈ 150.
- **El recorte actúa en cada píxel antes de sumar.** Con n = 2, todo lo que pasa de **46 cuentas** (+3,5σ) se guarda como 7.
- **Cuándo un burst es "fuerte":** pico por píxel ≈ S/N / √(canales con emisión × ancho en ms).
  - Banda ancha y ~10 ms (como B_001): satura con S/N ≳ 400.
  - 1 ms y banda angosta (~2000 canales): satura con S/N ≳ 90.
  - En los B de la prueba piloto, el pico por píxel va de 0,01 a 0,8σ.
- **Consecuencia:** se detectan igual, pero el S/N, la fluencia y la morfología salen subestimados. Los volcados del ring buffer de pirate también quedan en 4 bits. Si se asume la distribución estándar de flujos (N ∝ S^−1,5), serían ~1 % de los eventos con S/N > 100 y ~0,2 % con S/N > 300.
- **Mitigaciones:**
  1. Guardar o volcar los 8 bits del lado de CHARTS: detectar con 4 bits y medir con 8.
  2. Marcar los eventos con muchas muestras en ±7: su S/N es un mínimo.
  3. Probar una escala variable por bloque, con offset de largo plazo.

### 7.5 Recomendación para el emisor a pirate

```
q = redondeo((x − 32,5) / 2), recortado a ±7     escala = 2 cuentas (0,5σ), offset = 32,5
```

- n = 2 con centro 32,5 conserva 98,97 % del S/N en bursts débiles, prácticamente igual que n = 3 (99,03 %).
- Tiene **50 % más de rango** que n = 3: ±3,5σ, o sea 46 cuentas.
- Cada código corresponde exactamente a 2 cuentas del `.fil`.
- El set se mantiene en 8 bits: es lo que entrega CHARTS, sirve para todas las pipelines y la conversión es un paso del emisor que se puede cambiar.

### 7.6 Figuras (`simulador_FRB/plots_21504/`)

| Figura | Contenido |
|---|---|
| `cuantizacion_explicada.png` | histogramas del ruido, pulso dedispersado y barras teoría/medición |
| `pulsos_cuantizados_B_001.png` | el mismo pulso con cada cuantización: barrido, alineado al DM y píxeles crudos |
| `quantization_21504_teoria.png` | pérdida de S/N archivo por archivo, con la línea teórica |
| `rango_cuantizacion_B_001.png` | pérdida de rango: S/N conservado según el brillo y el pulso ×30 aplanado |
| `cuantizacion_en_cuentas.png` | todo en cuentas: ruido de 8 bits, conversión cuenta → código, pulso brillante en 8 bits y en 4 bits |

---

## 8. Próximos pasos

1. **Terminar el grupo 3:** generación en curso (sección 6). Al terminar, revisar el log y los `manifest.csv`, y actualizar `INVENTARIO_DATASETS.md`.
2. **Emisor de 64 beams para pirate:** una copia del `fake_xengine` que llena los `AssembledFrameSet` desde los `.fil`.
   - 8 → 4 bits con n = 2 y centro 32,5.
   - Reloj de 1,000 ms exactos (`dt_ns_per_seq = 5000`, `seq_per_frb_time_sample = 200`).
   - Contador de muestras recortadas por beam.
3. **Prueba de 64 beams** con los 4000 archivos repartidos en 64 streams (~1 h de stream por beam) y búsqueda hasta DM 1000 (rank 15, un árbol).
   - Recall sobre los archivos con DM ≤ 1000 (1209).
   - FP por ruido (/h) y FP no-FRB sobre todos.
   - Tiempo real y uso de GPU, CPU y RAM.
4. **RFI:** repetir con el modelo nuevo cuando lleguen las mediciones.
5. **Subbandas de pirate** (opcional): compilar `(0,0,0,3,1)` y ver si sube el Recall de B en bursts de banda angosta.

---

## 9. Archivos

| Archivo | Qué hace |
|---|---|
| `projects/pirate_charts/pirate_fil_search.py` | búsqueda offline de pirate sobre `.fil` (fp16 directo, opción `--int4`) |
| `projects/pirate_charts/calibrate_snr.py` | calibración del S/N con pulsos inyectados |
| `projects/pirate_charts/correr_tiempo_real.sh` | cadena completa en tiempo real |
| `projects/pirate_charts/tiempo_real/{summarize.sh,cruzar_sifter.py}` | resumen de corridas y cruce inyectados/detectados |
| `projects/pirate_charts/configs/` | configs de dedispersión y servidor para CHARTS |
| `simulador_FRB/resultados_pipelines_clasicas/bench/pipelines/pirate.py` | pirate en el benchmark |
| `simulador_FRB/resultados_pipelines_clasicas/comparar_pirate.py` | comparación de las 5 pipelines con el formato Recall + FP |
| `simulador_FRB/12_generate_comparison_set_21504.py` | generador del grupo 3 |
| `simulador_FRB/validate_21504.py` | validación visual y del S/N |
| `simulador_FRB/quantization_21504.py` | medición de cuantizaciones con el mismo ruido |
| `simulador_FRB/quantization_theory.py` | eficiencia teórica η |
| `simulador_FRB/figura_*.py`, `plot_quantization_theory.py` | figuras de la sección 7 |

---

## 10. Resultado: prueba de 64 beams en tiempo real (2026-10-10, sin RFI)

**Montaje**
- `projects/pirate_charts/emisor_fil.py` (el emisor) y `correr_prueba_64b.sh` (el lanzador); evaluación con `evaluate_test_64b.py`.
- Resultados en `tiempo_real/completa_sin_rfi/resultados.md`.
- Los 4000 archivos del grupo 3 se repartieron en 64 streams continuos de 3510 s cada uno. Pasan por la cadena completa: TCP → ring buffer → GPU → grouper → sifter.
- Configuración: 4 bits con n = 2 (centro 32,5); búsqueda DM 20–1105, rank 15 con un árbol, boxcar ≤ 32 ms.

**Recall + FP con DM ≤ 1000** (297 de A, 297 de B, 297 negativos y 318 cortados):

| Umbral | FP ruido/h (64 beams) | Recall A | Recall B | Recall cortados | FP no-FRB |
|---|---|---|---|---|---|
| S/N ≥ 5 | 1531 | 97,0 % | 86,5 % | 71,4 % | 61,6 % |
| S/N ≥ 6 | 243 | 95,3 % | 84,8 % | 69,5 % | 52,5 % |
| **S/N ≥ 6,54 (10 FP/h)** | **8,9** | **91,9 %** | **82,8 %** | **64,5 %** | **41,1 %** |
| S/N ≥ 7 | 0 | 84,8 % | 81,1 % | 60,1 % | 33,3 % |

**Lectura**
- **B:** se detecta el 98,3 % de los bursts cuyo S/N esperado supera el umbral. Lo que falta en B es casi todo por debajo del umbral por física.
- **Cortados:** se detecta el 93,7 % de los que superan el umbral con su parte visible.
- **A** (S/N 8, justo sobre el umbral):
  - se pierde el 8 %, sobre todo pulsos anchos o con scattering ≥ 10 ms a 600 MHz (80 % de Recall), que exceden el boxcar máximo de 32 ms;
  - el S/N medido por pirate tiene mediana 8,33, frente a 8 esperado.
- **FP por ruido:** caen muy rápido con el umbral. Con S/N 6,5 son ~9 por hora sumando los 64 beams (~0,14 por beam y hora).

**Rendimiento**
- La cadena completa procesó a 1,98× el tiempo real; el servidor da para ~126 beams equivalentes con un solo receptor.
- GPU con 36 % de uso medio y VRAM máxima de 23,4 GB (de 24; la ocupan servidor, grouper y la conversión del emisor).
- CPU 66 % ociosa.
- Muestras recortadas en ±7: 0,126 %, exactamente lo que da el ruido gaussiano solo, así que los bursts no se recortan.
