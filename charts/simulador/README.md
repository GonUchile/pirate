# Simulador de FRB: el set de prueba de 21 504 canales (grupo 3)

Genera los 4000 archivos `.fil` con los que se hizo la prueba de 64 beams: 1000 por set.

| Set | Carpeta | Qué es |
|---|---|---|
| A | `A_snr_fijo` | FRB con S/N esperado 8 (el flujo se ajusta a cada pulso) |
| B | `B_amplitud_fija` | los mismos FRB con flujo fijo de 616,87 Jy (el S/N sale de la física de cada uno) |
| NEG | `negativos` | pares de A con índice de dispersión ≠ −2 (no-FRB) |
| P | `parcial` | FRB con S/N 15 cortados por el borde del archivo (5–95 % de la energía visible) |

Física (de `07_generar_set_comparacion.py`, función `modelo()`): pulso gaussiano con scattering que conserva la
energía, espectro tomado de bursts reales del catálogo de CHIME (formas de `catalogos/`), dispersión con smearing
dentro de cada canal y sobremuestreo ×4. Ruido de 256 antenas (SEFD de `motor/charts_sefd.py`). DM uniforme entre
100 y 3000. Sin RFI (la RFI se agrega al emitir, ver `rfi_sitio/`).

Formato: 21 504 canales de 9,389 kHz (672 gruesos × 32), 299,85–501,75 MHz, 1 ms, 8 bits (piso 32, σ = 4 cuentas),
cabecera SIGPROC con canales en orden descendente. Largo variable (7–98 s). Un `manifest.csv` por set con los
parámetros de cada archivo y `t_top_s` = llegada a 501,6 MHz.

## Archivos

| Archivo | Qué hace |
|---|---|
| `12_generar_set_comparacion_21504.py` | **El generador del grupo 3.** Reanudable (`_hechos.jsonl`), en paralelo |
| `07_generar_set_comparacion.py` | Física del pulso y utilidades; también genera el grupo 2 (672 canales) |
| `03_convertir_a_fil.py` | Escritura de los `.fil` y escala de 8 bits |
| `motor/charts_sefd.py`, `motor/fitburst_sim.py` | SEFD de CHARTS y constantes/forma de pulso de fitburst |
| `catalogos/` | Catálogos de CHIME/FRB (Catálogo 1 limpio y Catálogo 2): de ahí salen ancho, scattering y espectro |
| `parametros_sorteados_grupo2.csv` | Los 2000 sorteos (1000 principales + 1000 cortados) que reutiliza el grupo 3 |
| `reconstruir_sorteos.py` | Cómo se obtuvo el CSV anterior desde los `manifest.csv` del grupo 3 |

El grupo 3 reutiliza los sorteos del grupo 2. El original (`outputs_comparacion/parametros_sorteados.csv`) está en un
disco externo; `parametros_sorteados_grupo2.csv` se reconstruyó desde los manifiestos del grupo 3 y se verificó que
regenera los archivos **idénticos byte a byte** (A_000 y P_p000_inicio, 2026-11-15).

## Uso

```bash
python 12_generar_set_comparacion_21504.py --out /ruta/grupo3 --workers 24            # los 4000 (~4,3 TB)
python 12_generar_set_comparacion_21504.py --out /ruta/prueba --limite 2 --sets A,B   # unos pocos para probar
```

Después: `export CHARTS_SET_DIR=/ruta/grupo3` para el emisor.

## Requisitos

Python 3.12 (probado), numpy 2.5, scipy 1.18 y **fitburst** de CHIME/FRB:
`pip install -r requirements.txt` (instala fitburst desde GitHub en el commit probado).
