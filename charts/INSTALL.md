# Instalación

Esta carpeta (`charts/`) vive dentro del fork de pirate de CHARTS, https://github.com/charts-experiment/pirate:
la carpeta de arriba **es** pirate. Se clona una vez y se compila ahí mismo.

Qué instalar depende de qué pieza corre en cada PC:

| PC | Corre | Necesita |
|---|---|---|
| **PC de búsqueda** (con GPU NVIDIA) | `servidor/` (pirate + grouper), `evaluacion/` | el fork compilado, ksgpu y su entorno conda |
| **PC que produce los datos** (puede ser el mismo) | `emisor/` | Python ≥ 3.9 con numpy y PyYAML, el clon del fork (solo usa `charts/`), los datos |
| **Para generar los datos de prueba** (una vez) | `simulador/` | Python 3.12, numpy, scipy y fitburst (`simulador/requirements.txt`) |
| **Para regenerar la RFI del sitio** (opcional) | `rfi_sitio/` 00–07 | numpy, scipy, h5py, matplotlib y el baseband del sitio |

## 1. Repositorios y versiones probadas

| Repositorio | Dirección | Versión probada | Para qué |
|---|---|---|---|
| **pirate (fork de CHARTS)** | https://github.com/charts-experiment/pirate | basado en kmsmith137/pirate commit `255edd26c1c77e3f154fc5accd8014f073582b66` (v1.5.0, 2026-09-07) | Recepción por red y dedispersión en árbol en GPU. Este repositorio; `charts/` es lo de CHARTS y el resto es pirate sin cambios |
| **ksgpu** | https://github.com/kmsmith137/ksgpu | commit `5c81be2dea54226587f80eb206a35ca92c7160c5` (v1.5.0, 2026-09-07) | Librería de GPU que usa pirate (dependencia obligatoria) |
| **fitburst** (CHIME/FRB) | https://github.com/CHIMEFRB/fitburst | commit `3c76da8f9e3ec7bc21951ce1b4a26a0255096b69` | Forma de pulso del simulador (`simulador/`) |
| Kotekan (opcional) | git@github.com:charts-experiment/kotekan.git | commit `1917da5b` (2021.10-426) | Referencia para la adaptación del emisor (`docs/PROTOCOLO_pirate_y_adaptacion_Kotekan.md`); no hace falta para correr la prueba |

El pirate original es https://github.com/kmsmith137/pirate. Si el fork se actualiza con versiones nuevas de
kmsmith137, puede funcionar, pero el protocolo de red, las configs y los kernels compilados se probaron con 1.5.0.

## 2. PC de búsqueda: pirate

### Requisitos del sistema (no los instala conda)

| | Probado con |
|---|---|
| GPU NVIDIA | RTX 4090, 24 GB (arquitectura sm_89) |
| Driver | 590.48.01 |
| CUDA toolkit del sistema (`nvcc`) | 13.1 |
| Compilador del sistema | gcc / g++ 13.3 (Ubuntu 24.04) |
| conda (miniconda o miniforge) | cualquiera reciente |
| RAM | ≥ 150 GB con `configs/prueba_64b.yml` (64 GB con la config optimizada) |
| Disco | ~10 GB para el entorno y la compilación |

### Instalación automática

```bash
git clone https://github.com/charts-experiment/pirate.git && cd pirate
charts/instalar_pirate.sh /ruta/del/entorno [ARQUITECTURA]
# ejemplo: charts/instalar_pirate.sh ~/entornos/pirate 89      (89 = RTX 4090; 80 = A100; 90 = H100)
```

El script **compila el fork en el que está** (no descarga pirate), clona ksgpu al lado del fork en el commit
probado, crea el entorno conda y compila los dos. Tarda ~5–10 min.

### Instalación manual (lo mismo que hace el script)

```bash
mkdir -p ~/fuentes && cd ~/fuentes
git clone https://github.com/charts-experiment/pirate.git       # el fork: pirate + charts/
git clone https://github.com/kmsmith137/ksgpu  && git -C ksgpu  checkout 5c81be2dea54226587f80eb206a35ca92c7160c5
(cd pirate && git submodule update --init --recursive)          # asdf-cxx

# compilar solo para la GPU que se tiene (acelera mucho la compilación)
echo 'NVCC_ARCH = -gencode arch=compute_89,code=sm_89' > ksgpu/config.mk
echo 'NVCC_ARCH = -gencode arch=compute_89,code=sm_89' > pirate/config.mk

conda env create -p ~/entornos/pirate -f pirate/environment_dev.yml
conda activate ~/entornos/pirate
conda install -y -c conda-forge clang-format                    # lo usa el generador de kernels y no viene en el yml

python -m pip install pipmake                                   # usar SIEMPRE "python -m pip" (ver problemas abajo)
(cd ksgpu  && python -m pip install --no-build-isolation -v -e .)
(cd pirate && make asdf-cxx/include/asdf/config.hxx)            # evita una carrera de make -j
(cd pirate && python -m pip install --no-build-isolation -v -e .)
```

### Verificar

```bash
conda activate ~/entornos/pirate
pirate_frb --help
pirate_frb show dedisperser charts/configs/charts_64b_dm1000.yml       # memoria del árbol: host_zones / gpu_zones
pirate_frb time_dedisperser -H -n 24 charts/configs/charts_64b_dm1000.yml   # RTX 4090: ~764 beams a tiempo real
```

### Problemas que aparecieron y cómo se resolvieron

| Síntoma | Causa | Solución |
|---|---|---|
| `pip` instala en Python 2.7 | `~/.local/bin/pip` gana en el PATH | usar `python -m pip` dentro del entorno |
| El generador de kernels falla por `clang-format` | `environment_dev.yml` no lo trae | `conda install -c conda-forge clang-format` |
| `asdf/config.hxx: No such file` con `make -j` | carrera en la compilación | `make asdf-cxx/include/asdf/config.hxx` antes |
| `Kernel not found in registry` al correr | la combinación pedida (rank, boxcar, subbandas) no está entre los kernels compilados | usar las configs del repositorio, o agregar la fila en `makefile_helper.py` de pirate y recompilar |
| `mmap(... MAP_HUGETLB) failed` en `time_dedisperser` | sin hugepages configuradas | agregar `-H` (o configurar `vm.nr_hugepages`) |
| `CUDA out of memory` al arrancar el servidor | pool de GPU de pirate + grouper + emisor > memoria de la GPU | bajar `servidor.memoria_gpu_gib` o `max_gpu_clag` en el YAML |

Para actualizar: `git pull` en el fork (y en ksgpu si cambia) y `make -j` en cada uno, con el entorno activo (la instalación es
editable; no hace falta repetir el pip).

## 3. PC que produce los datos: el emisor

```bash
git clone https://github.com/charts-experiment/pirate.git && cd pirate/charts
python3 -m pip install -r emisor/requirements.txt              # numpy y PyYAML
# opcional, para cuantizar o inyectar RFI en GPU (gpu: true en el YAML): cupy con la versión de CUDA del PC
```

No necesita compilar pirate ni GPU: solo usa la carpeta `charts/`. Si corre en el mismo PC de búsqueda, se puede usar el entorno de pirate (ya tiene numpy,
PyYAML y cupy).

## 4. Datos (no van en el repositorio)

| Dato | Tamaño | De dónde sale | Variable |
|---|---|---|---|
| Grupo 3: 4000 .fil de 21 504 canales (A, B, negativos, cortados) con `manifest.csv` | ~4,3 TB | `simulador/12_generar_set_comparacion_21504.py` (ver sección 4b) | `CHARTS_SET_DIR` |
| Modelo de RFI del sitio `modelo_rfi_sitio.npz` | 177 MB | `rfi_sitio/01_espectro_dinamico.py` + `02_modelo_rfi.py` desde el baseband | `CHARTS_RFI_MODELO` |
| Baseband del sitio (2026-09-02) | 25 GB | captura del X-engine de CHARTS | `CHARTS_BASEBAND_DIR` |

Para una prueba rápida sin el set completo, `--beams-con-datos 1 --limite 2` usa solo 8 archivos.

## 4b. Generar los datos de prueba

```bash
python3.12 -m venv ~/entornos/simulador && source ~/entornos/simulador/bin/activate
python -m pip install -r simulador/requirements.txt
cd simulador
python 12_generar_set_comparacion_21504.py --out /ruta/grupo3 --workers 24          # 4000 archivos, ~4,3 TB
python 12_generar_set_comparacion_21504.py --out /ruta/prueba --limite 2             # 8 archivos para probar
```

Es determinista: genera los mismos archivos que se usaron en las pruebas (verificado byte a byte). Ver `simulador/README.md`.

## 5. Primera corrida

```bash
cd pirate/charts                                  # todo se corre desde charts/
conda activate ~/entornos/pirate                  # pirate se autodetecta (o export CHARTS_PIRATE_ENV=~/entornos/pirate)
export CHARTS_SET_DIR=/ruta/al/grupo3
python servidor/lanzador.py local --config configs/prueba_64b.yml \
    --salida resultados/primera --limite 2 --procesos 16
cat resultados/primera/resultados.md
```

Si falta pirate o la carpeta de datos, el lanzador lo dice antes de arrancar. Con 8 archivos y 64 beams tarda ~3–6 min. Se espera completitud de 100 % en los FRB con DM ≤ 1000 de esos archivos.
