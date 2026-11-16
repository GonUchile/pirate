#!/bin/bash
# Compila ESTE fork de pirate (la carpeta de arriba de charts/) y su dependencia ksgpu, con su entorno conda.
# Uso: charts/instalar_pirate.sh CARPETA_ENTORNO [ARQUITECTURA_GPU] [CARPETA_KSGPU]
#   ARQUITECTURA_GPU: 89 (RTX 4090, por defecto), 86 (RTX 30xx/A40), 80 (A100), 90 (H100)
#   CARPETA_KSGPU: dónde clonar ksgpu (por defecto, al lado del fork)
# Requiere en el sistema: conda, git, nvcc (CUDA toolkit) y gcc/g++. Ver charts/INSTALL.md.
set -euo pipefail
ENTORNO=${1:?carpeta del entorno conda a crear}
ARCH=${2:-89}
PIRATE=$(cd "$(dirname "$0")/.." && pwd)                 # el fork: no se descarga, se compila esta carpeta
KSGPU=${3:-$(dirname "$PIRATE")/ksgpu}
KSGPU_COMMIT=5c81be2dea54226587f80eb206a35ca92c7160c5   # la versión con la que se probó (kmsmith137/ksgpu)
PIRATE_PROBADO=255edd26c1c77e3f154fc5accd8014f073582b66 # pirate 1.5.0 de kmsmith137, base del fork

for c in conda git nvcc g++; do command -v $c > /dev/null || { echo "falta '$c' en el sistema (ver charts/INSTALL.md)"; exit 1; }; done
[ -f "$PIRATE/pirate_frb/__init__.py" ] || { echo "no encuentro pirate en $PIRATE: correr desde el clon del fork"; exit 1; }
git -C "$PIRATE" merge-base --is-ancestor $PIRATE_PROBADO HEAD 2>/dev/null \
    || echo "AVISO: este pirate no contiene el commit probado ${PIRATE_PROBADO:0:8}; puede no ser compatible"

[ -d "$KSGPU" ] || git clone https://github.com/kmsmith137/ksgpu "$KSGPU"
git -C "$KSGPU" checkout $KSGPU_COMMIT
git -C "$PIRATE" submodule update --init --recursive       # asdf-cxx

echo "NVCC_ARCH = -gencode arch=compute_${ARCH},code=sm_${ARCH}" > "$KSGPU/config.mk"
echo "NVCC_ARCH = -gencode arch=compute_${ARCH},code=sm_${ARCH}" > "$PIRATE/config.mk"

source "$(conda info --base)/etc/profile.d/conda.sh"
[ -d "$ENTORNO" ] || conda env create -p "$ENTORNO" -f "$PIRATE/environment_dev.yml"
conda activate "$ENTORNO"
conda install -y -c conda-forge clang-format

python -m pip install pipmake
(cd "$KSGPU" && python -m pip install --no-build-isolation -v -e .)
(cd "$PIRATE" && make asdf-cxx/include/asdf/config.hxx)
(cd "$PIRATE" && python -m pip install --no-build-isolation -v -e .)

pirate_frb --help > /dev/null && echo "pirate (fork, $(git -C "$PIRATE" rev-parse --short HEAD)) instalado en $ENTORNO (sm_$ARCH)"
echo "Siguiente: conda activate $ENTORNO   (ver charts/INSTALL.md, sección 5)"
