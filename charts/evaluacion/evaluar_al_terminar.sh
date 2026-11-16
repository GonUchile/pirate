#!/bin/bash
# Espera a que termine una corrida (aparece "apagando" en lanzador.log) y la evalúa.
# Uso: evaluacion/evaluar_al_terminar.sh CARPETA_DE_LA_CORRIDA
AQUI=$(cd "$(dirname "$0")/.." && pwd); D=${1:?carpeta de la corrida}
until grep -qE "apagando" "$D/lanzador.log" 2>/dev/null; do sleep 30; done
sleep 90
PY=python; [ -n "$CHARTS_PIRATE_ENV" ] && PY=$CHARTS_PIRATE_ENV/bin/python   # o con el entorno de pirate activado
"$PY" "$AQUI/evaluacion/evaluar_prueba_64b.py" "$D" > "$D/evaluacion.log" 2>&1
echo "evaluada: $D/resultados.md"
