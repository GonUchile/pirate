#!/bin/bash
# Resume una corrida de tiempo real: ./resumir.sh CARPETA
D=$1
n=$(grep -a -c "FrbServer: beamset" $D/server.log)
echo "chunks procesados: $n (~$(echo "$n*2.048/60" | bc -l | cut -c1-4) min de datos)"
grep -a "FrbServer: beamset" $D/server.log | grep -a "beams=" | awk -F'beams=' '{split($2,a,"/"); print a[1], a[2]}' | sort -n | awk '{v[NR]=$1; b=$2} END{printf "beams equivalentes: min %s  mediana %s  max %s  (beams reales %s) -> %.2fx tiempo real\n", v[1], v[int(NR/2)], v[NR], b, v[int(NR/2)]/b}'
grep -a "Gbps=" $D/server.log | awk -F'Gbps=' '{split($2,a,","); print a[1]}' | sort -n | awk '{v[NR]=$1} END{print "Gbps entrada: mediana", v[int(NR/2)]}'
awk -F, 'NR>1{u+=$2; n++; m=($3+0>m)?$3+0:m; p+=$4} END{printf "GPU: uso medio %.0f %%, VRAM máx %d MiB, potencia media %.0f W\n", u/n, m, p/n}' $D/gpu.csv
awk 'NR>2 && $4 ~ /^[0-9]+$/ {us+=$13; sy+=$14; id+=$15; n++; f=$4/1048576; if(min==""||f<min)min=f} END{printf "CPU: us %.0f %%, sy %.0f %%, idle %.0f %%; RAM libre mínima %.1f GB\n", us/n, sy/n, id/n, min}' $D/vmstat.log
grep -a -h -iE "Traceback|RuntimeError" $D/*.log | sort | uniq -c | head -5
