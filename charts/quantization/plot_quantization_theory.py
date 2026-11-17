#!/usr/bin/env python3
"""Figura de quantization_21504.csv con la eficiencia teórica de cada variante (quantization_theory.py)."""
import csv
from pathlib import Path

HERE = Path(__file__).resolve().parent
(HERE / "plots_21504").mkdir(exist_ok=True)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

filas = list(csv.DictReader(open(HERE / "plots_21504" / "quantization_21504.csv")))
noms = ["8 bits", "4b 2niv fija", "4b 3niv fija", "4b 1niv fija", "4b pirate"]
teo = {"8 bits": 0.9974, "4b 2niv fija": 0.9822, "4b 3niv fija": 0.9903, "4b 1niv fija": 0.9535, "4b pirate": 0.9822}
fig, ax = plt.subplots(figsize=(11, 6))
for k, nom in enumerate(noms):
    for j, s in enumerate(("A", "B")):
        sub = [r for r in filas if r["set"] == s and float(r["float"]) > 4]
        rel = np.array([float(r[nom]) / float(r["float"]) for r in sub])
        snr = np.array([float(r["float"]) for r in sub])
        xs = k + (j - 0.5) * 0.28 + np.random.default_rng(k + 10 * j).uniform(-0.05, 0.05, rel.size)
        sc = ax.scatter(xs, 100 * (rel - 1), c=np.log10(snr), cmap="viridis", vmin=0.6, vmax=2.2,
                        marker="o" if s == "A" else "s", s=34, edgecolor="k", lw=0.3,
                        label=f"set {s}" if k == 0 else None)
    ax.hlines(100 * (teo[nom] - 1), k - 0.35, k + 0.35, color="C3", lw=2.5,
              label="teoría (señal débil)" if k == 0 else None)
ax.axhline(0, color="k", lw=0.8)
ax.set_xticks(range(len(noms)), ["8 bits\n(el .fil)", "4 bits\n2 niveles/σ", "4 bits\n3 niveles/σ",
                                  "4 bits\n1 nivel/σ", "4 bits pirate\nescala/offset c/256 muestras"])
ax.set_ylabel("S/N respecto al dato sin cuantizar (%)")
ax.set_title("Cuantización: misma realización de ruido y señal (prueba del grupo 3, 21 504 canales)\n"
             "color = S/N del burst; los de S/N alto miden la pérdida con más precisión")
cb = fig.colorbar(sc, ax=ax)
cb.set_label("log10 S/N (sin cuantizar)")
ax.set_ylim(-24, 7)
ax.annotate("A_007: scattering 25,6 ms;\nel offset por bloque de 256 ms\nresta parte del pulso",
            xy=(4 - 0.14, -21.7), xytext=(2.6, -20), fontsize=8, arrowprops=dict(arrowstyle="->", lw=0.6))
ax.legend(loc="lower left", fontsize=9)
fig.savefig(HERE / "plots_21504" / "quantization_21504_teoria.png", dpi=110, bbox_inches="tight")
print(HERE / "plots_21504" / "quantization_21504_teoria.png")
