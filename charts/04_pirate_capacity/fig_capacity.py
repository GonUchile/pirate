#!/usr/bin/env python3
"""Cómo escalan memoria y capacidad de pirate con el DM máximo, el tamaño de bloque y max_gpu_clag (64 beams, 21 504 canales).

Datos: timings.json (timings.py, 2026-10-21): `pirate_frb show dedisperser` (estado del árbol en GPU/RAM, PCIe a tiempo real)
y `pirate_frb time_dedisperser -H -n 24` (memoria reservada y beams a tiempo real con la GPU al 100 %), RTX 4090.
Rank 13 y 14 no tienen kernel compilado en pirate: solo memoria (sweep_show.json).
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

AQUI = Path(__file__).resolve().parent
R = json.loads((AQUI / "timings.json").read_text())
S = json.loads((AQUI / "sweep_show.json").read_text())
plt.rcParams.update({"font.size": 13, "axes.titlesize": 14, "legend.fontsize": 11})
CG, CR = plt.cm.viridis(0.25), plt.cm.viridis(0.75)


def busca(tipo, rank, nt, clag, ntrees):
    return next(r for r in R if (r["tipo"], r["rank"], r["nt"], r["clag"], r["ntrees"]) == (tipo, rank, nt, clag, ntrees))


fig, axs = plt.subplots(2, 2, figsize=(16, 9))

# (a) memoria del árbol según DM máximo
r14 = next(s for s in S if s["rank"] == 14 and s["nt"] == 2048)
casos = [("552\nr14", dict(ring_gpu=r14["gpu"], ring_host=r14["host"], beams_rt=None, h2g_rt=r14["h2g"])),
         ("1105\nr15 (prueba)", busca("dm_nt", 15, 2048, 5, 1)),
         ("2209\nr15 +1", busca("arboles", 15, 2048, 5, 2)),
         ("2209\nr16", busca("dm_nt", 16, 2048, 5, 1)),
         ("4418\nr15 +2", busca("arboles", 15, 2048, 5, 3)),
         ("4418\nr16 +1", busca("arboles", 16, 2048, 1, 2))]
x = np.arange(len(casos))
XL = "DM máximo · r = rank (árbol de 1 ms) · +n = árboles extra (2 y 4 ms)"
ax = axs[0, 0]
g = np.array([c[1]["ring_gpu"] for c in casos]); h = np.array([c[1]["ring_host"] for c in casos])
ax.bar(x, g, color=CG, label="en GPU")
ax.bar(x, h, bottom=g, color=CR, label="en RAM")
for i, (gi, hi) in enumerate(zip(g, h)):
    ax.text(i, gi + hi + 4, f"{gi + hi:.0f}", ha="center", fontsize=12)
ax.set_xticks(x, [c[0] for c in casos], fontsize=11)
ax.set_xlabel(XL, fontsize=11.5)
ax.set_ylabel("Estado del árbol (GiB)")
ax.set_ylim(0, 290)
ax.set_title("(a) Memoria del árbol según el DM máximo")
ax.legend(loc="upper left")
ax.text(0.27, 0.95, "rank +1 (DM ×2) → memoria ×4\nárbol extra → mucho más barato", transform=ax.transAxes,
        va="top", fontsize=12, bbox=dict(fc="white", ec="0.8"))

# (b) capacidad según DM
ax = axs[0, 1]
for i, c in enumerate(casos):
    bi = c[1]["beams_rt"]
    ax.bar(i, bi or 0, color=plt.cm.viridis(0.45))
    ax.text(i, (bi or 0) + (15 if bi else 90), (f"{bi:.0f}\n" if bi else "sin kernel\ncompilado\n") + f"PCIe {c[1]['h2g_rt']:.1f} GB/s",
            ha="center", va="bottom", fontsize=10.5)
ax.axhline(64, color="k", ls="--", lw=1.5)
ax.text(5.45, 50, "64 beams", ha="right", va="top", fontsize=11)
ax.set_xticks(x, [c[0] for c in casos], fontsize=11)
ax.set_xlabel(XL, fontsize=11.5)
ax.set_ylabel("Beams a tiempo real (GPU al 100 %)")
ax.set_ylim(0, 950)
ax.set_title("(b) Capacidad de una RTX 4090  ·  PCIe RAM→GPU con 64 beams")

# (c) tamaño de bloque
ax = axs[1, 0]
nts = [512, 1024, 2048, 4096, 8192]
rr = [busca("dm_nt", 15, nt, max(1, round(10240 / nt)), 1) for nt in nts]
ax.plot(nts, [r["beams_rt"] for r in rr], "o-", color=CG, lw=2, label="DM 1105 (GPU 13–21 GB, RAM 22–24 GB)")
for r in rr:
    ax.annotate(f"GPU {r['gpu_GB']:.0f} · RAM {r['cpu_GB']:.0f}", (r["nt"], r["beams_rt"]), textcoords="offset points",
                xytext=(0, 10), ha="center", fontsize=10, color="0.25")
r16 = [busca("dm_nt", 16, nt, max(1, round(10240 / nt)), 1) for nt in nts]
ok = [r for r in r16 if "beams_rt" in r]
ax.plot([r["nt"] for r in ok], [r["beams_rt"] for r in ok], "s-", color=CR, lw=2,
        label="DM 2209 (GPU 20–25 GB, RAM ~120 GB)")
ax.plot([8192], [0], "x", color=CR, ms=12, mew=3, clip_on=False)
ax.text(8192, 40, "DM 2209: no cabe\n(GPU 35 GB)", ha="center", fontsize=10, color="0.25")
ax.set_xscale("log", base=2)
ax.set_xticks(nts, [f"{n}\n({n / 1000:g} s)" for n in nts])
ax.set_xlabel("Tamaño de bloque (muestras de 1 ms) = espera mínima antes de procesar")
ax.set_ylabel("Beams a tiempo real")
ax.set_ylim(0, 1000)
ax.set_title("(c) Tamaño de bloque: cambia poco la capacidad")
ax.legend(loc="center", bbox_to_anchor=(0.5, 0.45))

# (d) max_gpu_clag
ax = axs[1, 1]
cl = [r for r in R if r["tipo"] == "clag"]
ok = [r for r in cl if "beams_rt" in r]
ax.plot([r["clag"] for r in ok], [r["beams_rt"] for r in ok], "o-", color=plt.cm.viridis(0.45), lw=2)
for r in ok:
    if r["clag"] >= 5:
        ax.annotate(f"GPU {r['gpu_GB']:.0f}\nRAM {r['cpu_GB']:.0f}", (r["clag"], r["beams_rt"]), textcoords="offset points",
                    xytext=(-26, 4), ha="center", fontsize=10, color="0.25")
ax.text(7.0, 620, "\n".join(f"clag {r['clag']}: GPU {r['gpu_GB']:.0f} · RAM {r['cpu_GB']:.0f} GB" for r in ok if r["clag"] < 5),
        fontsize=10.5, color="0.25", va="top", bbox=dict(fc="white", ec="0.8"))
for r in cl:
    if "beams_rt" not in r:
        ax.plot([r["clag"]], [0], "x", color="k", ms=12, mew=3, clip_on=False)
        ax.text(r["clag"], 60, f"no cabe\nGPU {r['gpu_GB']:.0f} GB", ha="center", fontsize=10, color="0.25")
ax.axvline(5, color="0.6", ls=":")
ax.text(5.2, 300, "prueba", fontsize=11, color="0.4")
ax.set_xlabel("max_gpu_clag (bloques de 2,048 s que el árbol guarda en la GPU), DM 1105")
ax.set_ylabel("Beams a tiempo real")
ax.set_xlim(0, 17)
ax.set_ylim(0, 1350)
ax.set_xticks([1, 2, 3, 5, 8, 12, 16])
ax.set_title("(d) Más estado en la GPU → menos PCIe → más beams")

fig.suptitle("Recursos de pirate: 64 beams × 21 504 canales, RTX 4090 (medido con time_dedisperser)", fontsize=17)
fig.text(0.5, 0.01, "Memoria GPU y RAM en GB = lo que reserva pirate para la dedispersión (no incluye el ring buffer de "
         "entrada).\nA máxima velocidad, en todas las configuraciones la copia RAM→GPU va a ~20–23 GB/s: el PCIe es lo que "
         "limita la capacidad cuando el árbol usa la RAM.\nRank 17 o más no está permitido en pirate (máximo 16); para DM "
         "más alto se agregan árboles con muestras de 2 y 4 ms.", ha="center", va="bottom", fontsize=11.5, color="0.3")
fig.tight_layout(rect=(0, 0.085, 1, 0.96))
out = AQUI / "pirate_capacity.png"
fig.savefig(out, dpi=150)
print(out)
