#!/usr/bin/env python3
"""Modelo de RFI del sitio a partir del espectro dinámico (01_espectro_dinamico.py).

Para cada entrada a (0–7):
  piso_a(f) = B_a(c) · S_a(k)   B: nivel del canal grueso c sin RFI (mediana corrida ±25 canales gruesos, iterando 4 veces sin
                                   los canales > 1,1 B; suavizado ±5)
                                S: scalloping de la PFB dentro del canal grueso (mediana sobre canales gruesos)
  r_a(t,f) = P_a(t,f) / piso_a(f) − 1      potencia de RFI en unidades de la potencia del sistema
Beam: r(t,f) = promedio de r_a sobre las entradas. Para un beam fuera del lóbulo principal (lóbulos laterales
con fases aleatorias) la razón RFI/sistema es en promedio la misma que en una antena: atenuación 0 dB.
  r_p(f)   = mediana temporal de r(t,f)                          → RFI persistente (líneas)
  σ_r(f)   = MAD de r(t,f) en el tiempo                          → ruido de la medición
  e(t,f)   = r − r_p si (r − r_p) > 5 σ_r, más los vecinos conectados > 3 σ_r; si no 0   → RFI rápida
Salida: modelo_rfi_sitio.npz (frecuencias, r_p, σ_r, e en float16, pisos), resumen_rfi.json, figuras.
Uso: python 02_modelo_rfi.py [--entradas 0,1,...]
"""
import argparse, json
from pathlib import Path
import numpy as np
from scipy import ndimage
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

AQUI = Path(__file__).resolve().parent
PSYS_CUENTAS = 4 * np.sqrt(9.375)      # potencia del sistema en cuentas del .fil (σ = 4 cuentas = P_sys/√(Δν τ))

ap = argparse.ArgumentParser()
ap.add_argument("--entradas", default="0,1,2,3,4,5,6,7")
ap.add_argument("--umbral", type=float, default=5.0)
ap.add_argument("--umbral-vecinos", type=float, default=3.0)
args = ap.parse_args()
ent = [int(x) for x in args.entradas.split(",")]

D = np.load(AQUI / "espectro_dinamico.npy", mmap_mode="r")
f = np.load(AQUI / "frecuencias_mhz.npy")
NT, NF = D.shape[1], D.shape[2]


def corre(x, w, fn):
    """Filtro de ventana ±w con borde reflejado."""
    xp = np.pad(x, w, mode="reflect")
    v = np.lib.stride_tricks.sliding_window_view(xp, 2 * w + 1)
    return fn(v, axis=-1)


pisos, ra_med, r = {}, {}, np.zeros((NT, NF), np.float32)
for a in ent:
    P = np.asarray(D[a])
    m = np.median(P, 0).reshape(672, 32)
    S = np.median(m / np.median(m, 1, keepdims=True), 0)
    L = np.median(m / S, 1)
    B = corre(L, 25, np.median)
    for _ in range(4):                       # piso iterativo: mediana corrida sólo con canales gruesos sin RFI
        B = corre(np.where(L < 1.1 * B, L, np.nan), 25, np.nanmedian)
        B = np.where(np.isfinite(B), B, np.nanmin(B))
    B = corre(B, 5, np.mean)
    piso = (B[:, None] * S[None, :]).ravel().astype(np.float32)
    ra = P / piso - 1
    pisos[a] = piso; ra_med[a] = np.median(ra, 0)
    r += ra / len(ent)
    print(f"entrada {a}: piso listo, r_p mediana {np.median(ra_med[a]):+.3f}", flush=True)

r_p = np.median(r, 0)
sig = 1.4826 * np.median(np.abs(r - r_p), 0)
ex = r - r_p
fuerte = ex > args.umbral * sig
debil = ex > args.umbral_vecinos * sig
lab, nlab = ndimage.label(debil)
keep = np.zeros(nlab + 1, bool); keep[np.unique(lab[fuerte])] = True; keep[0] = False
mask = keep[lab]
e = np.where(mask, ex, 0).astype(np.float16)

# --- estadísticas
lab2, nev = ndimage.label(mask)
obj = ndimage.find_objects(lab2)
dur = np.array([s[0].stop - s[0].start for s in obj]); anc = np.array([s[1].stop - s[1].start for s in obj])
pico = ndimage.maximum(ex, lab2, np.arange(1, nev + 1))
dnu = 0.009375
frac_banda_ms = mask.mean(1)
res = dict(
    entradas=ent, duracion_s=NT / 1000,
    sigma_sobre_media_esperado=float(1 / np.sqrt(9.375)), sigma_r_mediana=float(np.median(sig)),
    canales_rp_mayor_0p1=float((r_p > 0.1).mean()), canales_rp_mayor_1=float((r_p > 1).mean()),
    canales_rp_mayor_10dB=float((r_p > 9).mean()),
    pixeles_rapidos=float(mask.mean()), eventos_rapidos=int(nev), eventos_por_s=float(nev / (NT / 1000)),
    duracion_ms_p50_p90_p99=[float(np.percentile(dur, q)) for q in (50, 90, 99)],
    ancho_mhz_p50_p90_p99=[float(np.percentile(anc, q) * dnu) for q in (50, 90, 99)],
    pico_r_p50_p90_p99=[float(np.percentile(pico, q)) for q in (50, 90, 99)],
    ms_con_mas_10pct_banda=int((frac_banda_ms > 0.1).sum()),
    psys_cuentas=float(PSYS_CUENTAS),
    sat_255_si_lineas_visibles=float(((32 + PSYS_CUENTAS * (r_p + e.astype(np.float32))) > 255).mean()),
)
# coincidencia entre entradas de la RFI rápida (¿externa o local a una antena?)
co = {}
for a in ent:
    ea = (np.asarray(D[a]) / pisos[a] - 1) - ra_med[a]
    sa = 1.4826 * np.median(np.abs(ea), 0)
    ma = ea > args.umbral * sa
    otras = np.zeros_like(ma, dtype=np.int8)
    for b in ent:
        if b == a: continue
        eb = (np.asarray(D[b]) / pisos[b] - 1) - ra_med[b]
        otras += (eb > args.umbral * 1.4826 * np.median(np.abs(eb), 0)).astype(np.int8)
    co[a] = dict(frac_pixeles=float(ma.mean()), frac_tambien_en_otra=float((otras[ma] > 0).mean()) if ma.any() else None,
                 frac_en_4_o_mas=float((otras[ma] >= 3).mean()) if ma.any() else None)
    print("coincidencia", a, co[a], flush=True)
res["coincidencia_entradas"] = co
json.dump(res, open(AQUI / "resumen_rfi.json", "w"), indent=1)
np.savez(AQUI / "modelo_rfi_sitio.npz", frecuencias_mhz=f, r_p=r_p.astype(np.float32), sigma_r=sig.astype(np.float32),
         e=e, entradas=np.array(ent), dt_ms=1.0, psys_cuentas=PSYS_CUENTAS,
         r_p_por_entrada=np.array([ra_med[a] for a in ent], np.float32))
print(json.dumps({k: v for k, v in res.items() if k != "coincidencia_entradas"}, indent=1))

# --- figuras
fig, ax = plt.subplots(2, 2, figsize=(16, 9))
for i, a in enumerate(ent):
    ax[0, 0].plot(f, 10 * np.log10(1 + np.clip(ra_med[a], -0.9, None)), lw=0.5, alpha=0.6, color=plt.cm.viridis(i / len(ent)))
ax[0, 0].plot(f, 10 * np.log10(1 + np.clip(r_p, -0.9, None)), lw=0.8, color="k", label="promedio (beam)")
ax[0, 0].set(xlabel="MHz", ylabel="10 log10(1 + r_p)  (dB sobre el piso)", title="RFI persistente: mediana en 4,1 s"); ax[0, 0].legend()
ds = r[:, :].reshape(NT, 672, 32).mean(2)
im = ax[0, 1].imshow(10 * np.log10(1 + np.clip(ds, -0.5, None)).T, aspect="auto", origin="lower", cmap="viridis",
                     extent=[0, NT / 1000, f[0], f[-1]], vmin=-0.5, vmax=6)
fig.colorbar(im, ax=ax[0, 1], label="dB sobre el piso"); ax[0, 1].set(xlabel="tiempo (s)", ylabel="MHz", title="Beam (promedio 8 entradas), canales gruesos")
sel = (f > 405) & (f < 432)
im = ax[1, 0].imshow((ex[:1000, sel] / sig[sel]).T, aspect="auto", origin="lower", cmap="viridis",
                     extent=[0, 1, f[sel][0], f[sel][-1]], vmin=-3, vmax=10)
fig.colorbar(im, ax=ax[1, 0], label="(r − r_p)/σ_r"); ax[1, 0].set(xlabel="tiempo (s)", ylabel="MHz", title="RFI rápida, 405–432 MHz, primer segundo, 21 504 canales")
z = (ex / sig)[:, ::5].ravel()
h, b = np.histogram(z, bins=np.linspace(-6, 30, 145))
ax[1, 1].semilogy(b[:-1], h / h.sum(), drawstyle="steps-post", label="medido")
from scipy.stats import norm
ax[1, 1].semilogy(b[:-1], norm.pdf(b[:-1] + 0.125) * 0.25, "k--", label="gaussiano")
ax[1, 1].axvline(args.umbral, color="0.5", ls=":"); ax[1, 1].set(xlabel="(r − r_p)/σ_r", ylabel="fracción", title="Cola de la RFI rápida", ylim=(1e-9, 0.2)); ax[1, 1].legend()
fig.tight_layout(); fig.savefig(AQUI / "modelo_rfi_sitio.png", dpi=110)
