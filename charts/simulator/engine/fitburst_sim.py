#!/usr/bin/env python3
"""
Motor de simulación de FRBs sintéticos con Fitburst, según Fonseca et al.
(2024) "Modeling the Morphology of Fast Radio Bursts and Radio Pulsars with
fitburst" (ApJS 271:49, arXiv:2311.05829).

No se ejecuta directamente para generar un catálogo -- para eso ver
generar_simulaciones.py (que llama a generar_batch_desde_catalogo() de acá
con los parámetros de banda/instrumento y la selección de FRBs que
corresponda). Este archivo es el motor físico: dado DM/ancho/scattering/
espectro/flujo de UN FRB, produce el .npz con la señal + ruido.

Requiere el paquete `fitburst` (pip install -e, ver
https://github.com/CHIMEFRB/fitburst) instalado y disponible en el
entorno -- sin necesidad de manipular sys.path si está instalado
normalmente (pip install / pip install -e).

Física implementada:
- amplitude=[0.0] (unitario) y escalado por flujo real (Sección 2.2)
- dm_index=-2.0 (ley teórica plasma frío, Ecuación 5)
- scattering_index=-4.0 (espectro de Kolmogorov, Ecuación 4)
- ref_freq=400.1953125 MHz (borde inferior de la banda observada por
  CHIME, 400-800 MHz) -- ver deteccion.py, misma referencia en todo el
  pipeline
- La constante de dispersión de fitburst (4149.377593360996) entrega el
  delay DIRECTAMENTE EN SEGUNDOS (no en ms) -- ver fitburst/backend/general.yaml
- Como ref_freq es el borde INFERIOR de la banda, es la frecuencia que
  LLEGA AL FINAL (las frecuencias bajas se retrasan más) -- por eso
  arrival_time (definido en ref_freq) se ubica cerca del final de la
  ventana temporal, no al principio.
- Ruido con ecuación del radiómetro: sigma = SEFD / sqrt(n_pol * Δν * τ)
"""
import time as pytime
from pathlib import Path

import numpy as np
import fitburst as fb

# ============================================================================
# PARÁMETROS FÍSICOS FIJOS -- editar ACÁ, no en generar_frb_fitburst_paper().
# Los usa todo el pipeline (01_/02_/03_/04_/05_ y engine/charts_sefd.py) vía
# import desde este módulo, así que un cambio acá basta.
#
#   DM_CONST_SECONDS  Constante de dispersión del plasma frío (Ecuación 5,
#                      Fonseca et al. 2024). Delay en SEGUNDOS (no ms) --
#                      ver fitburst/backend/general.yaml. No tocar salvo que
#                      cambie la propia constante de fitburst.
#   DM_INDEX           Índice de la ley t(ν) ∝ DM·ν^dm_index. -2.0 es la ley
#                      EXACTA de plasma frío no magnetizado -- no es un
#                      parámetro libre por FRB. Se sobreescribe SOLO en
#                      02_generar_negativos.py (a propósito, para generar
#                      señales de control con índice inválido).
#   SCATTERING_INDEX   Índice de la ley de scattering usada por fitburst:
#                      τ(ν) = scattering_timescale · (ν/ref_freq)^scattering_index
#                      (ver fitburst/analysis/model.py, compute_profile()).
#                      -4.0 = espectro de Kolmogorov (Ecuación 4). Tampoco es
#                      un parámetro libre por FRB.
#   REF_FREQ            Frecuencia en la que fitburst define TODOS sus
#                      parámetros dependientes de frecuencia (arrival_time,
#                      scattering_timescale) -- ver fórmula de arriba, con
#                      ν=ref_freq da scattering_timescale tal cual. Fija en
#                      el borde inferior de la banda CHIME (400-800 MHz,
#                      Fonseca et al. 2024) -- OJO: no es la banda CHARTS
#                      simulada (300-501.6 MHz, ver CHARTS_FREQ_LO/HI en
#                      01_/02_), es solo el ancla de definición de fitburst.
# ============================================================================
DM_CONST_SECONDS = 4149.377593360996  # s * MHz^2 / (pc cm^-3) -- fitburst/backend/general.yaml
DM_INDEX = -2.0          # FIJO: ley de plasma frío
SCATTERING_INDEX = -4.0  # FIJO: espectro de Kolmogorov
REF_FREQ = 400.1953125   # FIJO: Fonseca et al. (2024), borde inferior banda CHIME

# SEFD por defecto si no se especifica uno (deriva de las specs de CHIME:
# T_sys=50K, área física 8000 m², eficiencia de apertura supuesta 0.6 --
# ver charts_sefd.py para el equivalente real de CHARTS).
K_BOLTZMANN = 1.380649e-23      # J/K
T_SYS_K = 50.0
A_PHYSICAL_M2 = 8000.0
APERTURE_EFFICIENCY = 0.6
A_EFFECTIVE_M2 = A_PHYSICAL_M2 * APERTURE_EFFICIENCY
SEFD_JY_DEFAULT = 2.0 * K_BOLTZMANN * T_SYS_K / A_EFFECTIVE_M2 / 1e-26  # 1 Jy = 1e-26 W/m^2/Hz


def dispersion_delay(dm, freq_mhz, ref_freq=REF_FREQ, dm_index=DM_INDEX):
    """Delay (segundos) de freq_mhz relativo a ref_freq. Fonseca Ecuación 5."""
    return dm * DM_CONST_SECONDS * (np.asarray(freq_mhz) ** dm_index - ref_freq ** dm_index)


def generar_frb_fitburst_paper(
    frb_name, dm, width_fitb_s, scat_s_600mhz, sp_idx, sp_run, flux_jy,
    snr_target=None,
    freq_lo=REF_FREQ, freq_hi=800.0, num_freq=1024, res_time_ms=0.5,
    sefd_jy=None, n_pol=2.0, dm_index=DM_INDEX,
    params_aproximados=None, output_dir=None, sin_ruido=False,
):
    """
    Genera un FRB sintético usando Fitburst.

    Parámetros
    ----------
    frb_name : str
    dm : float
        Medida de Dispersión [pc/cm³]
    width_fitb_s : float
        Ancho Gaussiano del pulso [s]
    scat_s_600mhz : float
        Scattering timescale a 600 MHz [s]
    sp_idx, sp_run : float
        Índice espectral y corrimiento espectral (Ecuación 2)
    flux_jy : float
        Flujo pico banda-promediado [Jy]
    snr_target : float, opcional
        Si se especifica, reescala la señal para alcanzar este SNR pico
    freq_lo, freq_hi : float
        Banda de frecuencia simulada [MHz]
    num_freq : int
        Número de canales
    res_time_ms : float
        Resolución temporal [ms]
    sefd_jy : float, opcional
        SEFD del instrumento [Jy]. None -> SEFD_JY_DEFAULT (CHIME aprox.).
        Para CHARTS usar charts_sefd.CHARTS_SEFD_JY.
    n_pol : float
        Polarizaciones sumadas en la ecuación del radiómetro. CHIME: 2.0
        (default). CHARTS: 1.0 (charts_sefd.CHARTS_N_POL) -- una sola
        polarización, Tabla 1 de Cassanelli et al. (2024).
    dm_index : float, opcional
        Índice de la ley de dispersión, t(ν) ∝ DM·ν^dm_index. FIJO en -2.0
        para cualquier FRB real (plasma frío no magnetizado, Ecuación 5 de
        Fonseca et al. 2024) -- no es un parámetro libre de ajuste
        astrofísico. Existe como argumento acá específicamente para poder
        generar señales de control "no-FRB" con un índice físicamente
        inválido (ej. fuera de [1.8, 2.2]): como los 3 detectores
        dedispersan asumiendo el índice estándar -2.0, una señal con índice
        distinto nunca se compensa correctamente en ningún trial de DM que
        prueben -- da una clase negativa genuina para medir especificidad
        (ver comparacion_deteccion_chime/generar_negativos_dm_index.py).
    params_aproximados : list[str], opcional
        Nombres de parámetros que NO son medición real de este FRB (se
        deja constancia en la metadata, no afecta el cómputo).
    output_dir : str o Path, opcional
        Carpeta de salida del .npz. Default: ./simulaciones/ relativo a
        este archivo.
    sin_ruido : bool, opcional
        Si True, no se suma ruido del radiómetro -- data_full queda
        idéntico a data_clean (señal pura). snr_peak/snr_integrated
        quedan en inf (no hay piso de ruido contra el cual medir SNR).
        Incompatible con snr_target (no tiene sentido reescalar a un SNR
        objetivo si no hay ruido de referencia).

    Returns
    -------
    dict con filename, path, shape, snr_peak, snr_integrated, dm, flux_jy.
    """
    print(f"\n{'='*80}\nGENERADOR FRB - FITBURST (Fonseca et al. 2024)\n{'='*80}")
    print(f"FRB: {frb_name} | DM: {dm:.2f} pc/cm³ | flujo: {flux_jy:.2f} Jy | "
          f"sp_idx: {sp_idx:.2f} | sp_run: {sp_run:.2f}")
    if params_aproximados:
        print(f"  ⚠ APROXIMADOS (mediana del catálogo, no medidos para este FRB): "
              f"{', '.join(params_aproximados)}")

    # ---- Banda, sweep de dispersión y ventana temporal ----
    res_freq_mhz = (freq_hi - freq_lo) / (num_freq - 1)
    freqs = np.linspace(freq_lo, freq_hi, num_freq)
    res_time = res_time_ms / 1000.0

    # Frecuencias MENORES que ref_freq tienen delay POSITIVO (llegan
    # después); MAYORES tienen delay NEGATIVO (llegan antes). Válido tanto
    # si ref_freq está en el borde de la banda como si queda dentro de ella
    # (p.ej. al extender la banda observada hacia frecuencias más bajas).
    delay_lo = dispersion_delay(dm, freq_lo, ref_freq=REF_FREQ, dm_index=dm_index)
    delay_hi = dispersion_delay(dm, freq_hi, ref_freq=REF_FREQ, dm_index=dm_index)
    sweep_s = delay_lo - delay_hi  # siempre positivo (freq_hi llega antes que freq_lo)

    margin_start = max(5.0 * width_fitb_s, 0.005)
    margin_end = max(5.0 * max(width_fitb_s, scat_s_600mhz), 0.01)

    # arrival_time se define EN ref_freq. Se ubica de modo que freq_hi (la
    # que llega primero) caiga justo en margin_start:
    #   t_peak(freq_hi) = margin_start  =>  arrival_time = margin_start - delay_hi
    arrival_time = margin_start - delay_hi
    total_duration = margin_start + sweep_s + margin_end
    num_time = int(np.ceil(total_duration / res_time))
    times = np.arange(num_time) * res_time

    print(f"  Banda: {freq_lo:.4f}-{freq_hi:.4f} MHz, {num_freq} canales, "
          f"{res_freq_mhz*1000:.2f} kHz/canal")
    print(f"  Sweep de dispersión: {sweep_s:.4f} s | ventana total: {total_duration:.4f} s "
          f"({num_time} muestras @ {res_time*1000:.3f} ms)")

    # ---- Parámetros Fitburst ----
    # fitburst define scattering_timescale EN ref_freq y escala a cualquier
    # frecuencia como sc_time(ν) = scattering_timescale * (ν/ref_freq)**SCATTERING_INDEX
    # (fitburst/analysis/model.py, compute_profile()). scat_s_600mhz es el
    # valor medido en el catálogo A 600 MHz, así que para recuperar
    # scattering_timescale hay que invertir esa misma fórmula evaluada en
    # ν=600: scat_s_600mhz = scattering_timescale * (600/ref_freq)**SCATTERING_INDEX
    #   => scattering_timescale = scat_s_600mhz * (600/ref_freq)**(-SCATTERING_INDEX)
    # OJO: el exponente es -SCATTERING_INDEX (+4), NO SCATTERING_INDEX (-4) --
    # con el signo de SCATTERING_INDEX directo (bug corregido acá, hasta
    # 2026-09-10) el scattering quedaba subestimado ~25x en toda la banda
    # CHARTS (verificado contra fitburst/analysis/model.py:359). Si tocás
    # esto, actualizá también 02_generar_negativos.py, que hace la
    # conversión inversa para preservar el scattering del positivo de
    # origen y depende del mismo signo.
    scattering_timescale = scat_s_600mhz * (600.0 / REF_FREQ) ** (-SCATTERING_INDEX)
    params = {
        "amplitude": [0.0],
        "arrival_time": [arrival_time],
        "burst_width": [width_fitb_s],
        "dm": [dm],
        "dm_index": [dm_index],
        "ref_freq": [REF_FREQ],
        "scattering_index": [SCATTERING_INDEX],
        "scattering_timescale": [scattering_timescale],
        "spectral_index": [sp_idx],
        "spectral_running": [sp_run],
    }

    # ---- Modelo unitario ----
    modeler = fb.analysis.model.SpectrumModeler(
        freqs, times, num_components=1, is_dedispersed=False, verbose=False,
    )
    modeler.update_parameters(params)
    t0 = pytime.time()
    unit_model = modeler.compute_model().astype(np.float32)
    print(f"  ✓ Modelo generado en {pytime.time()-t0:.2f} s | shape {unit_model.shape} (freq×time)")

    n_nonzero = int((unit_model.max(axis=1) > 0).sum())
    if n_nonzero < num_freq:
        print(f"  ⚠ {num_freq - n_nonzero} canales sin señal detectable -- revisar ventana/arrival_time")

    # ---- Calibrar por flujo real (Sección 2.2 del paper) ----
    # El flujo del catálogo es banda-promedio en la banda REAL de CHIME
    # (400-800 MHz). Si la banda simulada se extiende por debajo de 400 MHz,
    # esos canales extra NO deben entrar en el promedio de calibración.
    chime_mask = (freqs >= 400.0) & (freqs <= 800.0)
    band_avg_profile = unit_model[chime_mask].mean(axis=0) if chime_mask.any() else unit_model.mean(axis=0)
    peak_unit = float(band_avg_profile.max())
    scale = flux_jy / peak_unit if peak_unit > 0 else 1.0
    signal_jy = (unit_model * scale).astype(np.float32)

    # ---- Ruido (ecuación del radiómetro) ----
    if sefd_jy is None:
        sefd_jy = SEFD_JY_DEFAULT
        print(f"  SEFD por defecto (aprox. CHIME): {sefd_jy:.2f} Jy")

    delta_nu_hz = res_freq_mhz * 1e6
    noise_std = sefd_jy / np.sqrt(n_pol * delta_nu_hz * res_time)

    if sin_ruido:
        if snr_target is not None:
            raise ValueError("snr_target no tiene sentido con sin_ruido=True "
                              "(no hay piso de ruido contra el cual reescalar)")
        noise_jy = np.zeros_like(signal_jy)
        data_full = signal_jy.copy()
        snr_peak = float("inf")
        snr_integrated = float("inf")
        print(f"  SIN RUIDO (sefd_jy={sefd_jy:.2f} Jy calculado pero no aplicado) "
              f"-> SNR pico/integrada = inf")
    else:
        rng = np.random.default_rng()
        noise_jy = rng.normal(0, noise_std, signal_jy.shape).astype(np.float32)
        data_full = (signal_jy + noise_jy).astype(np.float32)

        snr_peak = float(signal_jy.max() / noise_std)
        band_avg_signal = signal_jy.mean(axis=0)
        noise_band_avg_std = noise_std / np.sqrt(num_freq)
        snr_integrated = float(band_avg_signal.max() / noise_band_avg_std)
        print(f"  SEFD={sefd_jy:.2f} Jy, n_pol={n_pol:.0f} | SNR pico={snr_peak:.2f}, "
              f"SNR integrada={snr_integrated:.2f}")

        if snr_target is not None:
            factor = snr_target / snr_peak
            signal_jy = (signal_jy * factor).astype(np.float32)
            data_full = (signal_jy + noise_jy).astype(np.float32)
            snr_peak = snr_target
            snr_integrated *= factor
            print(f"  -> Reescalado a SNR objetivo {snr_target:.1f} (factor {factor:.3f})")

    metadata = {
        "bad_chans": [], "freqs_bin0": float(freqs[0]), "is_dedispersed": False,
        "num_freq": num_freq, "num_time": num_time, "times_bin0": 0.0,
        "res_freq": float(res_freq_mhz), "res_time": res_time,
        "dm_const_seconds": DM_CONST_SECONDS, "ref_freq": REF_FREQ,
        "dm_index": dm_index, "scattering_index": SCATTERING_INDEX,
        "sweep_s": sweep_s, "sefd_jy": sefd_jy, "n_pol": n_pol, "sin_ruido": sin_ruido,
    }
    burst_parameters = {
        "amplitude": [0.0], "arrival_time": [arrival_time], "burst_width": [width_fitb_s],
        "dm": [dm], "dm_index": [dm_index], "ref_freq": [REF_FREQ],
        "scattering_index": [SCATTERING_INDEX], "scattering_timescale": [scattering_timescale],
        "spectral_index": [sp_idx], "spectral_running": [sp_run],
        "is_frb": abs(dm_index) == 2.0, "flux_jy_catalog": flux_jy,
        "snr_peak": snr_peak, "snr_integrated": snr_integrated,
        "frb_source": frb_name,
        "paper": "Fonseca et al. (2024) ApJS 271:49",
        "doi": "10.3847/1538-4365/ad27d6", "arxiv": "2311.05829",
        "params_aproximados": params_aproximados or [],
    }

    fname = f"{frb_name}_dm{dm:07.2f}_fitburst.npz"
    sim_dir = Path(output_dir) if output_dir is not None else Path(__file__).resolve().parent / "simulaciones"
    sim_dir.mkdir(parents=True, exist_ok=True)
    npz_path = sim_dir / fname
    np.savez_compressed(npz_path, data_full=data_full, data_clean=signal_jy,
                         metadata=metadata, burst_parameters=burst_parameters)
    print(f"  ✓ Guardado: {npz_path} ({npz_path.stat().st_size/1024/1024:.1f} MB)")

    return {
        "filename": fname, "path": npz_path, "shape": data_full.shape,
        "max": float(data_full.max()), "snr_peak": snr_peak,
        "snr_integrated": snr_integrated, "dm": dm, "flux_jy": flux_jy,
    }


def generar_batch_desde_catalogo(nombres_frb, catalogo_normalizado, **kwargs_comunes):
    """
    Genera simulaciones para una lista de FRBs, tomando sus parámetros de un
    catálogo ya cargado con catalogo.cargar_catalogo() -- funciona igual
    para catalog1 o catalog2 (o cualquier catálogo nuevo agregado en
    catalogo.py), porque ese módulo ya normalizó los nombres de columna.

    Parámetros
    ----------
    nombres_frb : list[str]
        Nombres a simular (deben existir como claves de catalogo_normalizado)
    catalogo_normalizado : dict
        Como el que devuelve catalogo.cargar_catalogo()
    **kwargs_comunes
        Parámetros compartidos por todas las simulaciones (freq_lo, freq_hi,
        num_freq, res_time_ms, sefd_jy, n_pol, output_dir, etc.)
    """
    resultados = []
    faltantes = [n for n in nombres_frb if n not in catalogo_normalizado]
    if faltantes:
        print(f"⚠ No encontrados en el catálogo: {faltantes}")

    for nombre in nombres_frb:
        if nombre not in catalogo_normalizado:
            continue
        row = catalogo_normalizado[nombre]
        if row["dm"] is None or row["flux_jy"] is None or row["sp_idx"] is None or row["sp_run"] is None:
            print(f"⚠ {nombre}: faltan parámetros obligatorios (dm/flux_jy/sp_idx/sp_run) -- omitido")
            continue
        resultado = generar_frb_fitburst_paper(
            frb_name=nombre,
            dm=row["dm"],
            width_fitb_s=row["width_s"] if row["width_s"] is not None else 0.001,
            scat_s_600mhz=row["scat_s"] or 0.0,
            sp_idx=row["sp_idx"],
            sp_run=row["sp_run"],
            flux_jy=row["flux_jy"],
            **kwargs_comunes,
        )
        resultados.append(resultado)

    print(f"\n{'='*80}\nRESUMEN DEL BATCH ({len(resultados)}/{len(nombres_frb)} generados)\n{'='*80}")
    for r in resultados:
        print(f"  {r['dm']:8.2f} pc/cm³ | SNR pico={r['snr_peak']:8.1f} | "
              f"SNR integrada={r['snr_integrated']:6.1f} | {r['filename']}")

    return resultados
