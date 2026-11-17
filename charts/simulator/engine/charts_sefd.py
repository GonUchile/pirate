#!/usr/bin/env python3
"""
SEFD real de CHARTS (no el de CHIME que se usó por error en
CHARTS_chime_data/), derivado de las specs publicadas en el paper de
diseño (Cassanelli et al. 2024, "CHARTS: Canadian-Chilean array for
radio transient studies", charts-wp.pdf, Tabla 1):

  - T_sys <= 50 K
  - Ganancia por antena (cenit): ~7.1 dBi
  - n_a = 256 antenas, UNA sola polarización (n_p=1)
  - Banda: 300-500 MHz

Supuesto explícito pedido por el usuario: se usan las 256 antenas
combinadas coherentemente (array en fase / beamforming), es decir
A_eff_total = n_a * A_eff_antena_individual.

A_eff_antena = G_lineal * lambda^2 / (4*pi), evaluada en ref_freq
(400.1953125 MHz, la misma referencia fija usada en todo el pipeline
fitburst). SEFD = 2*k_B*T_sys / A_eff_total.

CHIME (28.76 Jy, 8000 m^2 físicos, T_sys=50K) es un radiotelescopio de
platos/cilindros grandes; CHARTS es un array de antenas chicas de bajo
costo -- por diseño prioriza cantidad de antenas y campo de visión por
sobre sensibilidad (ver Sec. 2 del paper). Resultado: SEFD ~82x peor
que el de CHIME.
"""
import numpy as np

K_BOLTZMANN = 1.380649e-23   # J/K
C_LIGHT = 299792458.0        # m/s

CHARTS_T_SYS_K = 50.0
CHARTS_ANTENNA_GAIN_DBI = 7.1
CHARTS_N_ANTENNAS = 256
CHARTS_N_POL = 1.0            # Tabla 1: "Antenna polarization number n_p = 1"
CHARTS_REF_FREQ_MHZ = 400.1953125  # misma referencia fija del resto del pipeline

_gain_linear = 10 ** (CHARTS_ANTENNA_GAIN_DBI / 10.0)
_wavelength_m = C_LIGHT / (CHARTS_REF_FREQ_MHZ * 1e6)
_a_eff_single_m2 = _gain_linear * _wavelength_m**2 / (4 * np.pi)
CHARTS_A_EFF_SINGLE_M2 = _a_eff_single_m2
CHARTS_A_EFF_ARRAY_M2 = CHARTS_N_ANTENNAS * _a_eff_single_m2

CHARTS_SEFD_JY = 2.0 * K_BOLTZMANN * CHARTS_T_SYS_K / CHARTS_A_EFF_ARRAY_M2 / 1e-26

# Tamaños de arreglo soportados por el pipeline (ver 01_generar_positivos.py
# --n-antenas). No es una limitación física de CHARTS -- son los valores que
# se decidió explorar para ver cómo escala la sensibilidad con la cantidad
# de antenas combinadas coherentemente.
N_ANTENAS_SOPORTADAS = (8, 16, 32, 64, 128, 256)


def calcular_sefd_charts(n_antenas=CHARTS_N_ANTENNAS):
    """SEFD de CHARTS para un arreglo de `n_antenas` antenas combinadas
    coherentemente (A_eff_total = n_antenas * A_eff_antena_individual).
    Con n_antenas=256 (default) da exactamente CHARTS_SEFD_JY."""
    a_eff_total = n_antenas * CHARTS_A_EFF_SINGLE_M2
    return 2.0 * K_BOLTZMANN * CHARTS_T_SYS_K / a_eff_total / 1e-26


def noise_std_teorico(meta):
    """Sigma del ruido de sistema, ecuacion del radiometro -- identica
    formula que engine/fitburst_sim.py (delta_nu_hz = res_freq_mhz*1e6).
    Un solo escalar: SEFD/n_pol/res_freq/res_time son uniformes entre
    canales en este instrumento (sin ripple de bandpass simulado), asi
    que esta sigma es la misma para los 672 canales. Compartida entre
    03_convert_to_fil.py (escala de cuantizacion fija, sin RFI) y
    06_fil_con_rfi.py (misma escala, multiplicada por el exceso
    persistente de cada canal) para que ambas esten ancladas al mismo
    piso teorico y sean comparables entre si."""
    sefd_jy = float(meta["sefd_jy"])
    n_pol = float(meta["n_pol"])
    delta_nu_hz = float(meta["res_freq"]) * 1e6
    res_time = float(meta["res_time"])
    return sefd_jy / np.sqrt(n_pol * delta_nu_hz * res_time)


if __name__ == "__main__":
    print(f"Ganancia lineal: {_gain_linear:.4f}")
    print(f"Longitud de onda @ {CHARTS_REF_FREQ_MHZ} MHz: {_wavelength_m:.4f} m")
    print(f"A_eff por antena: {CHARTS_A_EFF_SINGLE_M2:.5f} m²")
    print(f"A_eff array ({CHARTS_N_ANTENNAS} antenas): {CHARTS_A_EFF_ARRAY_M2:.2f} m²")
    print(f"SEFD_CHARTS (256 antenas, n_pol=1): {CHARTS_SEFD_JY:.2f} Jy")
