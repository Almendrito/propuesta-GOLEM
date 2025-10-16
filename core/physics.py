import numpy as np
import pandas as pd
from scipy import constants, signal, interpolate
from scipy.signal import savgol_filter


# core/physics.py


def electron_temperature_Spitzer_eV(eta_measured, Z_eff=3, eps=0, coulomb_logarithm=14):
    """
    Calcula la temperatura de electrones usando la resistividad de Spitzer.
    (Esta es la función de main_app.py, convertida a una función normal sin 'self').
    """
    if not isinstance(eta_measured, pd.Series) or eta_measured.empty:
        return pd.Series(dtype=float)
    eta_s = eta_measured / Z_eff * (1 - np.sqrt(eps))**2
    term = 1.96 * (eta_s *3*constants.epsilon_0**2) / (np.sqrt(constants.m_e) * constants.elementary_charge**2 * coulomb_logarithm)
    Te_eV = (term )**(-2 / 3) / (constants.elementary_charge * 2 * np.pi)
    return Te_eV.replace([np.inf, -np.inf], np.nan)

def calculate_derived_data(ip_data, u_loop_data, bt_data):
    """
    Calcula todos los parámetros físicos derivados a partir de los datos básicos.
    Toma los DataFrames de Ip, U_loop y Bt y devuelve el DataFrame de Te.
    (Esta función contiene la lógica extraída del método 'load_shot').
    """
    # 1. Combinar los datos de entrada en un único DataFrame
    combined_data = pd.merge(ip_data, u_loop_data, on='time_ms', how='outer')
    combined_data = pd.merge(combined_data, bt_data, on='time_ms', how='outer').interpolate().fillna(0)

    # 2. Definir parámetros del Tokamak y calcular densidades de corriente
    R0, a0, nu = 0.4, 0.085, 2
    combined_data['R'], combined_data['a'] = R0, a0
    combined_data['j_avg_a'] = combined_data['Ip'] * 1e3 / (np.pi * combined_data['a']**2)
    combined_data['j_0'] = combined_data['j_avg_a'] * (nu + 1)

    # 3. Calcular inductancia del plasma y campo eléctrico toroidal
    l_i = np.log(1.65 + 0.89 * nu)
    combined_data['L_p'] = constants.mu_0 * combined_data['R'] * (np.log(8 * combined_data['R'] / combined_data['a']) - 7/4 + l_i / 2)
    
    dt = np.diff(combined_data['time_ms'].values[:2]).item() if len(combined_data['time_ms'].values) > 1 else 1.0
    # Asegurar que la ventana del filtro sea un entero impar y mayor que el orden del polinomio
    n_win = max(5, int(0.5 / dt) + (1 - int(0.5 / dt) % 2)) if dt > 0 else 5
    
    combined_data['dIp_dt'] = signal.savgol_filter(combined_data['Ip'] * 1e3, n_win, 3, 1, delta=dt * 1e-3)
    combined_data['E_phi'] = (combined_data['U_loop'] - combined_data['L_p'] * combined_data['dIp_dt']) / (2 * np.pi * combined_data['R'])

    # 4. Calcular resistividad y temperatura de electrones
    combined_data['eta_0'] = combined_data['E_phi'] / combined_data['j_0'].replace(0, np.nan)
    combined_data['eta_avg_a'] = combined_data['E_phi'] / combined_data['j_avg_a'].replace(0, np.nan)
    
    # Llamamos a la función local, ya no a un método de clase
    combined_data['Te_0'] = electron_temperature_Spitzer_eV(combined_data['eta_0'], eps=combined_data['a']/combined_data['R'])
    combined_data['Te_avg_a'] = electron_temperature_Spitzer_eV(combined_data['eta_avg_a'], eps=combined_data['a']/combined_data['R'])
    
    te_data = combined_data[['time_ms', 'Te_0', 'Te_avg_a']]
    
    return te_data

def calculate_confinement_time(ip_data, u_loop_data, ne_data):

    """
    Calcula el tiempo de confinamiento energético (tau_e).
    Interpola los datos de Ip y U_loop a la base de tiempo de 'ne'.
    (Esta función contiene la lógica extraída del método 'load_shot').
    """
    # 1. Interpolar Ip y U_loop a la base de tiempo de 'ne' para alinearlos
    Ip_interp = interpolate.interp1d(ip_data['time_ms'], ip_data['Ip'], bounds_error=False, fill_value=np.nan)(ne_data['time_ms'])
    U_l_interp = interpolate.interp1d(u_loop_data['time_ms'], u_loop_data['U_loop'], bounds_error=False, fill_value=np.nan)(ne_data['time_ms'])

    # 2. Filtrar para asegurar que todos los valores sean físicamente válidos (mayores a cero)
    valid_idx = (ne_data['ne'] > 0) & (Ip_interp > 0) & (U_l_interp > 0)
    if not np.any(valid_idx):
        return pd.DataFrame(columns=['time_ms', 'tau']) # Devolver DataFrame vacío si no hay datos válidos

    # 3. Calcular tau_e solo para los índices válidos
    ne_valid = ne_data['ne'][valid_idx]
    Ip_interp_valid = Ip_interp[valid_idx]
    U_l_interp_valid = U_l_interp[valid_idx]

    tau = (1.0345 * ne_valid) / (16e19 * Ip_interp_valid**(1/3) * U_l_interp_valid**(5/3))

    # 4. Crear y devolver el DataFrame con los resultados
    confinement_time_data = pd.DataFrame({
        'time_ms': ne_data['time_ms'][valid_idx],
        'tau': tau.values
    })
    
    return confinement_time_data

def find_plasma_formation_time(ip_data, threshold=0.02):  # 2% threshold más conservador
    """
    Encuentra el tiempo REAL cuando el plasma comienza basado en Ip.
    """
    if ip_data.empty or 'Ip' not in ip_data.columns:
        print("Warning: No Ip data available, using t=0")
        return 0.0

    ip_values = ip_data['Ip'].values
    time_values = ip_data['time_ms'].values
    
    # Filtrar para eliminar ruido
    if len(ip_values) > 10:
        try:
            ip_smooth = savgol_filter(ip_values, window_length=7, polyorder=2)
        except:
            ip_smooth = ip_values
    else:
        ip_smooth = ip_values
    
    max_ip = np.max(ip_smooth)
    threshold_value = max_ip * threshold
    
    print(f"=== Plasma Formation Detection ===")
    print(f"Max Ip: {max_ip:.2f} kA")
    print(f"Threshold ({threshold*100}%): {threshold_value:.3f} kA")
    
    # Buscar el primer punto que supera el threshold
    above_threshold = np.where(ip_smooth > threshold_value)[0]
    
    if len(above_threshold) > 0:
        start_idx = above_threshold[0]
        
        # Buscar hacia atrás para encontrar el inicio real (donde Ip es casi 0)
        lookback = min(20, start_idx)
        for i in range(start_idx, max(0, start_idx - lookback), -1):
            if ip_smooth[i] < threshold_value * 0.1:  # 10% del threshold
                formation_time = time_values[i]
                print(f"Plasma formation detected at: {formation_time:.2f} ms")
                print(f"Corresponding Ip: {ip_smooth[i]:.3f} kA")
                print("===")
                return formation_time
        
        formation_time = time_values[start_idx]
        print(f"Plasma formation (fallback) at: {formation_time:.2f} ms")
        print("===")
        return formation_time
    
    print("No clear plasma formation detected, using t=0")
    print("===")
    return 0.0