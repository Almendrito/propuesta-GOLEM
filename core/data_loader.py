# core/data_loader.py

import os
import io
import requests
import pandas as pd
import pickle
import json
import spectrometry_analyzer
from . import physics

def _load_data(url, local_path, column_names, sep=','):
    if os.path.exists(local_path):
        return pd.read_csv(local_path, sep=sep)
    try:
        response = requests.get(url, timeout=20)
        response.raise_for_status()
        data = pd.read_csv(io.StringIO(response.text), header=None, names=column_names, sep=sep)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        data.to_csv(local_path, index=False)
        return data
    except requests.exceptions.RequestException as e:
        print(f"Error descargando {url}: {e}")
        return pd.DataFrame(columns=column_names)

# En core/data_loader.py

def _load_fast_camera_data(url, local_path, column_name):
    """
    Carga datos de cámara rápida de forma robusta, ignorando líneas incompletas.
    """
    if os.path.exists(local_path):
        return pd.read_csv(local_path)

    try:
        response = requests.get(url, timeout=20)
        response.raise_for_status()
        lines = response.text.strip().split('\n')
        time_ms, values = [], []
        
        for line in lines:
            parts = line.strip().split(',')
            if len(parts) == 2:
                try:
                    # --- LÓGICA CORREGIDA ---
                    # 1. Intentar convertir ambos valores primero
                    time_val = float(parts[0])
                    pos_val = float(parts[1])
                    
                    # 2. Solo si ambos son válidos, añadirlos a las listas
                    time_ms.append(time_val)
                    values.append(pos_val)
                except ValueError:
                    # Si alguna conversión falla (ej. una cadena vacía), ignorar la línea
                    continue
        
        data = pd.DataFrame({'time_ms': time_ms, column_name: values})
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        data.to_csv(local_path, index=False)
        return data
        
    except requests.exceptions.RequestException as e:
        print(f"Error descargando datos de cámara rápida desde {url}: {e}")
        return pd.DataFrame(columns=['time_ms', column_name])

def fetch_shot_image_path(shot_number, base_dir):
    local_folder = os.path.join(base_dir, "data", f"shot_{shot_number}")
    local_full_path = os.path.join(local_folder, "ScreenShotAll_full.png")
    os.makedirs(local_folder, exist_ok=True)
    if os.path.exists(local_full_path): return local_full_path
    png_url = f"http://golem.fjfi.cvut.cz/shots/{shot_number}/Diagnostics/FastCameras/ScreenShotAll.png"
    try:
        response = requests.get(png_url, timeout=30)
        response.raise_for_status()
        with open(local_full_path, 'wb') as f: f.write(response.content)
        return local_full_path
    except requests.exceptions.RequestException as e:
        print(f"No se pudo cargar la imagen para {shot_number}: {e}")
        return None

def fetch_shot_data(shot_number, base_dir, nist_df, spec_peak_height):
    local_folder = os.path.join(base_dir, "data", f"shot_{shot_number}")
    os.makedirs(local_folder, exist_ok=True)
    pickle_path = os.path.join(local_folder, "shot_data.pkl")

    if os.path.exists(pickle_path):
        print(f"Cargando disparo #{shot_number} desde caché (pickle).")
        try:
            with open(pickle_path, "rb") as f: return pickle.load(f)
        except Exception as e:
            print(f"Error al leer pickle, se re-descargarán los datos. Error: {e}")

    print(f"No se encontró caché para #{shot_number}. Descargando y procesando...")
    
    # ... (descarga de datos básicos como bt_data, ip_data, etc. es igual) ...
    bt_data = _load_data(f"http://golem.fjfi.cvut.cz/shots/{shot_number}/Diagnostics/BasicDiagnostics/Results/Bt.csv", os.path.join(local_folder, "Bt.csv"), ['time_ms', 'Bt'])
    ip_data = _load_data(f"http://golem.fjfi.cvut.cz/shots/{shot_number}/Diagnostics/BasicDiagnostics/Results/Ip.csv", os.path.join(local_folder, "Ip.csv"), ['time_ms', 'Ip'])
    u_loop_data = _load_data(f"http://golem.fjfi.cvut.cz/shots/{shot_number}/Diagnostics/BasicDiagnostics/Results/U_loop.csv", os.path.join(local_folder, "U_loop.csv"), ['time_ms', 'U_loop'])
    ne_data = _load_data(f"http://golem.fjfi.cvut.cz/shots/{shot_number}/Diagnostics/Interferometry/ne_lav.csv", os.path.join(local_folder, "ne.csv"), ['time_ms', 'ne'])
    fast_camera_vertical_data = _load_fast_camera_data(f"http://golem.fjfi.cvut.cz/shots/{shot_number}/Diagnostics/FastCameras/Camera_Vertical/CameraVerticalPosition", os.path.join(local_folder, "CameraVerticalPosition.csv"), 'vertical_displacement')
    fast_camera_radial_data = _load_fast_camera_data(f"http://golem.fjfi.cvut.cz/shots/{shot_number}/Diagnostics/FastCameras/Camera_Radial/CameraRadialPosition", os.path.join(local_folder, "CameraRadialPosition.csv"), 'radial_displacement')

    h5_path = spectrometry_analyzer.download_h5(shot_number, local_folder)
    shot_ions_data = [] 
    if h5_path and nist_df is not None:
        # La función de detección ahora solo devuelve 2 listas
        ions, wls = spectrometry_analyzer._detect_main_ions_for_panel(h5_path, nist_df, peak_height=spec_peak_height)
        # Guardamos como una lista de tuplas (ion, wl)
        shot_ions_data = list(zip(ions, wls))
        with open(os.path.join(local_folder, "spectrometry_metadata.json"), "w") as f: json.dump(shot_ions_data, f)
    
    te_data = physics.calculate_derived_data(ip_data, u_loop_data, bt_data)
    confinement_time_data = physics.calculate_confinement_time(ip_data, u_loop_data, ne_data)
    formation_time = physics.find_plasma_formation_time(ip_data)
    
    shot_data = {
        'Bt': bt_data, 'Ip': ip_data, 'U_loop': u_loop_data, 'ne': ne_data,
        'fast_camera_vertical': fast_camera_vertical_data, 
        'fast_camera_radial': fast_camera_radial_data,
        'Te': te_data, 'confinement_time': confinement_time_data, 
        'h5_path': h5_path, 'formation_time': formation_time, 
        'shot_ions_data': shot_ions_data
    }
    
    print(f"Guardando datos de #{shot_number} en caché (pickle).")
    with open(pickle_path, "wb") as f: pickle.dump(shot_data, f)
        
    return shot_data