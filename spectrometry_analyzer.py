# spectrometry_analyzer.py

import os
import re
import requests
import h5py
import pandas as pd
import numpy as np
from urllib.parse import urljoin
from scipy.signal import savgol_filter, find_peaks
from numpy import trapezoid as trapz

# --- Constantes y Configuración del Módulo ---
WL_MIN, WL_MAX          = 400, 900
TOLERANCE               = 0.7
BASELINE_WIN            = 101
BASELINE_POLY           = 3
SMOOTH_WIN              = 5
SMOOTH_POLY             = 2
PRIORITY = ['AAA','AA','A','B+','B','C+','C','D+','D','E']
SPECTROMETER_BASE_FMT = "http://golem.fjfi.cvut.cz/shots/{shot_no}/Devices/Radiation/MiniSpectrometer/"

# --- Listas de Búsqueda Mejoradas ---
CANDIDATE_FILENAMES = [
    "IRVISUV_0.h5", 
    "IRVIS_0.h5", 
    "Spectrometer_vis_0.h5",
    "Spectrometer_VIS_0.h5", 
    "Spectrometer_vis.h5",
]
KNOWN_SUBDIRS = ["", "HR2000+ES-a/", "HR2000+ES-b/"]

# --- Funciones de Utilidad ---
def lighten_color(hex_color, amount=0.3):
    try:
        hex_color = hex_color.lstrip('#')
        r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        r = min(255, int(r * (1 + amount))); g = min(255, int(g * (1 + amount))); b = min(255, int(b * (1 + amount)))
        return f'#{r:02x}{g:02x}{b:02x}'
    except Exception: return hex_color

def _integrate_peak_local_baseline(spectrum, wavelengths, center_wl, integration_width=5.0):
    roi_mask = (wavelengths >= center_wl - integration_width / 2) & (wavelengths <= center_wl + integration_width / 2)
    roi_wl, roi_spec = wavelengths[roi_mask], spectrum[roi_mask]
    if len(roi_wl) < 2: return 0.0
    start_point, end_point = (roi_wl[0], roi_spec[0]), (roi_wl[-1], roi_spec[-1])
    try:
        baseline_coeffs = np.polyfit([start_point[0], end_point[0]], [start_point[1], end_point[1]], 1)
    except (np.linalg.LinAlgError, ValueError): baseline_coeffs = (0, start_point[1])
    local_baseline = np.polyval(baseline_coeffs, roi_wl)
    net_spectrum = np.maximum(roi_spec - local_baseline, 0)
    return trapz(net_spectrum, x=roi_wl)

# --- Funciones para Encontrar y Cargar Datos ---
def _http_ok(url, timeout=5):
    try:
        # Usamos HEAD para una verificación rápida sin descargar el archivo completo
        return requests.head(url, timeout=timeout).status_code == 200
    except requests.exceptions.RequestException:
        return False

def _find_spectrometer_url(shot_no):
    """
    Función de búsqueda robusta que itera sobre todas las combinaciones
    de subdirectorios y nombres de archivo conocidos.
    """
    base = SPECTROMETER_BASE_FMT.format(shot_no=shot_no)
    
    for sub_dir in KNOWN_SUBDIRS:
        for filename in CANDIDATE_FILENAMES:
            # Construir la URL completa para esta combinación
            url = urljoin(base, os.path.join(sub_dir, filename).replace("\\", "/"))
            
            # Verificar si la URL es válida
            if _http_ok(url):
                print(f"[Spectro] ¡Éxito! Archivo encontrado en: {url}")
                return url
    
    print(f"[Spectro] ADVERTENCIA: No se pudo encontrar un archivo .h5 conocido para el disparo #{shot_no}.")
    return None

def download_h5(shot_no, target_dir):
    url = _find_spectrometer_url(shot_no)
    if not url: return None
    os.makedirs(target_dir, exist_ok=True)
    target_path = os.path.join(target_dir, "spectrometer_data.h5")
    try:
        r = requests.get(url, timeout=30); r.raise_for_status()
        with open(target_path, 'wb') as f: f.write(r.content)
        print(f"[Spectro] H5 guardado en {target_path}")
        return target_path
    except requests.exceptions.RequestException as e:
        print(f"Error descargando H5 ({url}): {e}"); return None

def load_nist(csv_path="nist_spectral_lines.csv"):
    if not os.path.isabs(csv_path):
        base_dir = os.path.dirname(__file__); csv_path = os.path.join(base_dir, csv_path)
    try:
        df = pd.read_csv(csv_path, sep=';')
        df['Wavelength'] = pd.to_numeric(df['Wavelength'], errors='coerce')
        return df.dropna(subset=['Wavelength']).reset_index(drop=True)
    except FileNotFoundError: return None

# --- Funciones de Análisis y Ploteo ---
def _map_peaks(wl_arr, signal, nist_df, peak_height, peak_distance):
    idxs, _ = find_peaks(signal, height=peak_height, distance=peak_distance)
    if not idxs.any(): return [], [], []
    wls, intensities = wl_arr[idxs], signal[idxs]
    ions, mapped_wls = [], []
    for wl_peak in wls:
        sel = nist_df[(nist_df['Wavelength'] >= wl_peak - TOLERANCE) & (nist_df['Wavelength'] <= wl_peak + TOLERANCE)].copy()
        if not sel.empty:
            sel['rank'] = sel['Acc.'].apply(lambda a: PRIORITY.index(a) if a in PRIORITY else len(PRIORITY))
            sel['delta'] = np.abs(sel['Wavelength'] - wl_peak)
            best = sel.sort_values(['rank', 'delta']).iloc[0]
            ions.append(best['Ion']); mapped_wls.append(best['Wavelength'])
        else:
            ions.append("Unknown"); mapped_wls.append(wl_peak)
    return ions, mapped_wls, intensities.tolist()

# En spectrometry_analyzer.py

def _detect_main_ions_for_panel(h5_path, nist_df, peak_height=250):
    """
    Función exhaustiva que analiza CADA frame del disparo para encontrar todos los
    iones que superan el umbral en cualquier momento.
    """
    try:
        with h5py.File(h5_path, 'r') as f:
            all_wl, all_spectra = f['Wavelengths'][:], f['Spectra'][:]
        
        all_ions_found = set() # Usamos un 'set' para guardar los iones únicos ((ion, wl)) sin repetición

        # --- BUCLE PRINCIPAL: Iterar sobre cada frame del espectro ---
        for spectrum in all_spectra:
            # Procesar cada espectro individualmente
            smooth = savgol_filter(np.maximum(spectrum - savgol_filter(spectrum, BASELINE_WIN, BASELINE_POLY), 0), SMOOTH_WIN, SMOOTH_POLY)
            
            # Buscar picos en este frame específico
            ions, wls, intens = _map_peaks(all_wl, smooth, nist_df, peak_height, peak_distance=5)
            
            # Añadir los iones encontrados (que no sean 'Unknown') al set
            known_ions_in_frame = [(i, w) for i, w in zip(ions, wls) if i != "Unknown" and WL_MIN <= w <= WL_MAX]
            
            if known_ions_in_frame:
                all_ions_found.update(known_ions_in_frame)

        # Si encontramos iones en cualquier frame, los procesamos y devolvemos
        if all_ions_found:
            # Ordenar la lista final por longitud de onda para consistencia
            sorted_ions = sorted(list(all_ions_found), key=lambda x: x[1])
            
            if sorted_ions:
                ions_unzipped, wls_unzipped = zip(*sorted_ions)
                return list(ions_unzipped), list(wls_unzipped)
            
    except Exception as e:
        print(f"Error detectando iones de forma exhaustiva: {e}")
        
    # Si no se encuentra nada en ningún frame, devolver listas vacías
    return [], []


def plot_ion_evolution_on_ax(ax, shot_number, shot_color, h5_path, ions_to_process, scaling_dict, formation_time=0.0):
    ax.set_xlabel("Tiempo [ms]"); ax.set_ylabel("Intensidad (A.U.)")
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    if not h5_path or not os.path.exists(h5_path) or not ions_to_process:
        ax.text(0.5, 0.5, 'No hay iones para graficar', transform=ax.transAxes, ha='center')
        return

    try:
        # --- INICIO DE LA CORRECCIÓN ---
        with h5py.File(h5_path, 'r') as f:
            all_wl, all_spectra = f['Wavelengths'][:], f['Spectra'][:]
            
            # Cargar el vector de tiempo REAL del archivo H5
            if 'Time' in f:
                # Asumir que el tiempo está en SEGUNDOS y convertir a ms
                time_axis_ms = f['Time'][:] * 1000.0
            else:
                # Si no existe, usar la lógica anterior como fallback (pero advertir)
                print("ADVERTENCIA: No se encontró 'Time' en H5. Usando lógica de 'formation_time'.")
                time_step_ms = 1.6 
                time_axis_ms = (np.arange(all_spectra.shape[0]) * time_step_ms) + formation_time
        # --- FIN DE LA CORRECCIÓN ---

        baseline_frames_count = 3
        color_shades = [lighten_color(shot_color, amount=i * 0.25) for i in range(len(ions_to_process))]
        
        for i, (ion_label, center_wl) in enumerate(ions_to_process):
            scale_factor = scaling_dict.get((ion_label, center_wl), 1.0)
            raw_intensities = [
                _integrate_peak_local_baseline(savgol_filter(np.maximum(s - savgol_filter(s, BASELINE_WIN, BASELINE_POLY), 0), SMOOTH_WIN, SMOOTH_POLY), all_wl, center_wl) 
                for s in all_spectra
            ]
            intensities_np = np.array(raw_intensities)
            
            # La lógica de baseline ahora funcionará, ya que los primeros frames
            # corresponderán al tiempo real (antes del plasma)
            baseline_level = np.min(intensities_np[:baseline_frames_count])
            corrected_intensities = np.maximum(intensities_np - baseline_level, 0) * scale_factor
            final_evolution = np.maximum(savgol_filter(corrected_intensities, 5, 2), 0) if len(corrected_intensities) > 5 else np.maximum(corrected_intensities, 0)

            if np.max(final_evolution) > 0:
                ax.plot(time_axis_ms, final_evolution, color=color_shades[i], label=f"{ion_label} {center_wl:.1f} nm")

    except Exception as e:
        print(f"Error en ploteo de iones: {e}"); ax.text(0.5, 0.5, f'Error: {e}', transform=ax.transAxes, ha='center', color='red')

# En spectrometry_analyzer.py

def get_ion_evolution(h5_path, ions_to_process, scaling_dict, formation_time):
    """
    Calcula la evolución temporal para una lista de iones dada y la devuelve.
    No grafica nada, solo calcula.
    """
    results = {}

    with h5py.File(h5_path, 'r') as f:
        all_wl, all_spectra = f['Wavelengths'][:], f['Spectra'][:]
        
        # Cargar el vector de tiempo REAL del archivo H5
        if 'Time' in f:
            # Asumir que el tiempo está en SEGUNDOS y convertir a ms
            time_axis_ms = f['Time'][:] * 1000.0
        else:
            # Si no existe, usar la lógica anterior como fallback (pero advertir)
            print("ADVERTENCIA: No se encontró 'Time' en H5. Usando lógica de 'formation_time'.")
            time_step_ms = 1.6
            time_axis_ms = (np.arange(all_spectra.shape[0]) * time_step_ms) + formation_time

    baseline_frames_count = 3

    for ion_label, center_wl in ions_to_process:
        scale_factor = scaling_dict.get((ion_label, center_wl), 1.0)
        raw_intensities = [
            _integrate_peak_local_baseline(savgol_filter(np.maximum(s - savgol_filter(s, BASELINE_WIN, BASELINE_POLY), 0), SMOOTH_WIN, SMOOTH_POLY), all_wl, center_wl) 
            for s in all_spectra
        ]
        intensities_np = np.array(raw_intensities)
        baseline_level = np.min(intensities_np[:baseline_frames_count])
        corrected_intensities = np.maximum(intensities_np - baseline_level, 0) * scale_factor
        final_evolution = np.maximum(savgol_filter(corrected_intensities, 5, 2), 0) if len(corrected_intensities) > 5 else np.maximum(corrected_intensities, 0)
        results[(ion_label, center_wl)] = final_evolution
    
    return time_axis_ms, results