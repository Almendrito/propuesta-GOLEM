# ui/main_window.py

import tkinter as tk
from tkinter import simpledialog, messagebox
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.widgets import Slider
from scipy import interpolate, signal
import numpy as np
import os
import itertools
import platform
import subprocess
from PIL import Image, ImageTk
import pyperclip
import h5py
from scipy.signal import savgol_filter

import spectrometry_analyzer
from .widgets import IonSidebarPanel, FilterConfigDialog
from core import data_loader

class TokamakDataViewer:
    def __init__(self, root, base_dir):
        self.root = root
        self.root.title("GOLEM Tokamak Data Viewer")
        self.base_dir = base_dir
        self.shots = {}
        self.current_shot = None
        self.ion_sidebar_panel = None
        self.color_palette = ['#003f5c', '#7a5195', '#ef5675', '#ffa600']
        self.image_refs = []
        self.spec_peak_height = 150 
        self.filter_enabled = False
        self.cursor_dynamics_enabled = False
        self.savgol_window = 9
        self.savgol_polyorder = 3
        self.last_cursor_x = None

        plt.rcParams.update({'font.size': 8, 'axes.labelsize': 8, 'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 'small', 'lines.linewidth': 1.2, 'axes.titlesize': 10})

        try:
            self.nist_df = spectrometry_analyzer.load_nist("nist_spectral_lines.csv")
            if self.nist_df is None:
                messagebox.showwarning("Advertencia", "No se pudo cargar 'nist_spectral_lines.csv'.")
        except Exception as e:
            self.nist_df = None
            messagebox.showerror("Error", f"Error al cargar archivo NIST: {e}")

        self.main_frame = tk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        self.top_button_frame = tk.Frame(self.main_frame)
        self.top_button_frame.pack(side=tk.TOP, fill=tk.X)
        tk.Button(self.top_button_frame, text="Load Shot", command=self.load_shot).pack(side=tk.LEFT, padx=5, pady=5)
        tk.Button(self.top_button_frame, text="Load Local", command=self.load_local_shot).pack(side=tk.LEFT, padx=5, pady=5)
        tk.Button(self.top_button_frame, text="Clear Shots", command=self.clear_shots).pack(side=tk.LEFT, padx=5, pady=5)

        # --- NUEVO CONTROL DE UMBRAL EN TIEMPO REAL ---
        #tk.Label(self.top_button_frame, text="Peak Threshold:").pack(side=tk.LEFT, padx=(10,0))
        #self.peak_threshold_var = tk.StringVar(value=str(self.spec_peak_height))
        #threshold_entry = tk.Entry(self.top_button_frame, width=6, textvariable=self.peak_threshold_var)
        #threshold_entry.pack(side=tk.LEFT, padx=5)
        #threshold_entry.bind("<Return>", self.on_threshold_change)
        #threshold_entry.bind("<FocusOut>", self.on_threshold_change)

        #tk.Button(self.top_button_frame, text="Visualizar Picos", command=self.visualize_spectrum_peaks).pack(side=tk.LEFT, padx=5, pady=5)
        self.cursor_toggle_button = tk.Button(self.top_button_frame, text="Enable Cursor", command=self.toggle_cursor_dynamics)
        self.cursor_toggle_button.pack(side=tk.LEFT, padx=5, pady=5)
        self.sidebar_button = tk.Button(self.top_button_frame, text="Panel de Iones", command=self.show_ion_sidebar)
        self.sidebar_button.pack(side=tk.RIGHT, padx=5, pady=5)

        self.plot_frame = tk.Frame(self.main_frame)
        self.plot_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.canvas_container = tk.Frame(self.plot_frame)
        self.canvas_container.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.fig, self.axs = plt.subplots(4, 2, sharex=True)
        self.fig.subplots_adjust(left=0.07, right=0.97, top=0.96, bottom=0.07, hspace=0.3, wspace=0.18)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.canvas_container)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.canvas_container)
        canvas_widget = self.canvas.get_tk_widget()
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.canvas_container)
        self.toolbar.update()
        self.toolbar.pack(side=tk.BOTTOM, fill=tk.X)
        canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.data_box_label = tk.Label(self.toolbar, text="", font=("Courier New", 8))
        self.data_box_label.pack(side=tk.LEFT, padx=10)

        self.right_panel = tk.Frame(self.main_frame, bg='white')
        self.right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=False,padx=(0, 5))
        self.png_frame = tk.Frame(self.right_panel, bg='white')
        self.png_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.cursor_lines = []
        self.canvas.draw()

    def load_shot(self, shot_number=None):
        if shot_number is None:
            shot_number = simpledialog.askinteger("Input", "Enter shot number:", parent=self.root)
        
        if not shot_number: return

        try:
            shot_folder = os.path.join(self.base_dir, "data", f"shot_{shot_number}")
            pickle_path = os.path.join(shot_folder, "shot_data.pkl")
            if os.path.exists(pickle_path):
                os.remove(pickle_path)
                print(f"Caché para #{shot_number} eliminado para forzar re-análisis.")

            shot_data = data_loader.fetch_shot_data(shot_number, self.base_dir, self.nist_df, self.spec_peak_height)
            self.shots[shot_number] = shot_data
            self.current_shot = shot_number
            
            # --- LÓGICA DE AVISO MEJORADA Y MÁS PRECISA ---
            
            # Primero, verificar si el archivo .h5 siquiera existe
            if not shot_data.get('h5_path'):
                messagebox.showwarning(
                    "Sin Datos de Espectrómetro",
                    f"No se encontró un archivo de espectrómetro (.h5) en el servidor para el disparo #{shot_number}.\n\n"
                    "La gráfica de evolución de iones estará vacía."
                )
            # Si el archivo existe, pero no se encontraron iones, entonces es un problema de umbral
            elif not shot_data.get('shot_ions_data'):
                messagebox.showinfo(
                    "Aviso de Umbral",
                    f"No se detectaron iones para #{shot_number} con el umbral actual ({self.spec_peak_height}).\n\n"
                    "Sugerencia: Baje el umbral con 'Set Peak Threshold' y vuelva a cargar el disparo."
                )
            
            if self.ion_sidebar_panel and tk.Toplevel.winfo_exists(self.ion_sidebar_panel):
                self.ion_sidebar_panel.destroy()
                self.ion_sidebar_panel = None
            
            self.plot_data()
            self.load_png_image(shot_number)

        except Exception as e:
            messagebox.showerror("Error", f"Fallo al cargar el disparo {shot_number}: {e}")
            import traceback; traceback.print_exc()

    def load_local_shot(self):
        shot_number = simpledialog.askinteger("Input", "Enter local shot number:", parent=self.root)
        if not shot_number: return
        # Esta función ahora simplemente carga desde el caché si existe
        try:
            shot_data = data_loader.fetch_shot_data(shot_number, self.base_dir, self.nist_df, self.spec_peak_height)
            self.shots[shot_number] = shot_data
            self.current_shot = shot_number
            self.plot_data()
            self.load_png_image(shot_number)
        except Exception as e:
            messagebox.showerror("Error", f"Fallo al cargar el disparo local {shot_number}: {e}")

    def load_png_image(self, shot_number):
        image_path = data_loader.fetch_shot_image_path(shot_number, self.base_dir)
        if not image_path: return
        # Evitar duplicados
        for widget in self.png_frame.winfo_children():
            if hasattr(widget, "shot_number") and widget.shot_number == shot_number:
                return
        wrapper_frame = tk.Frame(self.png_frame, bg='white', padx=10); wrapper_frame.pack(side=tk.TOP, pady=5)
        wrapper_frame.shot_number = shot_number
        tk.Label(wrapper_frame, text=f"Shot #{shot_number}", bg='white', font=("Arial", 10, "bold")).pack(side=tk.TOP)
        image = Image.open(image_path); image.thumbnail((300, 300)); photo = ImageTk.PhotoImage(image)
        img_label = tk.Label(wrapper_frame, image=photo, bg='white'); img_label.image = photo; img_label.pack(side=tk.TOP, pady=5)
        self.image_refs.append(photo)
        img_label.bind("<Button-1>", lambda e, p=image_path: self.open_in_system_viewer(p))


    def plot_data(self):
        for ax in self.fig.get_axes():
            ax.clear()
        
        color_cycle = itertools.cycle(self.color_palette)
        
        for shot, data in self.shots.items():
            color = next(color_cycle)
            lighter_color = self.lighter_color(color, 1.5)
            
            # --- Ploteo de datos básicos ---
            self.axs[0, 0].plot(data['Bt']['time_ms'], self.apply_filter(data['Bt']['Bt']), label=f'Bt ({shot})', color=color)
            self.axs[0, 1].plot(data['Ip']['time_ms'], self.apply_filter(data['Ip']['Ip']), label=f'Ip ({shot})', color=color)
            self.axs[1, 0].plot(data['U_loop']['time_ms'], self.apply_filter(data['U_loop']['U_loop']), label=f'U_loop ({shot})', color=color)
            self.axs[1, 1].plot(data['ne']['time_ms'], self.apply_filter(data['ne']['ne']), label=f'ne ({shot})', color=color)
            self.axs[2, 0].plot(data['fast_camera_radial']['time_ms'], self.apply_filter(data['fast_camera_radial']['radial_displacement']), label=f'Δr ({shot})', color=color)
            self.axs[2, 0].plot(data['fast_camera_vertical']['time_ms'], self.apply_filter(data['fast_camera_vertical']['vertical_displacement']), label=f'Δv ({shot})', color=lighter_color)
            self.axs[2, 1].plot(data['Te']['time_ms'], self.apply_filter(data['Te']['Te_0']), label=f'Te_0 ({shot})', color=color)
            self.axs[2, 1].plot(data['Te']['time_ms'], self.apply_filter(data['Te']['Te_avg_a']), label=f'Te_avg_a ({shot})', color=lighter_color, linestyle='--')
            if not data['confinement_time'].empty:
                self.axs[3, 0].plot(data['confinement_time']['time_ms'], self.apply_filter(data['confinement_time']['tau'] * 1e6), label=f'τ_e ({shot})', color=color)

            # --- Lógica de Ploteo de Iones ---
            ax_spec = self.axs[3, 1]
            all_detected_ions = data.get('shot_ions_data', [])
            ions_to_process = []
            scaling_dict = {}

            if self.ion_sidebar_panel and tk.Toplevel.winfo_exists(self.ion_sidebar_panel):
                user_selection_dict = self.ion_sidebar_panel.get_active_ions_and_scales().get(shot, {})
                if user_selection_dict:
                    #user_selection = {ion: scale for ion, scale in user_selection_dict}
                    ions_to_process = list(user_selection_dict.keys())
                    scaling_dict = user_selection_dict
                else:
                    ions_to_process = []
            else:
                ions_to_process = all_detected_ions
                scaling_dict = {ion: 1.0 for ion, wl in all_detected_ions}

            spectrometry_analyzer.plot_ion_evolution_on_ax(
                ax=ax_spec, shot_number=shot, shot_color=color,
                h5_path=data.get('h5_path'),
                ions_to_process=ions_to_process,
                scaling_dict=scaling_dict, 
                formation_time=data.get('formation_time', 0.0)
            )

        # --- Configuración final de los ejes y la leyenda (COMPLETA) ---
        labels = [
            ['Bt [T]', 'Ip [kA]'], 
            ['U_loop [V]', 'ne [m^-3]'], 
            ['Desplazamiento [mm]', 'Te [eV]'], 
            ['τ_e [μs]', 'Evolución Iones [U.A.]']
        ]
        for i in range(4):
            for j in range(2):
                ax = self.axs[i, j]
                ax.grid(True, which='both', linestyle='--', linewidth=0.5)
                # Solo mostrar etiquetas del eje X en la última fila
                ax.tick_params(axis='x', labelbottom=(i == 3))
                
                if ax.has_data():
                    num_columns = len(self.shots) if (i, j) == (3, 1) else 1
                    ax.legend(fontsize=20, loc='best', ncol=num_columns)
                
                ax.set_ylabel(labels[i][j])
                
                # Asegurar que el eje X esté compartido correctamente
                if not (i == 0 and j == 0):
                    ax.sharex(self.axs[0, 0])
                           # Dentro de la función plot_data, en el bucle for de configuración de ejes

                if (i, j) == (3, 1): # Solo para el subplot de iones
                    num_columns = len(self.shots) if self.shots else 1
                    # CAMBIO: Usar bbox_to_anchor para colocarla fuera, a la derecha
                    ax.legend(fontsize=6, loc='best', ncol=num_columns,frameon=False)
                else:
                    ax.legend(loc='best')
        self.axs[3,0].set_xlabel('Tiempo [ms]')
        self.axs[3,1].set_xlabel('Tiempo [ms]')
        
        #self.fig.tight_layout(pad=1.0)
        self.canvas.draw()

    def clear_shots(self):
        self.shots.clear(); self.current_shot = None; self.image_refs.clear()
        if self.ion_sidebar_panel and tk.Toplevel.winfo_exists(self.ion_sidebar_panel):
            self.ion_sidebar_panel.destroy()
        for widget in self.png_frame.winfo_children(): widget.destroy()
        self.plot_data()

    def show_ion_sidebar(self):
        if not self.shots: messagebox.showwarning("Sin datos", "Carga un disparo."); return
        ions_for_panel = {shot: data.get('shot_ions_data', []) for shot, data in self.shots.items()}
        if not any(ions_for_panel.values()): messagebox.showwarning("Sin datos", "No se detectaron iones."); return
        if self.ion_sidebar_panel and tk.Toplevel.winfo_exists(self.ion_sidebar_panel):
            self.ion_sidebar_panel.lift()
        else:
            self.ion_sidebar_panel = IonSidebarPanel(self.root, ions_for_panel, self.plot_data)

    def apply_filter(self, y):
        return y # Placeholder

    def open_in_system_viewer(self, image_path):
        try:
            if platform.system() == "Windows": os.startfile(image_path)
            elif platform.system() == "Darwin": subprocess.run(["open", image_path])
            else: subprocess.run(["xdg-open", image_path])
        except Exception as e: messagebox.showerror("Error", f"No se pudo abrir la imagen: {e}")

    def on_threshold_change(self, event=None):
        """
        Se activa al cambiar el valor del umbral. Redetecta los iones y
        actualiza las gráficas y el panel sin recargar todo el disparo.
        """
        if not self.current_shot: return
        try:
            new_threshold = int(self.peak_threshold_var.get())
            if new_threshold <= 0: return
            self.spec_peak_height = new_threshold
            print(f"Nuevo umbral de picos: {self.spec_peak_height}")

            # --- LÓGICA DE ACTUALIZACIÓN EN TIEMPO REAL ---
            current_shot_data = self.shots[self.current_shot]
            h5_path = current_shot_data.get('h5_path')
            
            # 1. Volver a detectar iones con el nuevo umbral
            detected_ions = spectrometry_analyzer.detect_ions_in_shot(h5_path, self.nist_df, self.spec_peak_height)
            
            # 2. Actualizar la lista de iones del disparo actual
            current_shot_data['shot_ions_data'] = detected_ions
            
            # 3. Si el panel de iones está abierto, cerrarlo para forzar su recreación con la nueva lista
            if self.ion_sidebar_panel and tk.Toplevel.winfo_exists(self.ion_sidebar_panel):
                self.ion_sidebar_panel.destroy()
                self.ion_sidebar_panel = None
                messagebox.showinfo("Panel Actualizado", "La lista de iones en el panel ha sido actualizada. Ábrelo de nuevo para ver los cambios.")
            
            # 4. Volver a dibujar todas las gráficas
            self.plot_data()

        except ValueError:
            self.peak_threshold_var.set(str(self.spec_peak_height)) # Revertir si no es un número válido

    def visualize_spectrum_peaks(self):
        if not self.current_shot: messagebox.showwarning("Sin datos", "Carga un disparo."); return
        shot_data = self.shots[self.current_shot]; h5_path = shot_data.get('h5_path')
        if not h5_path or not os.path.exists(h5_path): messagebox.showerror("Error", "No se encontró archivo .h5."); return
        try:
            with h5py.File(h5_path, 'r') as f: all_wl, all_spectra = f['Wavelengths'][:], f['Spectra'][:]
            total_frames = all_spectra.shape[0]; ref_idx = np.argmax(np.sum(all_spectra, axis=1))
            peak_window = tk.Toplevel(self.root); peak_window.title(f"Picos - Disparo #{self.current_shot}"); peak_window.geometry("900x650")
            fig, ax = plt.subplots(); fig.subplots_adjust(bottom=0.2)
            canvas = FigureCanvasTkAgg(fig, master=peak_window); NavigationToolbar2Tk(canvas, peak_window).update()
            canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
            
            def update_plot(frame_idx_float):
                frame_idx = int(frame_idx_float); ax.clear()
                spectrum = all_spectra[frame_idx]
                smooth = savgol_filter(np.maximum(spectrum - savgol_filter(spectrum, 101, 3), 0), 5, 2)
                ions, wls, intensities = spectrometry_analyzer._map_peaks(all_wl, smooth, self.nist_df, self.spec_peak_height, 5)
                ax.plot(all_wl, smooth, label=f'Espectro Suavizado', color='cornflowerblue')
                ax.axhline(y=self.spec_peak_height, color='r', linestyle='--', label=f'Umbral ({self.spec_peak_height})')
                ax.scatter(wls, intensities, marker='x', color='red', s=50, zorder=5)
                for ion, wl, intensity in zip(ions, wls, intensities):
                    if ion != "Unknown": ax.text(wl, intensity * 1.05, f"{ion} {wl:.1f} nm", rotation=90, va='bottom', ha='center', fontsize=8)
                ax.set_title(f"Análisis del Frame: {frame_idx} (Tiempo: {frame_idx * 1.6:.1f} ms)"); ax.set_xlabel("Longitud de Onda (nm)"); ax.set_ylabel("Intensidad (A.U.)")
                ax.legend(); ax.grid(True, linestyle='--'); ax.set_xlim(400, 900); ax.set_ylim(bottom=0)
                fig.canvas.draw_idle()

            ax_slider = fig.add_axes([0.15, 0.05, 0.75, 0.03])
            frame_slider = Slider(ax=ax_slider, label='Frame', valmin=0, valmax=total_frames - 1, valinit=ref_idx, valstep=1)
            frame_slider.on_changed(update_plot)
            peak_window.slider = frame_slider
            update_plot(ref_idx)
        except Exception as e:
            messagebox.showerror("Error de Análisis", f"No se pudo analizar el espectro: {e}")


    # --- Métodos del Cursor Dinámico ---
    
    def toggle_cursor_dynamics(self):
        self.cursor_dynamics_enabled = not self.cursor_dynamics_enabled
        self.cursor_toggle_button.config(
            text="Disable Cursor Dynamics" if self.cursor_dynamics_enabled else "Enable Cursor Dynamics"
        )
        if self.cursor_dynamics_enabled:
            self.connect_cursor_events()
        else:
            self.disconnect_cursor_events()
            # Usar un bucle 'try/except' para eliminar las líneas de forma segura
            for line in self.cursor_lines:
                try:
                    line.remove()
                except Exception:
                    pass # Ignorar errores si la línea ya no existe o no se puede borrar
            self.cursor_lines.clear()
            if hasattr(self, "data_box_label"):
                self.data_box_label.config(text="")

        self.canvas.draw()
        
    def connect_cursor_events(self):
        # ... (Pega aquí el método connect_cursor_events COMPLETO de main_app.py) ...
        self.motion_cid = self.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)
        self.right_click_cid = self.canvas.mpl_connect('button_press_event', self.on_right_click)

    def disconnect_cursor_events(self):
        # ... (Pega aquí el método disconnect_cursor_events COMPLETO de main_app.py) ...
        if hasattr(self, 'motion_cid'): self.canvas.mpl_disconnect(self.motion_cid)
        if hasattr(self, 'right_click_cid'): self.canvas.mpl_disconnect(self.right_click_cid)

    def on_mouse_move(self, event):
        # ... (Pega aquí el método on_mouse_move COMPLETO de main_app.py) ...
        if not event.inaxes or not self.cursor_dynamics_enabled: return
        self.draw_cursor_at(event.xdata)

    def draw_cursor_at(self, x):
        # ... (Pega aquí el método draw_cursor_at COMPLETO de main_app.py) ...
        if x is None: return
        self.last_cursor_x = x
        for line in self.cursor_lines: line.remove()
        self.cursor_lines.clear()

        for ax_row in self.axs:
            for ax in ax_row:
                self.cursor_lines.append(ax.axvline(x=x, color='gray', linestyle='--', linewidth=0.8))

        header = "Shot\tTime(ms)\tBt(T)\tIp(kA)\tne(m-3)\tTe_0(eV)\ttau_e(us)"
        data_table = [header]
        for shot, data in self.shots.items():
            vals = {}
            for key, df in data.items():
                if isinstance(df, pd.DataFrame) and 'time_ms' in df.columns and not df.empty:
                    idx = (df['time_ms'] - x).abs().idxmin()
                    for col in df.columns:
                        if col != 'time_ms': vals[col] = df.loc[idx, col]
            row = (f"{shot}\t{x:.2f}\t"
                   f"{vals.get('Bt', np.nan):.2f}\t{vals.get('Ip', np.nan):.2f}\t"
                   f"{vals.get('ne', np.nan):.2e}\t{vals.get('Te_0', np.nan):.1f}\t"
                   f"{vals.get('tau', np.nan) * 1e6:.1f}")
            data_table.append(row.replace("nan", "---"))

        self.data_box_label.config(text="\n".join(data_table))
        self.canvas.draw_idle()

    def on_right_click(self, event):
        # ... (Pega aquí el método on_right_click COMPLETO de main_app.py) ...
        if not event.inaxes or not self.cursor_dynamics_enabled or event.button != 3: return
        x = event.xdata
        clipboard_text = "Shot\tTime(ms)\tBt(T)\tIp(kA)\tne(m-3)\tTe_0(eV)\ttau_e(us)\n"
        for shot, data in self.shots.items():
            vals = {}
            for key, df in data.items():
                if isinstance(df, pd.DataFrame) and 'time_ms' in df.columns and not df.empty:
                    idx = (df['time_ms'] - x).abs().idxmin()
                    for col in df.columns:
                        if col != 'time_ms': vals[col] = df.loc[idx, col]
            row = (f"{shot}\t{x:.2f}\t"
                   f"{vals.get('Bt', np.nan):.2f}\t{vals.get('Ip', np.nan):.2f}\t"
                   f"{vals.get('ne', np.nan):.2e}\t{vals.get('Te_0', np.nan):.1f}\t"
                   f"{vals.get('tau', np.nan) * 1e6:.1f}\n")
            clipboard_text += row.replace("nan", "")
        pyperclip.copy(clipboard_text)
        messagebox.showinfo("Copiado", "Datos del cursor copiados al portapapeles.")
        
    # --- Funciones de Utilidad ---
    
    @staticmethod
    def lighter_color(color, factor=1.5):
        # ... (Pega aquí el método lighter_color COMPLETO de main_app.py) ...
        r, g, b = tuple(int(color[i:i+2], 16) for i in (1, 3, 5))
        r = min(255, int(r * factor))
        g = min(255, int(g * factor))
        b = min(255, int(b * factor))
        return f"#{r:02x}{g:02x}{b:02x}"
    
    # En la clase TokamakDataViewer (ui/main_window.py), añade este nuevo método

# En la clase TokamakDataViewer (ui/main_window.py), REEMPLAZA el método entero

    def visualize_spectrum_peaks(self):
        """
        Abre una ventana con un slider para analizar el espectro y los picos
        detectados en cualquier frame del disparo actual.
        """
        if self.current_shot is None or self.current_shot not in self.shots:
            messagebox.showwarning("Sin datos", "Carga un disparo antes de visualizar los picos.")
            return

        shot_data = self.shots[self.current_shot]
        h5_path = shot_data.get('h5_path')

        if not h5_path or not os.path.exists(h5_path):
            messagebox.showerror("Error", f"No se encontró el archivo .h5 para el disparo {self.current_shot}.")
            return

        try:
            # Cargar todos los datos del espectrómetro una sola vez
            with h5py.File(h5_path, 'r') as f:
                all_wl, all_spectra = f['Wavelengths'][:], f['Spectra'][:]
            
            total_frames = all_spectra.shape[0]
            ref_idx = np.argmax(np.sum(all_spectra, axis=1)) # Frame inicial sugerido

            # --- Creación de la Ventana y Layout ---
            peak_window = tk.Toplevel(self.root)
            peak_window.title(f"Análisis de Picos - Disparo #{self.current_shot}")
            peak_window.geometry("900x650")

            # Frame principal para la gráfica y la barra de herramientas
            plot_frame = tk.Frame(peak_window)
            plot_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
            
            fig, ax = plt.subplots()
            fig.subplots_adjust(bottom=0.2) # Dejar espacio para el slider
            canvas = FigureCanvasTkAgg(fig, master=plot_frame)
            toolbar = NavigationToolbar2Tk(canvas, plot_frame)
            toolbar.update()
            canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
            
            # --- Función de Actualización de la Gráfica ---
            def update_plot(frame_idx_float):
                frame_idx = int(frame_idx_float)
                ax.clear() # Limpiar la gráfica anterior

                # Procesar el espectro para el frame seleccionado
                spectrum = all_spectra[frame_idx]
                bg = savgol_filter(spectrum, spectrometry_analyzer.BASELINE_WIN, spectrometry_analyzer.BASELINE_POLY)
                residual = np.maximum(spectrum - bg, 0)
                smooth = savgol_filter(residual, spectrometry_analyzer.SMOOTH_WIN, spectrometry_analyzer.SMOOTH_POLY)

                # Encontrar e identificar los picos con el umbral actual
                ions, wls, intensities = spectrometry_analyzer._map_peaks(
                    all_wl, smooth, self.nist_df, self.spec_peak_height, peak_distance=5
                )

                # Graficar los resultados
                ax.plot(all_wl, smooth, label='Espectro Suavizado', color='cornflowerblue', linewidth=1.2)
                ax.axhline(y=self.spec_peak_height, color='r', linestyle='--', label=f'Umbral ({self.spec_peak_height})', linewidth=1)
                ax.scatter(wls, intensities, marker='x', color='red', s=50, zorder=5) # zorder para que esté encima

                for ion, wl, intensity in zip(ions, wls, intensities):
                    if ion != "Unknown":
                        label = f"{ion} {wl:.1f} nm"
                        ax.text(wl, intensity * 1.05, label, rotation=90, va='bottom', ha='center', fontsize=8)

                # Configurar ejes y títulos
                ax.set_title(f"Análisis del Frame: {frame_idx} (Tiempo: {frame_idx * 1.6:.1f} ms)")
                ax.set_xlabel("Longitud de Onda (nm)")
                ax.set_ylabel("Intensidad (A.U.)")
                ax.legend()
                ax.grid(True, linestyle='--', alpha=0.6)
                ax.set_xlim(spectrometry_analyzer.WL_MIN, spectrometry_analyzer.WL_MAX)
                ax.set_ylim(bottom=0, top=np.max(residual) * 1.2 if np.max(residual) > 0 else 100)
                
                fig.canvas.draw_idle()

            # --- Creación del Slider ---
            ax_slider = fig.add_axes([0.15, 0.05, 0.75, 0.03]) # Posición del slider [left, bottom, width, height]
            frame_slider = Slider(
                ax=ax_slider,
                label='Frame',
                valmin=0,
                valmax=total_frames - 1,
                valinit=ref_idx,
                valstep=1 # Moverse de 1 en 1 frame
            )
            frame_slider.on_changed(update_plot)
            
            # Guardar referencia al slider para que no sea eliminado por el garbage collector
            peak_window.slider = frame_slider

            # --- Dibujar la gráfica inicial ---
            update_plot(ref_idx)

        except Exception as e:
            messagebox.showerror("Error de Análisis", f"No se pudo analizar el espectro: {e}")
            if 'peak_window' in locals(): peak_window.destroy()


    def configure_peak_threshold(self):
        """
        Abre un diálogo para que el usuario introduzca un nuevo umbral
        para la detección de picos de espectrometría.
        """
        new_threshold = simpledialog.askinteger(
            "Configurar Umbral de Pico",
            "Introduce la nueva altura mínima para detectar un pico (peak_height):",
            parent=self.root,
            initialvalue=self.spec_peak_height
        )

        # Si el usuario introduce un valor y no cancela
        if new_threshold is not None and new_threshold > 0:
            self.spec_peak_height = new_threshold
            print(f"Nuevo umbral de detección de picos establecido en: {self.spec_peak_height}")
            self.plot_data() # Volvemos a dibujar las gráficas con el nuevo umbral

    @staticmethod
    def lighter_color(color, factor=1.5):
        r, g, b = tuple(int(color[i:i+2], 16) for i in (1, 3, 5))
        return f"#{min(255, int(r*factor)):02x}{min(255, int(g*factor)):02x}{min(255, int(b*factor)):02x}"