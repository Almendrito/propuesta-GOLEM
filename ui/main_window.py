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
        self.spec_peak_height = 250 
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
        tk.Button(self.top_button_frame, text="Remove Shot(s)", command=self.remove_shot).pack(side=tk.LEFT, padx=5, pady=5)

        # --- NUEVO CONTROL DE UMBRAL EN TIEMPO REAL ---
        tk.Label(self.top_button_frame, text="Peak Threshold:").pack(side=tk.LEFT, padx=(10,0))
        self.peak_threshold_var = tk.StringVar(value=str(self.spec_peak_height))
        threshold_entry = tk.Entry(self.top_button_frame, width=6, textvariable=self.peak_threshold_var)
        threshold_entry.pack(side=tk.LEFT, padx=5)
        threshold_entry.bind("<Return>", self.on_threshold_change)
        threshold_entry.bind("<FocusOut>", self.on_threshold_change)

        tk.Button(self.top_button_frame, text="Visualizar Picos", command=self.visualize_spectrum_peaks).pack(side=tk.LEFT, padx=5, pady=5)
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
        self.data_box_label = tk.Label(self.toolbar, text="", font=("Arial", 8))
        self.data_box_label.pack(side=tk.LEFT, padx=10)

        # --- REEMPLAZA la línea de self.ion_pick_label CON ESTA ---
        self.ion_hover_label = tk.Label(self.toolbar, text="", font=("Arial", 9, "bold"), anchor='w')
        self.ion_hover_label.pack(side=tk.LEFT, padx=10, fill=tk.X, expand=True)
        #self.ion_pick_label.pack(side=tk.RIGHT, padx=10)
        self.canvas.mpl_connect('motion_notify_event', self.on_hover_ion_axes)

        self.right_panel = tk.Frame(self.main_frame, bg='white')
        self.right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=False,padx=(0, 5))
        self.ion_legend_frame = tk.Frame(self.right_panel, bg='white', padx=10, pady=5)
        self.ion_legend_frame.pack(side=tk.TOP, fill=tk.X, expand=False)
        self.png_frame = tk.Frame(self.right_panel, bg='white')
        self.png_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.ion_legend_frame = tk.Frame(self.right_panel, bg='white', padx=10, pady=5)
        self.ion_legend_frame.pack(side=tk.BOTTOM, fill=tk.X, expand=False)
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
                scaling_dict = {ion: 1.0 for (ion, wl) in all_detected_ions} # Corregido: la clave es una tupla

            spectrometry_analyzer.plot_ion_evolution_on_ax(
                ax=ax_spec, shot_number=shot, shot_color=color,
                h5_path=data.get('h5_path'),
                ions_to_process=ions_to_process,
                scaling_dict=scaling_dict, 
                formation_time=data.get('formation_time', 0.0)
            )

        # --- Configuración final de los ejes y la leyenda (CORREGIDA) ---
        labels = [
            ['Bt [T]', 'Ip [kA]'], 
            ['U_loop [V]', 'ne [m^-3]'], 
            ['Desplazamiento [mm]', 'Te [eV]'], 
            ['τ_e [μs]', 'Evolución Iones [U.A.]']
        ]
        ion_handles, ion_labels = [], []
        for i in range(4):
            for j in range(2):
                ax = self.axs[i, j]
                ax.grid(True, which='both', linestyle='--', linewidth=0.5)
                # Solo mostrar etiquetas del eje X en la última fila
                ax.tick_params(axis='x', labelbottom=(i == 3))
                ax.set_ylabel(labels[i][j])
                
                # --- LÓGICA DE LEYENDA SIMPLIFICADA ---
                if ax.has_data():
                    if (i, j) == (3, 1): # Eje de Iones
                        ion_handles, ion_labels = ax.get_legend_handles_labels()
                        if ax.get_legend():
                            ax.legend().remove()
                    else: # Todos los demás ejes con datos
                        ax.legend(loc='best')
                
                # Asegurar que el eje X esté compartido correctamente
                if not (i == 0 and j == 0):
                    ax.sharex(self.axs[0, 0])
        
        self.axs[3,0].set_xlabel('Tiempo [ms]')
        self.axs[3,1].set_xlabel('Tiempo [ms]')

        self._build_external_ion_legend(ion_handles, ion_labels)
        
        self.canvas.draw()

    def on_hover_ion_axes(self, event):
        """
        Se activa al mover el mouse. Muestra qué línea de ion está
        debajo del cursor.
        """
        # Eje de interés (Evolución de Iones)
        ion_ax = self.axs[3, 1]
        
        # Si el mouse está dentro del eje de iones
        if event.inaxes == ion_ax:
            found_labels = []
            # Revisamos todas las líneas en ese eje
            for line in ion_ax.get_lines():
                # Verificamos si el mouse "contiene" (está sobre) la línea
                contains, _ = line.contains(event)
                if contains:
                    label = line.get_label()
                    # Ignoramos líneas de 'fallback' de matplotlib
                    if not label.startswith('_'):
                        found_labels.append(label)

            if found_labels:
                # Si encuentra líneas (incluso solapadas), las une
                display_text = " | ".join(found_labels)
                self.ion_hover_label.config(text=display_text, foreground="black")
            else:
                # Si está en el eje pero no sobre una línea
                self.ion_hover_label.config(text="Mueve el mouse sobre una línea...", foreground="gray")
        else:
            # Si el mouse está fuera del eje de iones, limpiamos el texto
            self.ion_hover_label.config(text="", foreground="gray")
    def _build_external_ion_legend(self, handles, labels):
        """
        Limpia y reconstruye la leyenda de iones en el panel lateral (self.ion_legend_frame).
        """
        # 1. Limpiar leyenda anterior
        for widget in self.ion_legend_frame.winfo_children():
            widget.destroy()

        if not handles:
            return # No hay nada que dibujar

        # 2. Añadir un título
        tk.Label(
            self.ion_legend_frame, 
            text="Leyenda de Iones", 
            font=("Arial", 9, "bold"), 
            bg='white'
        ).pack(anchor='w')

        # 3. Crear la leyenda línea por línea
        num_cols = max(1, len(handles) // 10) # Poner en 2 columnas si es muy larga
        col_frame = None
        
        for i, (handle, label) in enumerate(zip(handles, labels)):
            # Crear un nuevo frame de columna si es necesario
            if i % 10 == 0:
                col_frame = tk.Frame(self.ion_legend_frame, bg='white')
                col_frame.pack(side=tk.LEFT, fill=tk.X, anchor='n', padx=5)

            color = handle.get_color()
            
            # Contenedor para una línea de leyenda
            line_f = tk.Frame(col_frame, bg='white')
            
            # El símbolo de línea
            tk.Label(
                line_f, text='—', 
                fg=color, 
                bg='white', 
                font=('Courier New', 10, 'bold')
            ).pack(side=tk.LEFT)
            
            # El texto
            tk.Label(
                line_f, 
                text=label, 
                bg='white', 
                font=("Courier New", 8)
            ).pack(side=tk.LEFT, padx=4)
            
            line_f.pack(anchor='w')
    def clear_all_shots(self): # <--- Nombre cambiado
        self.shots.clear()
        self.current_shot = None
        self.image_refs.clear() # Esto funciona igual para un diccionario
        if self.ion_sidebar_panel and tk.Toplevel.winfo_exists(self.ion_sidebar_panel):
            self.ion_sidebar_panel.destroy()
        for widget in self.png_frame.winfo_children():
            widget.destroy()
        self.plot_data()


    def remove_shot(self):
        """
        Abre un diálogo para eliminar un disparo específico o todos los disparos.
        """
        # 1. Comprobar si hay disparos que eliminar
        if not self.shots:
            messagebox.showinfo("Sin Datos", "No hay disparos cargados para eliminar.")
            return

        # 2. Preguntar al usuario qué disparo eliminar, con la opción de borrar todos
        shot_to_remove = simpledialog.askinteger(
            "Remove Shot(s)",
            "Introduce el número del disparo a eliminar.\n\n(O introduce 0 para eliminarlos TODOS)",
            parent=self.root
        )

        # Si el usuario cancela, el valor es None
        if shot_to_remove is None:
            return 

        # 3. Si el usuario introduce 0, llamar a la función que borra todo
        if shot_to_remove == 0:
            self.clear_all_shots() # Reutilizamos el método que ya existe
            return

        # --- De aquí en adelante, es la misma lógica para borrar un solo disparo ---
        
        if shot_to_remove not in self.shots:
            messagebox.showerror("Error", f"El disparo #{shot_to_remove} no está cargado en la sesión.")
            return

        # Eliminar los datos del disparo
        print(f"Eliminando disparo #{shot_to_remove}...")
        del self.shots[shot_to_remove]

        # Eliminar la imagen y su referencia
        if shot_to_remove in self.image_refs:
            del self.image_refs[shot_to_remove]
        
        for widget in self.png_frame.winfo_children():
            if hasattr(widget, "shot_number") and widget.shot_number == shot_to_remove:
                widget.destroy()
                break

        # Actualizar el estado de la aplicación
        if self.current_shot == shot_to_remove:
            self.current_shot = None
        
        # Actualizar el panel de iones si está abierto
        if self.ion_sidebar_panel and tk.Toplevel.winfo_exists(self.ion_sidebar_panel):
            new_ions_for_panel = {shot: data.get('shot_ions_data', []) for shot, data in self.shots.items()}
            if not any(new_ions_for_panel.values()):
                self.ion_sidebar_panel.destroy()
            else:
                self.ion_sidebar_panel.update_ions(new_ions_for_panel)

        # Volver a dibujar las gráficas
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
        if not self.shots: return
        current_shot_number = self.current_shot
        if not current_shot_number: return

        try:
            new_threshold = int(self.peak_threshold_var.get())
            if new_threshold <= 0: return
            self.spec_peak_height = new_threshold
            print(f"Nuevo umbral de picos: {self.spec_peak_height}")

            # --- LÓGICA DE ACTUALIZACIÓN EN TIEMPO REAL ---
            # Actualizamos para todos los disparos cargados que tengan datos de espectrómetro
            for shot_num, shot_data in self.shots.items():
                h5_path = shot_data.get('h5_path')
                if h5_path and self.nist_df is not None:
                    # 1. Volver a detectar iones con el nuevo umbral (USANDO EL NOMBRE CORRECTO)
                    ions, wls = spectrometry_analyzer._detect_main_ions_for_panel(
                        h5_path, self.nist_df, peak_height=self.spec_peak_height
                    )
                    detected_ions = list(zip(ions, wls))
                    
                    # 2. Actualizar la lista de iones del disparo actual
                    shot_data['shot_ions_data'] = detected_ions
            
            # 3. Si el panel de iones está abierto, actualizarlo en lugar de destruirlo
            if self.ion_sidebar_panel and tk.Toplevel.winfo_exists(self.ion_sidebar_panel):
                new_ions_for_panel = {shot: data.get('shot_ions_data', []) for shot, data in self.shots.items()}
                self.ion_sidebar_panel.update_ions(new_ions_for_panel) # Llamamos al nuevo método
                print("Panel de iones actualizado dinámicamente.")
            
            # 4. Volver a dibujar todas las gráficas
            self.plot_data()

        except ValueError:
            self.peak_threshold_var.set(str(self.spec_peak_height)) # Revertir si no es un número válido



    def visualize_spectrum_peaks(self):
        """
        Abre una ventana con un slider para analizar el espectro y los picos
        detectados en cualquier frame del disparo. Permite seleccionar el disparo.
        """
        # --- INICIO DE LA CORRECCIÓN 1: SELECCIÓN DE DISPARO ---
        if not self.shots:
            messagebox.showwarning("Sin datos", "Carga un disparo antes de visualizar los picos.")
            return

        shot_list = list(self.shots.keys())
        shot_to_analyze = None

        if len(shot_list) == 1:
            shot_to_analyze = shot_list[0]
        else:
            # Preguntar al usuario si hay múltiples disparos
            shot_to_analyze = simpledialog.askinteger(
                "Seleccionar Disparo",
                "Introduce el número del disparo para analizar los picos:",
                parent=self.root,
                initialvalue=self.current_shot if self.current_shot else shot_list[0]
            )

        if shot_to_analyze is None: # El usuario canceló
            return
        
        if shot_to_analyze not in self.shots:
            messagebox.showerror("Error", f"El disparo #{shot_to_analyze} no está cargado.")
            return

        shot_data = self.shots[shot_to_analyze]
        h5_path = shot_data.get('h5_path')
        # --- FIN DE LA CORRECCIÓN 1 ---

        if not h5_path or not os.path.exists(h5_path):
            messagebox.showerror("Error", f"No se encontró el archivo .h5 para el disparo {shot_to_analyze}.")
            return

        try:
            # --- INICIO DE LA CORRECCIÓN 2: CARGAR VECTOR DE TIEMPO REAL ---
            with h5py.File(h5_path, 'r') as f:
                all_wl, all_spectra = f['Wavelengths'][:], f['Spectra'][:]
                
                # Cargar el vector de tiempo REAL del archivo H5
                if 'Time' in f:
                    # Asumir que el tiempo está en SEGUNDOS y convertir a ms
                    time_vector_ms = f['Time'][:] * 1000.0
                else:
                    # Fallback si no existe (usando 2.0ms como pediste, aunque esto es menos preciso)
                    print("ADVERTENCIA (Picos): No se encontró 'Time' en H5. Usando time_step=2.0ms.")
                    time_vector_ms = np.arange(all_spectra.shape[0]) * 2.0 
            # --- FIN DE LA CORRECCIÓN 2 ---
            
            total_frames = all_spectra.shape[0]
            ref_idx = np.argmax(np.sum(all_spectra, axis=1)) # Frame inicial sugerido

            # --- Creación de la Ventana y Layout (con título actualizado) ---
            peak_window = tk.Toplevel(self.root)
            peak_window.title(f"Análisis de Picos - Disparo #{shot_to_analyze}") # Título actualizado
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
                
                # --- CORRECCIÓN 2: USAR EL TIEMPO REAL ---
                # Asegurarse de que el frame_idx no esté fuera de los límites del vector de tiempo
                if frame_idx >= len(time_vector_ms):
                    frame_idx = len(time_vector_ms) - 1
                current_time_ms = time_vector_ms[frame_idx]
                
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

                # --- CORRECCIÓN 2: USAR EL TIEMPO REAL EN EL TÍTULO ---
                ax.set_title(f"Análisis del Frame: {frame_idx} (Tiempo: {2.0*int(frame_idx):.1f} ms)")
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
        if x is None: return
        self.last_cursor_x = x
        
        # Usamos un bloque try/except para borrar las líneas de forma segura
        # Esto evita que el programa se caiga si una línea ya no existe
        for line in self.cursor_lines:
            try:
                line.remove()
            except Exception:
                pass # Ignorar errores si la línea ya no existe o está huérfana
        
        self.cursor_lines.clear()

        for ax_row in self.axs:
            for ax in ax_row:
                self.cursor_lines.append(ax.axvline(x=x, color='gray', linestyle='--', linewidth=0.8))

        # (El resto de tu código para el data_box_label va aquí, no cambia)
        # 1. Definimos anchos fijos para cada columna
        w_shot = 8
        w_time = 10
        w_bt = 8
        w_ip = 8
        w_ne = 12
        w_te = 10
        w_tau = 10

        # 2. Cabeceras (todas alineadas a la izquierda '<')
        header = (f"{'Shot':<{w_shot}}"
                  f"{'Time(ms)':<{w_time}}"
                  f"{'Bt(T)':<{w_bt}}"
                  f"{'Ip(kA)':<{w_ip}}"
                  f"{'ne(m-3)':<{w_ne}}"
                  f"{'Te_0(eV)':<{w_te}}"
                  f"{'tau_e(us)':<{w_tau}}")
        
        data_table = [header]
        for shot, data in self.shots.items():
            vals = {}
            for key, df in data.items():
                if isinstance(df, pd.DataFrame) and 'time_ms' in df.columns and not df.empty:
                    idx = (df['time_ms'] - x).abs().idxmin()
                    for col in df.columns:
                        if col != 'time_ms': vals[col] = df.loc[idx, col]
            
            # 3. Formatear los números a strings primero
            s_shot = f"{shot}"
            s_time = f"{x:.2f}"
            s_bt   = f"{vals.get('Bt', np.nan):.2f}"
            s_ip   = f"{vals.get('Ip', np.nan):.2f}"
            s_ne   = f"{vals.get('ne', np.nan):.2e}"
            s_te   = f"{vals.get('Te_0', np.nan):.1f}"
            s_tau  = f"{vals.get('tau', np.nan) * 1e6:.1f}"

            # 4. Reemplazar 'nan' por '---' en los strings
            s_bt = s_bt.replace("nan", "---")
            s_ip = s_ip.replace("nan", "---")
            s_ne = s_ne.replace("nan", "---")
            s_te = s_te.replace("nan", "---")
            s_tau = s_tau.replace("nan", "---")

            # 5. Formatear los strings finales con alineación izquierda '<'
            row = (f"{s_shot:<{w_shot}}"
                   f"{s_time:<{w_time}}"
                   f"{s_bt:<{w_bt}}"
                   f"{s_ip:<{w_ip}}"
                   f"{s_ne:<{w_ne}}"
                   f"{s_te:<{w_te}}"
                   f"{s_tau:<{w_tau}}")
            
            data_table.append(row)

        self.data_box_label.config(text="\n".join(data_table))
        self.canvas.draw_idle()

# En la clase TokamakDataViewer (ui/main_window.py)

    def on_right_click(self, event):
        if not event.inaxes or not self.cursor_dynamics_enabled or event.button != 3: return
        x = event.xdata
        
        # Usamos tabulaciones ('\t') aquí porque es para pegar en Excel/Hojas de Cálculo
        # El problema anterior era solo de VISUALIZACIÓN en el Label.
        # Para el portapapeles, las tabulaciones son lo correcto.
        # PERO, el error de 'nan' debe corregirse.

        clipboard_text = "Shot\tTime(ms)\tBt(T)\tIp(kA)\tne(m-3)\tTe_0(eV)\ttau_e(us)\n"
        for shot, data in self.shots.items():
            vals = {}
            for key, df in data.items():
                if isinstance(df, pd.DataFrame) and 'time_ms' in df.columns and not df.empty:
                    idx = (df['time_ms'] - x).abs().idxmin()
                    for col in df.columns:
                        if col != 'time_ms': vals[col] = df.loc[idx, col]
            
            # Creamos los valores primero
            val_bt = f"{vals.get('Bt', np.nan):.2f}"
            val_ip = f"{vals.get('Ip', np.nan):.2f}"
            val_ne = f"{vals.get('ne', np.nan):.2e}"
            val_te = f"{vals.get('Te_0', np.nan):.1f}"
            val_tau = f"{vals.get('tau', np.nan) * 1e6:.1f}"

            # Construimos la fila
            row = (f"{shot}\t{x:.2f}\t"
                   f"{val_bt}\t{val_ip}\t"
                   f"{val_ne}\t{val_te}\t"
                   f"{val_tau}\n")
            
            # Reemplazamos 'nan' por una cadena vacía para limpiar la salida
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