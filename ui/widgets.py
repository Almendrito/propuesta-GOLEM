# ui/widgets.py

import tkinter as tk
from tkinter import simpledialog, messagebox
import numpy as np

# En ui/widgets.py

class IonSidebarPanel(tk.Toplevel):
    def __init__(self, parent, shot_ions_dict, on_update, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.title("Control de Iones por Disparo")
        self.transient(parent)
        self.resizable(True, True)
        self.on_update = on_update
        self.ion_vars = {}
        self.scale_vars = {}

        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        self.build_ui(shot_ions_dict)

# En la clase IonSidebarPanel (ui/widgets.py)
# REEMPLAZA ESTE MÉTODO COMPLETO

    def build_ui(self, shot_ions_dict):
        """Construye o reconstruye la interfaz de usuario del panel."""
        
        # 1. Limpiamos los hijos de self (la ventana)
        for widget in self.winfo_children():
            widget.destroy()

        self.shot_ions_dict = shot_ions_dict
        self.ion_vars.clear()
        self.scale_vars.clear()

        # 2. Creamos los widgets como hijos de self (la ventana)
        vscrollbar = tk.Scrollbar(self, orient="vertical")
        hscrollbar = tk.Scrollbar(self, orient="horizontal")

        # 3. Creamos el Canvas como hijo de self
        canvas = tk.Canvas(self, borderwidth=0,
                           yscrollcommand=vscrollbar.set,
                           xscrollcommand=hscrollbar.set)
        
        # 4. Configurar los comandos de las barras
        vscrollbar.config(command=canvas.yview)
        hscrollbar.config(command=canvas.xview)

        # 5. Organizar los widgets en la cuadrícula de self
        canvas.grid(row=0, column=0, sticky="nsew")
        vscrollbar.grid(row=0, column=1, sticky="ns")
        hscrollbar.grid(row=1, column=0, sticky="ew")
        
        # 6. Crear el frame interior (el que se moverá)
        frame = tk.Frame(canvas)
        canvas.create_window((0, 0), window=frame, anchor="nw")
        
        def _on_frame_configure(event):
            canvas.configure(scrollregion=canvas.bbox("all"))
        
        frame.bind("<Configure>", _on_frame_configure)
        
        # (El resto del método es para poner los checkboxes DENTRO del 'frame')

        title = tk.Label(frame, text="Panel de Iones por Disparo", font=("Arial", 12, "bold"))
        title.grid(row=0, column=0, columnspan=max(1, len(shot_ions_dict)), sticky="w", pady=3)

        shot_labels = list(shot_ions_dict.keys())
        for col, shot in enumerate(shot_labels):
            lbl = tk.Label(frame, text=f"Shot #{shot}", font=("Arial", 10, "bold"))
            lbl.grid(row=1, column=col, sticky="n", padx=5)

        all_ion_lines = set()
        for ions_list in shot_ions_dict.values():
            all_ion_lines.update(ions_list)
        all_ion_lines = sorted(list(all_ion_lines), key=lambda x: x[1])

        for row, (ion_name, ion_wl) in enumerate(all_ion_lines, start=2):
            for col, shot in enumerate(shot_labels):
                f = tk.Frame(frame); f.grid(row=row, column=col, sticky="w", padx=5)
                
                ion_exists_in_shot = (ion_name, ion_wl) in shot_ions_dict.get(shot, [])
                
                if ion_exists_in_shot:
                    key = (shot, (ion_name, ion_wl))
                    checkbox_label = f"{ion_name} {ion_wl:.1f} nm"
                    
                    var = tk.BooleanVar(value=True)
                    scale_var = tk.DoubleVar(value=1.0)
                    self.ion_vars[key] = var
                    self.scale_vars[key] = scale_var
                    
                    cb = tk.Checkbutton(f, text=checkbox_label, variable=var, command=self.on_update)
                    cb.pack(side=tk.LEFT)
                    tk.Label(f, text=" x ").pack(side=tk.LEFT)
                    entry = tk.Entry(f, width=5, textvariable=scale_var)
                    entry.pack(side=tk.LEFT)
                    entry.bind("<Return>", lambda e: self.on_update())
                    entry.bind("<FocusOut>", lambda e: self.on_update())
                else:
                    tk.Label(f, text="—", anchor="w").pack(side=tk.LEFT, padx=10)
        
        #tk.Button(frame, text="Actualizar Gráfica", command=self.on_update)\
            #.grid(row=len(all_ion_lines) + 2, column=0, columnspan=max(1, len(shot_labels)), pady=6)

    def update_ions(self, new_shot_ions_dict):
        """
        Método público para reconstruir el panel con una nueva lista de iones.
        """
        self.build_ui(new_shot_ions_dict)
    def get_active_ions_and_scales(self):
        active_ions = {}
        for (shot, ion_tuple), var in self.ion_vars.items():
            if shot not in active_ions: active_ions[shot] = {}
            if var.get():
                try: scale = float(self.scale_vars[(shot, ion_tuple)].get())
                except (ValueError, TypeError): scale = 1.0
                active_ions[shot][ion_tuple] = scale
        return active_ions

class FilterConfigDialog(simpledialog.Dialog):
    def __init__(self, parent, viewer):
        self.viewer = viewer
        super().__init__(parent, title="Configure Savitzky-Golay Filter")

    def body(self, master):
        tk.Label(master, text="Window Length (odd int):").grid(row=0); tk.Label(master, text="Poly. Order (< window):").grid(row=1)
        self.window_entry = tk.Entry(master); self.polyorder_entry = tk.Entry(master)
        self.window_entry.insert(0, str(self.viewer.savgol_window)); self.polyorder_entry.insert(0, str(self.viewer.savgol_polyorder))
        self.window_entry.grid(row=0, column=1); self.polyorder_entry.grid(row=1, column=1)
        return self.window_entry

    def apply(self):
        try:
            window, polyorder = int(self.window_entry.get()), int(self.polyorder_entry.get())
            if window % 2 == 0 or polyorder >= window: raise ValueError
            self.result = (window, polyorder)
        except ValueError:
            messagebox.showerror("Input Inválido", "La ventana debe ser impar y el orden < ventana.")
            self.result = None