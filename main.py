# main.py
import os
import sys
import tkinter as tk


# 1. Obtiene la ruta de la carpeta donde se encuentra main.py (la raíz del proyecto)
project_root = os.path.dirname(os.path.abspath(__file__))

# 2. Añade esa ruta a la lista de lugares donde Python busca módulos
sys.path.insert(0, project_root)

# Ahora esta importación funcionará sin problemas
from ui.main_window import TokamakDataViewer


if __name__ == "__main__":
    """
    Punto de entrada principal para la aplicación Golem Tokamak Data Viewer.
    """
    root = tk.Tk()
    # Le pasamos la ruta base que ya calculamos
    app = TokamakDataViewer(root, base_dir=project_root)
    root.mainloop()