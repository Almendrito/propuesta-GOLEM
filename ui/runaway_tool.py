# ui/runaway_tool.py

import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import pandas as pd
import requests
import re
import io
import threading
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from lmfit import Parameters, minimize
import matplotlib
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.ticker import AutoMinorLocator

matplotlib.use("TkAgg")

class RunawayAppToplevel:
    def __init__(self, root, shot_number):
        self.root = root
        self.shot_no = shot_number
        self.root.title(f"Análisis de Runaway Electrons (HXR) - Disparo {self.shot_no}")
        self.root.geometry("1000x800")

        # --- Variables de control ---
        self.status_var = tk.StringVar(value="Presiona 'Ejecutar Análisis' para descargar datos del osciloscopio...")

        # --- UI Layout ---
        top_frame = tk.Frame(self.root, pady=10)
        top_frame.pack(side=tk.TOP, fill=tk.X)

        tk.Label(top_frame, text=f"Disparo GOLEM: {self.shot_no}", font=("Arial", 12, "bold")).pack(side=tk.LEFT, padx=10)
        
        self.btn_run = tk.Button(top_frame, text="Ejecutar Análisis HXR", command=self.on_run_analysis, bg="#d9edf7")
        self.btn_run.pack(side=tk.LEFT, padx=10)

        status_bar = tk.Label(self.root, textvariable=self.status_var, bd=1, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        self.canvas_frame = tk.Frame(self.root)
        self.canvas_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def on_run_analysis(self):
        self.btn_run.config(state=tk.DISABLED)
        self.status_var.set("Descargando datos del osciloscopio (puede tardar un momento)...")
        threading.Thread(target=self.run_analysis_logic).start()

    def run_analysis_logic(self):
        try:
            # 1. Descargar datos del osciloscopio (Scintillator)
            url_ch3 = f'http://golem.fjfi.cvut.cz/shots/{self.shot_no}/Devices/Oscilloscopes/TektrMSO58-a/ch3.csv'
            response = requests.get(url_ch3, timeout=30)
            response.raise_for_status()
            df = pd.read_csv(io.StringIO(response.text), header=None, names=["time", "voltage"])
            volts = df["voltage"].values
            time = df["time"].values

            # 2. Descargar Corriente de Plasma (Ip)
            url_ip = f"http://golem.fjfi.cvut.cz/shots/{self.shot_no}/Diagnostics/BasicDiagnostics/Results/Ip.csv"
            resp_ip = requests.get(url_ip, timeout=10)
            if resp_ip.status_code == 200:
                ip_data = pd.read_csv(io.StringIO(resp_ip.text), header=None, names=['time_ms', 'Ip'])
            else:
                ip_data = pd.DataFrame(columns=['time_ms', 'Ip'])

            # 3. Extraer U_NIM_A2 del FullCommandLine
            self.root.after(0, lambda: self.status_var.set("Analizando picos de voltaje y ajustando espectro..."))
            url_cmd = f"http://golem.fjfi.cvut.cz/shots/{self.shot_no}/Production/Parameters/FullCommandLine"
            resp_cmd = requests.get(url_cmd, timeout=10)
            u_nim_a2_value = 1000 # Valor por defecto seguro
            if resp_cmd.status_code == 200:
                match = re.search(r"U_NIM_A2=(\d+)", resp_cmd.text)
                if match: u_nim_a2_value = float(match.group(1))

            # 4. Encontrar picos
            max_value = np.max(-volts * 1e3)
            threshold = 0.1 * max_value
            locs, peaks = find_peaks(-volts * 1e3, prominence=threshold, distance=50)
            result = -volts[locs] * 1e3

            # 5. Histograma y Calibración
            amp_bins = np.linspace(0, 250, 70)
            n, edges = np.histogram(result, bins=amp_bins)
            centre = (edges[1:] + edges[:-1]) / 2

            # Conversión a eV
            a, b, c, d, f_const = 1593.78, -6.19274, -0.384572, 0.00572227, 0.00111747
            x = u_nim_a2_value
            def VoltoeV(z): return -(a + b*x + d*x**2 - z) / (c + f_const*x)
            
            eV = VoltoeV(centre)
            
            # Limpiar datos bajos
            minus_bordel = n > 2
            eV = eV[minus_bordel]
            n = n[minus_bordel]
            if len(n) == 0:
                raise ValueError("No se encontraron suficientes conteos válidos para analizar HXR.")

            mask = eV > eV[np.argmax(n)]
            n = n[mask]
            eV = eV[mask]

            # 6. Fit Exponencial (lmfit)
            def fce(params, x_val, y_val):
                A_val, B_val = params['A'], params['B']
                return (A_val * np.exp(B_val * x_val)) - y_val

            params = Parameters()
            params.add('A', min=100, max=1e15)
            params.add('B', min=-0.05, max=0)
            fitted_params = minimize(fce, params, args=(eV, n), method='least_squares')
            A_fit = fitted_params.params['A'].value
            B_fit = fitted_params.params['B'].value

            # Cota inferior de energía máxima
            max_energy = np.log(2 / A_fit) / B_fit if B_fit != 0 else 0

            # Fit para x0 (Promedio Er)
            def exp_fit(x_val, A_val, B_val, x0_val):
                return A_val * np.exp(-B_val * (x_val - x0_val))
            
            p0 = [np.max(n), 1, np.min(eV)]
            popt, _ = curve_fit(exp_fit, eV, n, p0=p0, maxfev=10000)
            _, B_curve, x0_curve = popt
            Er = -1/B_curve + x0_curve

            # Llamamos a GUI para actualizar la gráfica en el main thread
            self.root.after(0, self.plot_results, time, volts, ip_data, eV, n, A_fit, B_fit, u_nim_a2_value, max_energy, Er)

        except Exception as e:
            self.root.after(0, lambda e=e: messagebox.showerror("Error en el análisis", str(e)))
            self.root.after(0, lambda: self.status_var.set("Error durante el análisis."))
        finally:
            self.root.after(0, lambda: self.btn_run.config(state=tk.NORMAL))

    def plot_results(self, time, volts, ip_data, eV, n, A, B, u_nim_a2, max_energy, Er):
        for widget in self.canvas_frame.winfo_children():
            widget.destroy()

        fig = Figure(figsize=(10, 8), dpi=100)
        fig.subplots_adjust(hspace=0.4, top=0.95, bottom=0.08)

        # Plot 1: Voltaje vs Tiempo
        ax0 = fig.add_subplot(311)
        ax0.plot(time*1000, -volts*1000, color='darkred')
        ax0.set_ylabel('V (mV)')
        ax0.set_title("Señal Scintillator CeBr(A)")
        ax0.set_xlim(left=-1, right=20)
        ax0.grid(True, linestyle='--', alpha=0.6)

        # Plot 2: Ip vs Tiempo
        ax1 = fig.add_subplot(312, sharex=ax0)
        if not ip_data.empty:
            ax1.plot(ip_data['time_ms'], ip_data['Ip'], color='blue')
        ax1.set_xlabel('Time (ms)')
        ax1.set_ylabel('Ip (kA)')
        ax1.grid(True, linestyle='--', alpha=0.6)

        # Plot 3: Espectro Exponencial
        ax2 = fig.add_subplot(313)
        ax2.set_xlabel('E [keV]')
        ax2.set_ylabel('N [-]')
        ax2.errorbar(eV, n, fmt='.', yerr=np.sqrt(n), label=f'CeBr(A), $U_d={u_nim_a2:.0f}$', color='purple')
        
        eV_new = np.linspace(eV[0], eV[-1], 1000)
        ax2.plot(eV_new, A * np.exp(B * eV_new), color='blue', 
                 label=f"Lower bound max $E_{{RE}}$: {max_energy:.1f} keV\n$E_r$: {Er:.1f} keV")
        
        ax2.set_yscale('log')
        ax2.set_ylim(bottom=0.9)
        ax2.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax2.yaxis.set_minor_locator(AutoMinorLocator(2))
        ax2.grid(which='major', color='gray', linewidth=0.5, linestyle='solid')
        ax2.grid(which='minor', color='gray', linewidth=0.3, linestyle='dashed')
        
        leg = ax2.legend(loc='best', shadow=True)
        leg.get_frame().set_edgecolor('k')

        # Embeber a Tkinter
        canvas = FigureCanvasTkAgg(fig, master=self.canvas_frame)
        canvas.draw()
        toolbar = NavigationToolbar2Tk(canvas, self.canvas_frame)
        toolbar.update()
        toolbar.pack(side=tk.BOTTOM, fill=tk.X)
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.status_var.set(f"Análisis completado exitosamente. Energía RE Promedio (Er): {Er:.1f} keV")