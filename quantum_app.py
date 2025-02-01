import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from quantum_system import EarthCalculator, EarthConstants

class QuantumSystemApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Quantum System Calculator")
        self.calculator = EarthCalculator()
        self.constants = EarthConstants()
        
        # Create main frame
        self.main_frame = ttk.Frame(root, padding="10")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Create input fields
        self.create_inputs()
        
        # Create plot
        self.create_plot()
        
        # Create calculate button
        self.create_buttons()
        
        # Create results display
        self.create_results_display()

    def create_inputs(self):
        # Mass input
        ttk.Label(self.main_frame, text="Mass (kg):").grid(row=0, column=0, padx=5, pady=5)
        self.mass_var = tk.StringVar(value="1.0")
        ttk.Entry(self.main_frame, textvariable=self.mass_var).grid(row=0, column=1, padx=5, pady=5)
        
        # Altitude input
        ttk.Label(self.main_frame, text="Altitude (km):").grid(row=1, column=0, padx=5, pady=5)
        self.altitude_var = tk.StringVar(value="400")
        ttk.Entry(self.main_frame, textvariable=self.altitude_var).grid(row=1, column=1, padx=5, pady=5)

    def create_plot(self):
        self.fig, self.ax = plt.subplots(figsize=(6, 4))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.main_frame)
        self.canvas.get_tk_widget().grid(row=3, column=0, columnspan=2, padx=5, pady=5)

    def create_buttons(self):
        ttk.Button(self.main_frame, text="Calculate", command=self.calculate).grid(row=2, column=0, columnspan=2, pady=10)

    def create_results_display(self):
        self.result_text = tk.Text(self.main_frame, height=4, width=40)
        self.result_text.grid(row=4, column=0, columnspan=2, padx=5, pady=5)

    def calculate(self):
        try:
            mass = float(self.mass_var.get())
            altitude = float(self.altitude_var.get()) * 1000  # Convert km to m
            
            # Calculate values
            force = self.calculator.gravitational_force(mass, self.constants.RADIUS + altitude)
            velocity = self.calculator.orbital_velocity(altitude)
            
            # Update results display
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, 
                f"Gravitational Force: {force:.2f} N\n"
                f"Orbital Velocity: {velocity:.2f} m/s\n"
                f"Escape Velocity: {self.constants.ESCAPE_VELOCITY:.2f} m/s\n")
            
            # Update plot
            self.update_plot(altitude)
            
        except ValueError:
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, "Please enter valid numbers")

    def update_plot(self, altitude):
        self.ax.clear()
        
        # Create data for plot
        altitudes = np.linspace(0, altitude * 2, 100)
        velocities = [self.calculator.orbital_velocity(alt) for alt in altitudes]
        
        # Plot
        self.ax.plot(altitudes/1000, velocities, 'b-', label='Orbital Velocity')
        self.ax.axvline(x=altitude/1000, color='r', linestyle='--', label='Current Altitude')
        
        self.ax.set_xlabel('Altitude (km)')
        self.ax.set_ylabel('Velocity (m/s)')
        self.ax.set_title('Orbital Velocity vs Altitude')
        self.ax.legend()
        self.ax.grid(True)
        
        self.canvas.draw()

def main():
    root = tk.Tk()
    app = QuantumSystemApp(root)
    root.mainloop()

if __name__ == "__main__":
    main() 