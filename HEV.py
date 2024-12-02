import numpy as np
from scipy.interpolate import RectBivariateSpline
import plotly.graph_objects as go

# Constants
vehicle_mass = 2000  # kg
drag_coefficient = 0.25
frontal_area = 2  # m^2
air_density = 1.2  # kg/m^3
rolling_friction = 0.015
gravity = 9.8  # m/s^2
tire_radius = 0.3  # m

# HEV-specific parameters
Voc = 201.6  # V
R_batt = 0.504  # Ω
Q_batt = 6.5 * 3600  # A·s
final_drive = 4.113
sun_gear = 30
ring_gear = 78
gear_ratio = ring_gear / sun_gear
SOC_init = 0.8  # Initial state of charge

# Fuel map data (example placeholders)
engine_speeds = np.linspace(1000, 4000, 12) * 2 * np.pi / 60  # rad/s
engine_torques = np.linspace(0, 75, 12)  # Nm
fuel_map = np.random.rand(12, 12)  # Replace with actual map

# Interpolator for fuel map
fuel_map_interp = RectBivariateSpline(engine_speeds, engine_torques, fuel_map)

# Load FTP 75 data (time, speed)
ftp_data = np.loadtxt('HEV.txt', skiprows=1)
time = ftp_data[:, 0]
speed = ftp_data[:, 1] * 0.44704  # Convert mph to m/s

# Calculate acceleration
acceleration = np.gradient(speed, time)

# Initialize variables
SOC = np.full_like(time, SOC_init)
T_engine = np.zeros_like(time)
T_motor = np.zeros_like(time)
T_generator = np.zeros_like(time)
omega_engine = np.zeros_like(time)
omega_motor = np.zeros_like(time)
omega_generator = np.zeros_like(time)
P_battery = np.zeros_like(time)
fuel_consumption = np.zeros_like(time)

# Helper functions
def calculate_resistances(speed, acceleration):
    F_aero = 0.5 * air_density * drag_coefficient * frontal_area * speed**2
    F_roll = rolling_friction * vehicle_mass * gravity
    F_grade = vehicle_mass * gravity * np.sin(5 * np.sin(0.01 * speed))  # Grade in radians
    F_inertia = vehicle_mass * acceleration
    return F_aero + F_roll + F_grade + F_inertia

def calculate_battery_power(T_motor, omega_motor, T_generator, omega_generator):
    P_motor = T_motor * omega_motor
    P_generator = T_generator * omega_generator
    return P_motor - P_generator

# Main loop
lambda_costate = 0.1  # Example constant value for simplicity
for i in range(1, len(time)):
    dt = time[i] - time[i - 1]
    T_required = calculate_resistances(speed[i], acceleration[i]) * tire_radius
    omega_output = speed[i] / tire_radius

    # Optimization logic (simplified)
    T_engine[i] = T_required / 2
    T_motor[i] = T_required / 2
    T_generator[i] = T_required - T_engine[i] - T_motor[i]
    omega_engine[i] = omega_output
    omega_motor[i] = omega_output - omega_generator[i]
    omega_generator[i] = omega_engine[i] / gear_ratio

    # Update SOC and fuel consumption
    P_battery[i] = calculate_battery_power(T_motor[i], omega_motor[i], T_generator[i], omega_generator[i])
    SOC[i] = SOC[i - 1] + (-P_battery[i] / (Voc * Q_batt)) * dt
    fuel_consumption[i] = fuel_map_interp(omega_engine[i], T_engine[i])[0] * dt

# Plot SOC
fig_SOC = go.Figure()
fig_SOC.add_trace(go.Scatter(x=time, y=SOC, mode='lines', name='SOC'))
fig_SOC.update_layout(
    title='State of Charge (SOC) Over Time',
    xaxis_title='Time (s)',
    yaxis_title='SOC',
    template='plotly'
)

# Plot Cumulative Fuel Consumption
fig_fuel = go.Figure()
fig_fuel.add_trace(go.Scatter(x=time, y=fuel_consumption.cumsum(), mode='lines', name='Cumulative Fuel Consumption'))
fig_fuel.update_layout(
    title='Cumulative Fuel Consumption Over Time',
    xaxis_title='Time (s)',
    yaxis_title='Fuel Consumed (g)',
    template='plotly'
)

# Plot Component Speeds
fig_speeds = go.Figure()
fig_speeds.add_trace(go.Scatter(x=time, y=omega_engine, mode='lines', name='Engine Speed (rad/s)', line=dict(color='green')))
fig_speeds.add_trace(go.Scatter(x=time, y=omega_motor, mode='lines', name='Motor Speed (rad/s)', line=dict(color='orange')))
fig_speeds.add_trace(go.Scatter(x=time, y=omega_generator, mode='lines', name='Generator Speed (rad/s)', line=dict(color='purple')))
fig_speeds.update_layout(
    title='Component Speeds Over Time',
    xaxis_title='Time (s)',
    yaxis_title='Speed (rad/s)',
    template='plotly'
)

# Plot Component Torques
fig_torques = go.Figure()
fig_torques.add_trace(go.Scatter(x=time, y=T_engine, mode='lines', name='Engine Torque (Nm)', line=dict(color='green')))
fig_torques.add_trace(go.Scatter(x=time, y=T_motor, mode='lines', name='Motor Torque (Nm)', line=dict(color='orange')))
fig_torques.add_trace(go.Scatter(x=time, y=T_generator, mode='lines', name='Generator Torque (Nm)', line=dict(color='purple')))
fig_torques.update_layout(
    title='Component Torques Over Time',
    xaxis_title='Time (s)',
    yaxis_title='Torque (Nm)',
    template='plotly'
)

# Show plots
fig_SOC.show()
fig_fuel.show()
fig_speeds.show()
fig_torques.show()
