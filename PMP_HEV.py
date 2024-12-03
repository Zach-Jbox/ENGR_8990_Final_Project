import numpy as np
import pandas as pd
from scipy.interpolate import RectBivariateSpline
from scipy.optimize import minimize_scalar
import plotly.graph_objects as go

# Constants
m = 2000  # kg
Cd = 0.25  # Drag Coefficient
Area = 2  # m^2
p = 1.2  # kg/m^3
f = 0.015
g = 9.8  # m/s^2
reff = 0.3  # m
Voc = 201.6  # Battery open circuit voltage (V)
Q_batt = 6.5 * 3600  # Battery capacity (As)
R_batt = 0.504  # Battery Resistance
Paux = 500  # Auxiliary Power Demand
SOC_init = 0.8  # Initial state of charge
SOC_min, SOC_max = 0.2, 0.8  # SOC bounds
S = 30  # sun gear
R = 78  # ring gear
K = 4.113  # final drive

lambda_costate_init = 0.1  # Initial costate value

# Load the data
txt_file_path = 'ftpcol.csv'
txt_engine_path = 'HEV_output.csv'
df_speed = pd.read_csv(txt_file_path)
df_engine = pd.read_csv(txt_engine_path)

# Speed and acceleration calculations
mph_to_mps = 0.44704
df_speed['Speed (m/s)'] = df_speed['Speed (mph)'] * mph_to_mps
df_speed['Distance (m)'] = df_speed['Speed (m/s)'].cumsum()
df_speed['Grade (%)'] = 5 * np.sin(0.01 * df_speed['Distance (m)'])
df_speed['Grade (radians)'] = np.arctan(df_speed['Grade (%)'] / 100)
df_speed['F_aero (N)'] = 0.5 * p * Cd * Area * df_speed['Speed (m/s)']**2
df_speed['F_roll (N)'] = f * m * g * np.cos(df_speed['Grade (radians)'])
df_speed['F_grade (N)'] = m * g * np.sin(df_speed['Grade (radians)'])
df_speed['Acceleration (m/s^2)'] = df_speed['Speed (m/s)'].diff().fillna(0)
df_speed['F_inertia (N)'] = m * df_speed['Acceleration (m/s^2)']

# Calculate forces
X = df_speed['F_aero (N)'] + df_speed["F_grade (N)"] + df_speed["F_roll (N)"] + df_speed["F_inertia (N)"]
df_speed['Fthrust (N)'] = np.maximum(X, 0)
df_speed['Fbrake (N)'] = np.minimum(X, 0)

# Torque and speed calculations
df_speed['Tv (Nm)'] = ((reff / K) * (df_speed['F_inertia (N)'] + df_speed['F_aero (N)'] +
                                     df_speed['F_roll (N)'] + df_speed['F_grade (N)'] + df_speed['Fbrake (N)']))
df_speed['Tm (Nm)'] = (df_speed['Tv (Nm)'] / K) - (df_engine['Te (Nm)'] / R) * S
df_speed['Tg (Nm)'] = -df_engine['Te (Nm)'] * (S / (S + R))
df_speed['Wm (rad/s)'] = (K / reff) * df_speed['Speed (m/s)']
df_speed['Wg (rad/s)'] = (df_engine['We (rad/s)'] * (R + S) - df_speed['Wm (rad/s)'] * R) / S

total_fuel_consumption = df_engine['FuelConsumption'].sum()
total_time = 1874
fuel_consumption_rate = total_fuel_consumption / total_time


# Define functions
def calculate_battery_power(Pm, SOC):
    Pbatt = Pm + Paux
    Vbatt = Voc - R_batt * (SOC * Q_batt)
    Ibatt = Pbatt / Vbatt
    return Pbatt, Vbatt, Ibatt

def optimal_control(Tm, Wm, lambda_costate):
    def cost(x):
        motor_power = Tm * x * Wm
        _, _, current = calculate_battery_power(motor_power, SOC_init)
        return current + lambda_costate * (motor_power / Voc)
    
    result = minimize_scalar(cost, bounds=(0, 1), method='bounded')
    return result.x

def hamiltonian(fuel_rate, lambda_costate, SOC_dot):
    return fuel_rate + lambda_costate * SOC_dot

def calculate_SOC_dot(Tm, Wm, Tg, Wg, SOC):
    Pm = Tm * Wm
    Pg = Tg * Wg
    Pbatt = Pm + Pg + Paux
    Vbatt = Voc - R_batt * (SOC * Q_batt)
    Ibatt = Pbatt / Vbatt
    SOC_dot = -Ibatt / Q_batt
    return SOC_dot

# Initialize variables
SOC = SOC_init  # Initial SOC
lambda_costate = lambda_costate_init  # Initial costate
SOC_dots = []
SOC_values = [SOC]  # Include the initial SOC value
Hamiltonians = []
Optimal_splits = []
engine_powers = []
motor_powers = []
Pbatt_values = [] 

# Iterate through each time step in the driving cycle
for i in range(len(df_speed)):
    Tm = df_speed.loc[i, 'Tm (Nm)']  # Motor torque
    Wm = df_speed.loc[i, 'Wm (rad/s)']  # Motor speed
    Tg = df_speed.loc[i, 'Tg (Nm)']  # Generator torque
    Wg = df_speed.loc[i, 'Wg (rad/s)']  # Generator speed
    Te = df_engine.loc[i, 'Te (Nm)']  # Engine torque
    We = df_engine.loc[i, 'We (rad/s)']  # Engine speed
    fuel_rate = fuel_consumption_rate  # Fuel consumption rate
    
    # Engine and motor power calculations
    P_e = Te * We  # Engine power
    P_m = Tm * Wm  # Motor power

    engine_powers.append(P_e)
    motor_powers.append(P_m)

    # Calculate Pbatt and SOC_dot for the current step
    Pbatt, _, _ = calculate_battery_power(P_m, SOC_values[-1])
    Pbatt_values.append(Pbatt)  # Store Pbatt for each step
    SOC_dot = calculate_SOC_dot(Tm, Wm, Tg, Wg, SOC)
    SOC_dots.append(SOC_dot)
    
    # Calculate Hamiltonian
    H = hamiltonian(fuel_rate, lambda_costate, SOC_dot)
    Hamiltonians.append(H)

    # Calculate optimal control torque split (if applicable)
    if 'Tm (Nm)' in df_speed.columns and 'Wm (rad/s)' in df_speed.columns:
        optimal_split = optimal_control(Tm, Wm, lambda_costate)
    else:
        optimal_split = None
    Optimal_splits.append(optimal_split)

    # Update SOC using Euler integration
    if i < len(df_speed) - 1:
        dt = 1  # Assuming time step is 1 second; adjust if you have a 'Time' column
        SOC += SOC_dot * dt
        # Enforce SOC bounds
        SOC = max(SOC_min, min(SOC, SOC_max))
        SOC_values.append(SOC)

# Add calculated data to DataFrame
df_speed['SOC_dot'] = SOC_dots
df_speed['SOC'] = SOC_values
df_speed['Hamiltonian'] = Hamiltonians
df_speed['Optimal_Split'] = Optimal_splits
df_speed['Engine Power (W)'] = engine_powers
df_speed['Motor Power (W)'] = motor_powers
df_speed['Pbatt (W)'] = Pbatt_values

# Calculate Power Split Percentage (using absolute values)
total_engine_power = sum(abs(p) for p in engine_powers)  # Total absolute power from engine
total_motor_power = sum(abs(p) for p in motor_powers)  # Total absolute power from motor
total_power = total_engine_power + total_motor_power  # Total combined power

# Ensure no division by zero
if total_power > 0:
    engine_power_percentage = (total_engine_power / total_power) * 100  # Percentage of power from engine
    motor_power_percentage = (total_motor_power / total_power) * 100  # Percentage of power from motor
else:
    engine_power_percentage = 0
    motor_power_percentage = 0

# Torques plot
torques_plot = go.Figure()
torques_plot.add_trace(go.Scatter(x=df_speed.index, y=df_engine['Te (Nm)'], mode='lines', name='Engine Torque'))
torques_plot.add_trace(go.Scatter(x=df_speed.index, y=df_speed['Tm (Nm)'], mode='lines', name='Motor Torque'))
torques_plot.add_trace(go.Scatter(x=df_speed.index, y=df_speed['Tg (Nm)'], mode='lines', name='Generator Torque'))
torques_plot.update_layout(
    title="Torques: Engine, Motor, Generator",
    xaxis_title="Time Steps",
    yaxis_title="Torque (Nm)"
)

# Speeds plot
speeds_plot = go.Figure()
speeds_plot.add_trace(go.Scatter(x=df_speed.index, y=df_engine['We (rad/s)'], mode='lines', name='Engine Speed'))
speeds_plot.add_trace(go.Scatter(x=df_speed.index, y=df_speed['Wm (rad/s)'], mode='lines', name='Motor Speed'))
speeds_plot.add_trace(go.Scatter(x=df_speed.index, y=df_speed['Wg (rad/s)'], mode='lines', name='Generator Speed'))
speeds_plot.update_layout(
    title="Speeds: Engine, Motor, Generator",
    xaxis_title="Time Steps",
    yaxis_title="Speed (rad/s)"
)

# SOC rate of change plot
soc_rate_of_change_plot = go.Figure()
soc_rate_of_change_plot.add_trace(go.Scatter(x=df_speed.index, y=df_speed['SOC_dot'], mode='lines', name='SOC Rate of Change'))
soc_rate_of_change_plot.update_layout(
    title="SOC Rate of Change",
    xaxis_title="Time Steps",
    yaxis_title="Rate of Change (SOC/s)"
)

# Cumulative Fuel Consumption Plot
fuel_consumption_plot = go.Figure()
fuel_consumption_plot.add_trace(go.Scatter(
    x=df_speed.index,  # Using the index as the time steps
    y=np.cumsum(df_engine['FuelConsumption']),  # Cumulative sum of fuel consumption
    mode='lines',
    name='Cumulative Fuel Consumption'
))
fuel_consumption_plot.update_layout(
    title="Cumulative Fuel Consumption",
    xaxis_title="Time Steps",
    yaxis_title="Fuel Consumed (g)"
)

# Battery usage plot
battery_usage_plot = go.Figure()
battery_usage_plot.add_trace(go.Scatter(x=df_speed.index, y=df_speed['SOC'], mode='lines', name='Battery State of Charge (SOC)'))
battery_usage_plot.update_layout(
    title="Battery Usage (State of Charge)",
    xaxis_title="Time Steps",
    yaxis_title="SOC"
)

# Power split plot
power_split_plot = go.Figure()
power_split_plot.add_trace(go.Scatter(
    x=df_speed.index, y=df_speed['Engine Power (W)'], mode='lines', name='Engine Power'
))
power_split_plot.add_trace(go.Scatter(
    x=df_speed.index, y=df_speed['Motor Power (W)'], mode='lines', name='Motor Power'
))
power_split_plot.update_layout(
    title="Power Split: Engine vs. Motor",
    xaxis_title="Time Steps",
    yaxis_title="Power (W)"
)

# Create a Plotly pie chart
power_split_pie_chart = go.Figure(data=[go.Pie(
    labels=["Engine Power Contribution", "Motor Power Contribution"],
    values=[engine_power_percentage, motor_power_percentage],
    hoverinfo="label+percent",
    textinfo="value+percent"
)])

# Customize the layout
power_split_pie_chart.update_layout(
    title="Power Split: Engine vs. Motor Contribution"
)

# Display all figures
torques_plot.show()
speeds_plot.show()
soc_rate_of_change_plot.show()
fuel_consumption_plot.show()
battery_usage_plot.show()
power_split_plot.show()
power_split_pie_chart.show()