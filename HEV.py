import numpy as np
import pandas as pd
from scipy.interpolate import interp1d, interp2d
from scipy.stats import linregress
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
SOC_init = 0.8  # Initial state of charge
SOC_min, SOC_max = 0.2, 0.8  # SOC bounds
S = 30  # sun gear
R = 78  # ring gear
K = 4.113  # final drive

# Load the data
ftpcol = 'ftpcol.csv'
df_speed = pd.read_csv(ftpcol)
engine_parameters = 'HEVPowerSplitData.csv'
df_engine = pd.read_csv(engine_parameters)

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

eta = 0.85
etam = np.full_like(df_speed['Tg (Nm)'], eta)
etag = np.full_like(df_speed['Tg (Nm)'], eta)
etam[(df_speed['Tm (Nm)'] * df_speed['Wm (rad/s)']) >= 0] = 1 / eta
etag[(df_speed['Tg (Nm)'] * df_speed['Wg (rad/s)']) >= 0] = 1 / eta

Pveh = 20.3*1000
df_speed['Pbatt (W)'] = (etam * df_speed['Tm (Nm)'] * df_speed['Wm (rad/s)'] + etag * df_speed['Tg (Nm)'] * df_speed['Wg (rad/s)'])
df_speed['Peng (W)'] = Pveh - df_speed['Pbatt (W)']

df_speed['SOC'] = -((Voc - np.sqrt(Voc**2 - 4 * R_batt * df_speed['Pbatt (W)'])) / (2 * R_batt * Q_batt))

# Engine fuel map data
enginemap_spd = np.array([1000, 1250, 1500, 1750, 2000, 2250, 2500, 2750, 3000, 3250, 3500, 4000]) * 2 * np.pi / 60
lbft2Nm = 1.356
enginemap_trq = np.array([6.3, 12.5, 18.8, 25.1, 31.3, 37.6, 43.9, 50.1, 56.4, 62.7, 68.9, 75.2]) * lbft2Nm * 20 / 14
enginemap = np.array([
    [0.1513, 0.1984, 0.2455, 0.2925, 0.3396, 0.3867, 0.4338, 0.4808, 0.5279, 0.5279, 0.5279, 0.5279],
    [0.1834, 0.2423, 0.3011, 0.3599, 0.4188, 0.4776, 0.5365, 0.5953, 0.6541, 0.6689, 0.6689, 0.6689],
    [0.2145, 0.2851, 0.3557, 0.4263, 0.4969, 0.5675, 0.6381, 0.7087, 0.7793, 0.8146, 0.8146, 0.8146],
    [0.2451, 0.3274, 0.4098, 0.4922, 0.5746, 0.6570, 0.7393, 0.8217, 0.9041, 0.9659, 0.9659, 0.9659],
    [0.2759, 0.3700, 0.4642, 0.5583, 0.6525, 0.7466, 0.8408, 0.9349, 1.0291, 1.1232, 1.1232, 1.1232],
    [0.3076, 0.4135, 0.5194, 0.6253, 0.7312, 0.8371, 0.9430, 1.0490, 1.1549, 1.2608, 1.2873, 1.2873],
    [0.3407, 0.4584, 0.5761, 0.6937, 0.8114, 0.9291, 1.0468, 1.1645, 1.2822, 1.3998, 1.4587, 1.4587],
    [0.3773, 0.5068, 0.6362, 0.7657, 0.8951, 1.0246, 1.1540, 1.2835, 1.4129, 1.5424, 1.6395, 1.6395],
    [0.4200, 0.5612, 0.7024, 0.8436, 0.9849, 1.1261, 1.2673, 1.4085, 1.5497, 1.6910, 1.8322, 1.8322],
    [0.4701, 0.6231, 0.7761, 0.9290, 1.0820, 1.2350, 1.3880, 1.5410, 1.6940, 1.8470, 1.9999, 2.0382],
    [0.5290, 0.6938, 0.8585, 1.0233, 1.1880, 1.3528, 1.5175, 1.6823, 1.8470, 2.0118, 2.1766, 2.2589],
    [0.6789, 0.8672, 1.0555, 1.2438, 1.4321, 1.6204, 1.8087, 1.9970, 2.1852, 2.3735, 2.5618, 2.7501]
])

# Compute power and g/kWh
T, w = np.meshgrid(enginemap_trq, enginemap_spd)
enginemap_kW = T * w / 1000
enginemap_gpkWh = enginemap / enginemap_kW * 3600

fuel_map_gps = (enginemap_gpkWh * enginemap_kW * 1000) / 3600

# Max torque line data
MaxTq_pt = np.array([enginemap_trq[0], 77.2920, 82.0380, 84.7500, 86.7840, 89.3604,
                     91.1232, 92.8860, 94.6488, 96.4116, 98.1744, 99.9372, 101.9712])
MaxSp_pt = np.array([1000, 1010, 1250, 1500, 1750, 2000, 2250, 2500, 2750, 3000, 3250, 3500, 4000])

# Create a finer grid for smooth background
fine_spd = np.linspace(enginemap_spd.min(), enginemap_spd.max(), 100)  # Fine grid for engine speed
fine_trq = np.linspace(enginemap_trq.min(), enginemap_trq.max(), 100)  # Fine grid for torque

# Interpolate the fuel map to the finer grid
from scipy.interpolate import interp2d
interp_func = interp2d(enginemap_trq, enginemap_spd, enginemap, kind='linear')  # Interpolation function
fine_fuel_map = interp_func(fine_trq, fine_spd)  # Interpolated data

# Resample max torque line
MaxSp = np.linspace(1000, 4000, 300)  # 1 RPM step
MaxTq = interp1d(MaxSp_pt, MaxTq_pt, kind='linear')(MaxSp)
MaxSp_radsec = MaxSp * 2 * np.pi / 60  # Convert to rad/s

radsec2rpm = 60 / (2 * np.pi)

# Extract 10 equally spaced points
num_points = 10
selected_speeds = np.linspace(MaxSp.min(), MaxSp.max(), num_points)  # Speeds in RPM
selected_torques = interp1d(MaxSp, MaxTq, kind='linear')(selected_speeds)
selected_speeds_radsec = selected_speeds * (2 * np.pi / 60)  # Convert to rad/s

# Fuel consumption interpolation with realistic scaling
interp_fuel_func = interp2d(enginemap_trq, enginemap_spd, fuel_map_gps, kind='linear')
results = []
for speed_radsec, torque in zip(selected_speeds_radsec, selected_torques):
    fuel_rate = interp_fuel_func(torque, speed_radsec) / 1000  # Scale down by 1000 for realistic units
    results.append((speed_radsec, torque, float(fuel_rate)))

# Create DataFrame for results
df_results = pd.DataFrame(results, columns=["Speed (rad/s)", "Torque (Nm)", "Fuel Rate (g/s)"])
print(df_results)

# Create the contour plot
fig = go.Figure()

# Add contour lines for the fuel map
fig.add_trace(go.Contour(
    z=fine_fuel_map,
    x=fine_spd * radsec2rpm,  # Convert back to RPM
    y=fine_trq,  # Torque
    colorscale='Viridis',
    colorbar=dict(title="Fuel Consumption (g/s)"),
    contours=dict(
        coloring="lines",  # Show only boundary lines
        showlabels=True,  # Show labels on lines
        labelfont=dict(size=10, color="black")
    )
))

# Add the maximum torque line
fig.add_trace(go.Scatter(
    x=MaxSp,  # Maximum engine speed in RPM
    y=MaxTq,  # Maximum torque
    mode='lines',
    line=dict(color='red', width=2),
    name='Max Torque Line'
))

fig.add_trace(go.Scatter(
    x=selected_speeds,
    y=selected_torques,
    mode='markers',
    marker=dict(color='red', size=8),
    name='Selected Points'
))

# Layout adjustments
fig.update_layout(
    title="Engine Fuel Map with Full Background",
    xaxis=dict(title="We (RPM)", range=[1000, 4000]),
    yaxis=dict(title="Te (Nm)", range=[10, 105]),
    template="plotly_white"
)

fig.show()

# Constants
Pveh = 20.3 * 1000  # Vehicle power in W

# Calculate Peng and Pbatt for each point and add them as new columns
df_results["Peng (W)"] = df_results["Torque (Nm)"] * df_results["Speed (rad/s)"]  # Peng in W
df_results["Pbatt (W)"] = Pveh - df_results["Peng (W)"]  # Pbatt in W
df_results['Pbatt (kW)'] = df_results['Pbatt (W)'] / 1000
df_results['Peng (kW)'] = df_results['Peng (W)'] / 1000

# Create the scatter plot
minf_Vs_Pbatt = go.Figure()

minf_Vs_Pbatt.add_trace(go.Scatter(
    x=df_results['Pbatt (kW)'],  # Pbatt in kW as x-axis
    y=df_results['Fuel Rate (g/s)'],  # Fuel rate as y-axis
    mode='markers+lines',  # Markers and lines
    marker=dict(color='blue', size=8),
    line=dict(color='blue'),
    name='Fuel Rate vs Pbatt'
))

# Update layout
minf_Vs_Pbatt.update_layout(
    title="Minimum Fuel Rate vs Battery Power",
    xaxis=dict(title="Pbatt (kW)"),
    yaxis=dict(title="Fuel Rate (g/s)"),
    template="plotly_white"
)

# Show the plot
minf_Vs_Pbatt.show()

# Assuming df_results contains the data with columns "Pbatt (kW)" and "Fuel Rate (g/s)"
x = df_results['Pbatt (kW)']  # Battery power in kW (x-axis)
y = df_results['Fuel Rate (g/s)']  # Minimum fuel rate in g/s (y-axis)

# Perform linear regression
slope, intercept, r_value, p_value, std_err = linregress(x, y)

# Print the results
print(f"Slope (a): {slope:.4f}")  # a = slope
print(f"Intercept (b): {intercept:.4f}")  # b = intercept

# Compute the total summation (Delta SOC)
delta_SOC = df_speed['SOC'].sum()

# Display the result
print(f"Delta SOC: {delta_SOC}")
