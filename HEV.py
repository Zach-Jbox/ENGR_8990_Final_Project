import numpy as np
import pandas as pd
from scipy.interpolate import interp1d, interp2d
from scipy.interpolate import RegularGridInterpolator
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
Q_batt = 6.5  # Battery capacity (As)
R_batt = 0.504  # Battery Resistance
S = 30  # sun gear
R = 78  # ring gear
K = 4.113  # final drive
dSOC = 0

# Load the data
ftpcol = 'ftpcol.csv'
df_speed = pd.read_csv(ftpcol)

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

# Calculate power required by the vehicle (P_veh) in kilo Watts
df_speed['P_veh (kW)'] = (df_speed['Fthrust (N)'] * df_speed['Speed (m/s)']) / 1000

# Set power to zero where speed is zero to avoid NaN values during idle times
df_speed.loc[df_speed['Speed (m/s)'] == 0, 'P_veh (kW)'] = 0

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

# Create a 2D grid for fine-gridded torque and speed
fine_T, fine_W = np.meshgrid(fine_trq, fine_spd)

# Compute the engine power (in kW) for the fine grid
fine_eng_power = (fine_T * fine_W) / 1000  # Convert W to kW

# Interpolate the fuel map to the finer grid
interp_func = RegularGridInterpolator((enginemap_spd, enginemap_trq), enginemap, method='linear')
fine_points = np.array(np.meshgrid(fine_spd, fine_trq)).T.reshape(-1, 2)
fine_fuel_map = interp_func(fine_points).reshape(100, 100)  # Interpolated data reshaped to (100, 100)
delta_Peng = 1 #kW

# Resample max torque line
MaxSp = np.linspace(1000, 4000, 300)  # 1 RPM step
MaxTq = interp1d(MaxSp_pt, MaxTq_pt, kind='linear')(MaxSp)
MaxSp_radsec = MaxSp * 2 * np.pi / 60  # Convert to rad/s

delta_Peng = 1  # kW increment

# Initialize an array to hold valid Peng values
valid_peng_values = []

# Iterate through each engine speed (We) in the fine grid
for we in fine_spd:  # fine_spd is already in rad/s
    # Find the corresponding maximum torque for the current speed
    max_torque = interp1d(MaxSp_radsec, MaxTq, kind='linear', bounds_error=False, fill_value="extrapolate")(we)
    
    # Iterate over each torque value in the fine grid
    for torque in fine_trq:
        # Check if torque is within the max torque constraint
        if torque <= max_torque:
            # Calculate Peng for valid torque values
            peng = (we * torque) / 1000  # Convert from W to kW
            valid_peng_values.append(peng)

# Convert valid Peng values to a sorted numpy array
valid_peng_values = np.array(valid_peng_values)
valid_peng_values.sort()

# Create array from Peng_min to Peng_max with delta_Peng increments
Peng_min = valid_peng_values.min()
Peng_max = valid_peng_values.max()
Peng_array = np.arange(Peng_min, Peng_max + delta_Peng, delta_Peng)

# Find minimum fuel consumption for each Peng in Peng_array
min_fuel_rates = []
best_torque_values = []
best_speed_values = []

# Iterate through Peng values in Peng_array
for Peng_target in Peng_array:
    # Initialize a list to store fuel consumption rates for matching Peng
    fuel_rates = []
    torque_values = []
    speed_values = []

    # Iterate through the fine grid and find points matching Peng_target
    for i in range(fine_W.shape[0]):
        for j in range(fine_W.shape[1]):
            # Calculate Peng at the current point
            Peng_actual = fine_eng_power[i, j]

            # Check if Peng_actual is close to Peng_target
            if np.isclose(Peng_actual, Peng_target, atol=0.5):
                # Append the corresponding fuel rate, torque, and speed
                fuel_rates.append(fine_fuel_map[i, j])
                torque_values.append(fine_T[i, j])
                speed_values.append(fine_W[i, j])

    # If any valid fuel rates are found, add the minimum to min_fuel_rates and corresponding torque and speed
    if fuel_rates:
        min_fuel_idx = np.argmin(fuel_rates)
        min_fuel_rates.append(fuel_rates[min_fuel_idx])
        best_torque_values.append(torque_values[min_fuel_idx])
        best_speed_values.append(speed_values[min_fuel_idx])
    else:
        min_fuel_rates.append(None)
        best_torque_values.append(None)
        best_speed_values.append(None)

# Create DataFrame to show the results and filter out None or NaN values
df_results = pd.DataFrame({
    "Peng (kW)": Peng_array,
    "Min Fuel Rate (g/s)": min_fuel_rates,
    "Best Te (Nm)": best_torque_values,
    "Best We (rad/s)": best_speed_values
})

# Filter out rows with None or NaN values
df_results.dropna(inplace=True)

# Initialize lists for coefficients and calculated values
a_values = []
b_values = []
Tg_values = []
Tm_values = []
Wg_values = []
Wm_values = []
Pbatt_values = []
Teng_values = []
Weng_values = []
Peng_values = []  # New list to store Peng values
fuel_rate_values = []  # List to store fuel rate in g/s

# Iterate over each Pveh in df_speed
for idx, Pveh in enumerate(df_speed['P_veh (kW)']):
    # Proceed with calculations for each Pveh
    Pbatt_array = np.zeros(np.shape(Peng_array))
    best_Tg, best_Tm, best_Wg, best_Wm, best_Pbatt, best_Teng, best_Weng, best_Peng, best_fuel_rate = None, None, None, None, None, None, None, None, None
    min_error = float('inf')  # Track minimum error between Pveh and (Pbatt + Peng)

    # Iterate through possible Peng values
    for i, Peng in enumerate(Peng_array):
        # Find corresponding Teng and Weng from df_results for the selected Peng
        result_row = df_results[df_results['Peng (kW)'] == Peng]
        if not result_row.empty:
            Teng = result_row['Best Te (Nm)'].values[0]
            Weng = result_row['Best We (rad/s)'].values[0]

            # Calculate generator torque (Tg) and speed (Wg) based on HW1 Solution
            Tg = -Teng * (S / (S + R))
            Tm = -Teng * (R / (S + R)) + df_speed['Fthrust (N)'].iloc[idx] * reff / K
            Wm = K / reff * df_speed['Speed (m/s)'].iloc[idx]
            Wg = (Weng * (R + S) - Wm * R) / S

            # Efficiency calculations
            eta = 0.85
            etam = 1 / eta if Tm * Wm >= 0 else eta
            etag = 1 / eta if Tg * Wg >= 0 else eta

            # Calculate battery power (Pbatt)
            Pbatt = (etam * Tm * Wm + etag * Tg * Wg) / 1000
            P_veh = df_speed['P_veh (kW)'].iloc[idx]  # Power required by vehicle
            # Pbatt = P_veh - Peng
            Pbatt_array[i] = Pbatt

            # Calculate the error between Pveh and (Pbatt + Peng)
            total_power = Pbatt + Peng
            error = abs(Pveh - total_power)

            # Update the best values if this combination yields a smaller error
            if error < min_error:
                min_error = error
                best_Tg, best_Tm, best_Wg, best_Wm, best_Pbatt, best_Teng, best_Weng, best_Peng = Tg, Tm, Wg, Wm, Pbatt, Teng, Weng, Peng

                # Calculate the fuel rate (g/s)
                fuel_rate = None
                if best_Teng is not None and best_Weng is not None:
                    # Interpolate the fuel map for the given engine speed and torque
                    fuel_interp = RegularGridInterpolator((enginemap_spd, enginemap_trq), enginemap_gpkWh, method='linear')
                    fuel_consumption_gpkWh = fuel_interp((best_Weng, best_Teng))

                    # Convert fuel consumption to g/s
                    fuel_rate = (fuel_consumption_gpkWh * best_Peng * 1000) / 3600

                best_fuel_rate = fuel_rate

    # Append the final calculated values for this Pveh
    Tg_values.append(best_Tg)
    Tm_values.append(best_Tm)
    Wg_values.append(best_Wg)
    Wm_values.append(best_Wm)
    Pbatt_values.append(best_Pbatt)
    Teng_values.append(best_Teng)
    Weng_values.append(best_Weng)
    Peng_values.append(best_Peng)
    fuel_rate_values.append(best_fuel_rate)

    # Perform linear regression if valid points exist
    if not np.isnan(Pbatt_array).any() and len(min_fuel_rates) > 0:
        slope, intercept, _, _, _ = linregress(Pbatt_array, min_fuel_rates)
        a_values.append(slope)
        b_values.append(intercept)

# Convert results to numpy arrays
a_values = np.array(a_values).reshape(-1)
b_values = np.array(b_values).reshape(-1)
Tg_values = np.array(Tg_values).reshape(-1)
Tm_values = np.array(Tm_values).reshape(-1)
Wg_values = np.array(Wg_values).reshape(-1)
Wm_values = np.array(Wm_values).reshape(-1)
Pbatt_values = np.array(Pbatt_values).reshape(-1)
Teng_values = np.array(Teng_values).reshape(-1)
Weng_values = np.array(Weng_values).reshape(-1)
Peng_values = np.array(Peng_values).reshape(-1)  # Convert Peng values to numpy array
fuel_rate_values = np.array(fuel_rate_values).reshape(-1)

# Add to df_speed
df_speed['a'] = a_values
df_speed['b'] = b_values
df_speed['Tg (Nm)'] = Tg_values
df_speed['Tm (Nm)'] = Tm_values
df_speed['Wg (rad/s)'] = Wg_values
df_speed['Wm (rad/s)'] = Wm_values
df_speed['Pbatt (kW)'] = Pbatt_values
df_speed['Teng (Nm)'] = Teng_values
df_speed['Weng (rad/s)'] = Weng_values
df_speed['Peng (kW)'] = Peng_values
df_speed['Fuel Rate (g/s)'] = fuel_rate_values  # Add fuel rate values to df_speed

# Save selected columns to CSV
df_speed[['Peng (kW)', 'Pbatt (kW)', 'P_veh (kW)']].to_csv('specific_values.csv', index=False)

# Time bounds
t0 = 0
tf = len(df_speed)  # Total time steps based on the data
dt = 1  # Step size (time interval)

# Compute the summation term from t0 to (tf - dt)
summation_result = 0
time_values = np.arange(t0, tf, dt)  # Create time values from t0 to tf with step size dt

for t in time_values:
    summation_result += 1 / a_values

# Take the inverse of the summation result
summation_inverse = 1 / summation_result

# Compute the remaining terms
tf_t0_ratio = (tf - t0) / dt
voc_term = (Voc / (2 * Q_batt * R_batt))

# Calculate the multiplicative expression
multiplicative_expression = tf_t0_ratio * voc_term - dSOC

# Compute lambda
lambda_value = (2 * Q_batt**2 * R_batt) * summation_inverse * multiplicative_expression

df_speed['lambda'] = lambda_value

# Use already calculated Peng and Pbatt values
df_speed['Optimal_Peng (kW)'] = df_speed['Peng (kW)']
df_speed['Optimal_Pbatt (kW)'] = df_speed['Pbatt (kW)']

# Initialize list for Hamiltonian values
Hamiltonian_values = []

# Iterate over each row in df_speed to calculate Hamiltonian
for idx, row in df_speed.iterrows():
    P_veh = row['P_veh (kW)']  # Power required by the vehicle
    lambda_value = row['lambda']  # Costate variable for current timestep
    a = row['a']  # Coefficient from linear regression
    b = row['b']  # Intercept from linear regression

    Peng = row['Optimal_Peng (kW)']  # Precomputed engine power
    Pbatt = row['Optimal_Pbatt (kW)']  # Precomputed battery power

    # Ensure Pbatt is within feasible range
    if Pbatt < 0:
        Hamiltonian_values.append(None)  # Append None for invalid values
        continue

    # Calculate SOC_dot based on Pbatt
    Pbatt_watts = Pbatt * 1000  # Convert to watts
    sqrt_term = Voc**2 - 4 * R_batt * Pbatt_watts
    if sqrt_term < 0:
        Hamiltonian_values.append(None)  # Append None for invalid values
        continue
    SOC_dot = - (Voc - np.sqrt(sqrt_term)) / (2 * Q_batt * R_batt)

    # Calculate the Hamiltonian
    Hamiltonian = (a * Peng + b) + (lambda_value * SOC_dot)
    Hamiltonian_values.append(Hamiltonian)

# Add Hamiltonian values to the DataFrame
df_speed['Hamiltonian'] = Hamiltonian_values

# Calculate overall power contributions
total_Peng = df_speed['Optimal_Peng (kW)'].sum()  # Total engine power
total_Pbatt = df_speed['Optimal_Pbatt (kW)'].sum()  # Total battery power
total_Pveh = df_speed['P_veh (kW)'].sum()  # Total vehicle power demand

# Calculate percentage contributions
overall_eng_percent = (total_Peng / total_Pveh) * 100 if total_Pveh > 0 else 0
overall_batt_percent = (total_Pbatt / total_Pveh) * 100 if total_Pveh > 0 else 0

# Print results
print(f"Overall Engine Contribution: {overall_eng_percent:.2f}%")
print(f"Overall Battery Contribution: {overall_batt_percent:.2f}%")

# Save results to file
df_speed[['Optimal_Peng (kW)', 'Optimal_Pbatt (kW)', 'P_veh (kW)', 'Hamiltonian']].to_csv('optimized_power_split.csv', index=False)
