# Time bounds
t0 = 0
tf = 1875  # Replace with the actual final time in seconds or as per your data
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

Pbatt_watts = df_speed['Pbatt (kW)'] * 1000  # Convert Pbatt from kW to W

sqrt_term = Voc**2 - 4 * R_batt * Pbatt_watts

# Ensure the square root term is non-negative
#sqrt_term[sqrt_term < 0] = 0  # Replace negative values with 0

# Calculate SOC_dot
df_speed['SOC_dot'] = - (Voc - np.sqrt(sqrt_term)) / (2 * Q_batt * R_batt)

# Replace NaN or invalid SOC_dot values with zero
df_speed['SOC_dot'] = df_speed['SOC_dot'].fillna(0)

# Initialize lists for storing optimal power split values and Hamiltonian values
optimal_Peng_values = []
optimal_Pbatt_values = []
Hamiltonian_values = []

# Iterate over each row in df_speed to determine the power split
for idx in range(len(df_speed)):
    P_veh = df_speed['P_veh (kW)'].iloc[idx]  # Power required by vehicle
    lambda_value = df_speed['lambda'].iloc[idx]  # Costate variable at the current time step
    a = df_speed['a'].iloc[idx]  # Coefficient from linear regression
    b = df_speed['b'].iloc[idx]  # Intercept from linear regression

    # Initialize variables for tracking the optimal values
    #min_Hamiltonian = float('inf')
    optimal_Peng = None
    optimal_Pbatt = None

    # Iterate through possible engine powers (Peng) to minimize the Hamiltonian
    for Peng in Peng_array:
        Pbatt = P_veh - Peng  # Battery power to balance the vehicle power demand

        # Calculate SOC_dot for this Pbatt
        sqrt_term = Voc**2 - 4 * R_batt * Pbatt  # Convert Pbatt to watts
        sqrt_term = max(sqrt_term, 0)  # Ensure non-negative square root term
        SOC_dot = - (Voc - np.sqrt(sqrt_term)) / (2 * Q_batt * R_batt)

        # Calculate the Hamiltonian
        Hamiltonian = (a * Peng + b) + (lambda_value * SOC_dot)

        # Check if this Hamiltonian is the minimum
        #if Hamiltonian < min_Hamiltonian:
            #min_Hamiltonian = Hamiltonian
        optimal_Peng = Peng
        optimal_Pbatt = Pbatt

    # Store the optimal values
    optimal_Peng_values.append(optimal_Peng)
    optimal_Pbatt_values.append(optimal_Pbatt)
    Hamiltonian_values.append(Hamiltonian)

# Add optimal power split and Hamiltonian values to df_speed
df_speed['Optimal_Peng (kW)'] = optimal_Peng_values
df_speed['Optimal_Pbatt (kW)'] = optimal_Pbatt_values
df_speed['Hamiltonian'] = Hamiltonian_values

# Calculate the total power contributions
total_Peng = df_speed['Optimal_Peng (kW)'].sum()  # Total engine power
total_Pbatt = df_speed['Optimal_Pbatt (kW)'].sum()  # Total battery power
total_Pveh = df_speed['P_veh (kW)'].sum()  # Total vehicle power demand

# Calculate the overall percentages
overall_eng_split = (total_Peng / total_Pveh) * 100 if total_Pveh > 0 else 0
overall_batt_split = (total_Pbatt / total_Pveh) * 100 if total_Pveh > 0 else 0

# Print results
print(f"Overall Engine Contribution: {overall_eng_split:.2f}%")
print(f"Overall Battery Contribution: {overall_batt_split:.2f}%")