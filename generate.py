from trident import Simulation
import numpy as np
import json
import matplotlib.pyplot as plt
import pandas as pd

with open("./params.json", mode = "r", encoding = "utf-8") as f:
    params = json.load(f)
    seed_value = params["seed_value"]
    interval_steps = params["simulation"]["interval_steps"]

np.random.seed(seed_value)

epsilon = 0.12394270273516043
N_0_squared = 318.8640217310387
r_m = 0.1
k = 2 * np.pi * 6
m = 2 * np.pi * 3
m_u = 2 * np.pi * 7
dt = 0.001
total_time = 1000

initial_sim = Simulation(
    epsilon = epsilon,
    N_0_squared = N_0_squared,
    r_m = r_m,
    k = k,
    m = m,
    m_u = m_u,
    dt = dt,
    total_time = total_time
)

print("Initial simulation running...")
initial_sim.simulate()

no_random_sim = Simulation(
    epsilon = epsilon,
    N_0_squared = N_0_squared,
    r_m = 0,
    k = k,
    m = m,
    m_u = m_u,
    dt = dt,
    total_time = 1,  # 1000 time-steps
    randomness = False
)

starting_step = 1000
dataset = []

print("No randomness simulation running...")
for i in range(starting_step, initial_sim.U_history.shape[0], interval_steps):
    psi_e = initial_sim.phi_e_history[i, 0]
    b_e = initial_sim.phi_e_history[i, 1]
    psi_plus = initial_sim.phi_plus_history[i, 0]
    b_plus = initial_sim.phi_plus_history[i, 1]
    u = initial_sim.U_history[i]

    print(f"    Based on time-step {i} parameters")
    no_random_sim.simulate(
        phi_e = np.array([psi_e, b_e]),
        phi_plus = np.array([psi_plus, b_plus]),
        U = u
    )
    dataset.append(no_random_sim.get_json_simulation_data())

df = pd.DataFrame(
    dataset,
    columns = [
        "eps",
        "n_0_squared",
        "psi_e",
        "b_e",
        "psi_plus",
        "b_plus",
        "u_list",
        "r_list",
        "k_e_psi_e_list",
        "k_e_b_e_list",
        "k_e_psi_plus_list",
        "k_e_b_plus_list",
        "heat_flux_psi_e_b_e_list",
        "heat_flux_psi_e_b_plus_list",
        "b_e_psi_plus_list",
        "b_e_b_plus_list",
        "psi_plus_b_plus_list",
        "eta_list"
    ]
)

print("Saving to file...")
df.index.name = "id"
df.to_csv("./deterministic_dataset.csv", index = True)
