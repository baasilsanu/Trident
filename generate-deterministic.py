from trident import Simulation
import numpy as np
import json
import gc
from utils import save_to_csv


with open("./params.json", mode = "r", encoding = "utf-8") as f:
    params = json.load(f)
    # seed_value = params["seed_value"]
    num_runs = params["deterministic"]["num_runs"]
    save_seed_batch_size = params["deterministic"]["save_seed_batch_size"]
    dataset_save_root_path = params["deterministic"]["dataset_save_root_path"]
    dataset_name = params["deterministic"]["dataset_name"]

    epsilon = params["deterministic"]["random_simulation"]["epsilon"]
    N_0_squared = params["deterministic"]["random_simulation"]["N_0_squared"]
    r_m = params["deterministic"]["random_simulation"]["r_m"]
    k = 2 * np.pi * 6
    m = 2 * np.pi * 3
    m_u = 2 * np.pi * 7
    dt = params["deterministic"]["random_simulation"]["dt"]
    total_time = params["deterministic"]["random_simulation"]["total_time"]
    initial_U = params["deterministic"]["random_simulation"]["initial_U"]

    interval_steps = params["deterministic"]["deterministic_simulation"]["interval_steps"]
    starting_step = params["deterministic"]["deterministic_simulation"]["starting_step"]
    selected_features = params["deterministic"]["deterministic_simulation"]["selected_features"]

print("Expectation:")
per_run_num = (int(total_time / dt) - starting_step) // interval_steps + 1
print(f"    Number of data points per run: {per_run_num}")
print(f"    Number of total data points: {num_runs * per_run_num}")
print(f"    Number of features: {len(selected_features)}")
print()





dataset = []
num_data_points = 0
starting_idx = 0

for seed_val in range(1, num_runs + 1):
    print(f"==================== Random Seed: {seed_val} ====================")
    np.random.seed(seed_val)

    initial_sim = Simulation(
        epsilon = epsilon,
        N_0_squared = N_0_squared,
        r_m = r_m,
        k = k,
        m = m,
        m_u = m_u,
        dt = dt,
        total_time = total_time,
        randomness = True
    )

    print("Initial simulation running...")
    initial_sim.simulate(
        phi_e = np.array([0.0, 0.0]),
        phi_plus = np.array([0.0, 0.0]),
        U = initial_U
    )

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

    print("Deterministic simulation running...")
    print(f"Number of new data points: {(initial_sim.U_history.shape[0] - starting_step) // interval_steps + 1}")
    for i in range(starting_step, initial_sim.U_history.shape[0], interval_steps):
        psi_e = initial_sim.phi_e_history[i, 0]
        b_e = initial_sim.phi_e_history[i, 1]
        psi_plus = initial_sim.phi_plus_history[i, 0]
        b_plus = initial_sim.phi_plus_history[i, 1]
        u = initial_sim.U_history[i]

        # print(f"    Based on time-step {i} parameters")
        no_random_sim.simulate(
            phi_e = np.array([psi_e, b_e]),
            phi_plus = np.array([psi_plus, b_plus]),
            U = u
        )
        dataset.append(no_random_sim.get_json_simulation_data(selected_features))
        num_data_points += 1

    del initial_sim
    del no_random_sim
    gc.collect()

    if(seed_val % save_seed_batch_size == 0):
        save_to_csv(
            dataset = dataset,
            selected_features = selected_features,
            start_idx = starting_idx,
            end_idx = num_data_points,
            dataset_save_path = f"{dataset_save_root_path}/{dataset_name}"
        )
        
        dataset = []
        starting_idx = num_data_points


if(len(dataset) != 0):
    save_to_csv(
        dataset = dataset,
        selected_features = selected_features,
        start_idx = starting_idx,
        end_idx = num_data_points,
        dataset_save_path = f"{dataset_save_root_path}/{dataset_name}"
    )

print(f"\nTotal number of data points: {num_data_points}")
