# import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
from numba import jit
# import os
# import random
# import psycopg2
# import json

@jit(nopython=True)
def simulate_numba(num_steps, k_e_square, W_e, L_e_plus, W_plus, L_plus_e, eta_batch, dt, epsilon, r_m, k_plus_square):
    phi_e = np.array([0.0, 0.0])
    phi_plus = np.array([0.0, 0.0])
    U = 0.01
    
    phi_e_history = np.zeros((num_steps, 2))
    phi_plus_history = np.zeros((num_steps, 2))
    U_history = np.zeros(num_steps)
    R_vals = np.zeros(num_steps)
    # k_e_psi_e_vals = np.zeros(num_steps)
    # k_e_b_e_vals = np.zeros(num_steps)
    # k_e_psi_plus_vals = np.zeros(num_steps)
    # k_e_b_plus_vals = np.zeros(num_steps)
    # heat_flux_psi_e_b_e_vals = np.zeros(num_steps)
    # heat_flux_psi_e_b_plus_vals = np.zeros(num_steps)
    # b_e_psi_plus_vals = np.zeros(num_steps)
    # b_e_b_plus_vals = np.zeros(num_steps)
    # psi_plus_b_plus_vals = np.zeros(num_steps)

    # switch_times = []

    for i in range(num_steps):
        eta = eta_batch[i]
        xi = np.array([2 * np.sqrt(2) * eta[0] / np.sqrt(k_e_square), 0.0])
        phi_e_dot = np.zeros(2)
        phi_plus_dot = np.zeros(2)

        phi_e_dot[0] = W_e[0, 0] * phi_e[0] + W_e[0, 1] * phi_e[1] + U * (
            L_e_plus[0, 0] * phi_plus[0] + L_e_plus[0, 1] * phi_plus[1]) + (np.sqrt(epsilon) * xi[0]) / np.sqrt(dt)
        phi_e_dot[1] = W_e[1, 0] * phi_e[0] + W_e[1, 1] * phi_e[1] + U * (
            L_e_plus[1, 0] * phi_plus[0] + L_e_plus[1, 1] * phi_plus[1]) + (np.sqrt(epsilon) * xi[1]) / np.sqrt(dt)

        phi_plus_dot[0] = W_plus[0, 0] * phi_plus[0] + W_plus[0, 1] * phi_plus[1] + U * (
            L_plus_e[0, 0] * phi_e[0] + L_plus_e[0, 1] * phi_e[1])
        phi_plus_dot[1] = W_plus[1, 0] * phi_plus[0] + W_plus[1, 1] * phi_plus[1] + U * (
            L_plus_e[1, 0] * phi_e[0] + L_plus_e[1, 1] * phi_e[1])

        phi_e[0] += phi_e_dot[0] * dt
        phi_e[1] += phi_e_dot[1] * dt
        phi_plus[0] += phi_plus_dot[0] * dt
        phi_plus[1] += phi_plus_dot[1] * dt

        R = 0.25 * k * (k_plus_square - k_e_square) * phi_e[0] * phi_plus[0]
        print(k)
        U_dot = R - r_m * U
        U += U_dot * dt

        phi_e_history[i, 0] = phi_e[0]
        phi_e_history[i, 1] = phi_e[1]
        phi_plus_history[i, 0] = phi_plus[0]
        phi_plus_history[i, 1] = phi_plus[1]

        # psi_e = phi_e[0]
        # b_e = phi_e[1]
        # psi_plus = phi_plus[0]
        # b_plus = phi_plus[1]

        U_history[i] = U
        R_vals[i] = R

        # k_e_psi_e_vals[i] = psi_e*psi_e
        # k_e_b_e_vals[i] = b_e*b_e
        # k_e_psi_plus_vals[i] = psi_plus*psi_plus
        # k_e_b_plus_vals[i] = b_plus*b_plus
        # heat_flux_psi_e_b_e_vals[i] = psi_e*b_e
        # heat_flux_psi_e_b_plus_vals[i] = psi_e*b_plus
        # b_e_psi_plus_vals[i] = b_e*psi_plus
        # b_e_b_plus_vals[i] = b_e*b_plus
        # psi_plus_b_plus_vals[i] = psi_plus*b_plus
        


        # if i > 0:
        #     if U_history[i - 1] > 0 and U_history[i] < 0:
        #         switch_times.append(i)
    # return phi_e_history, phi_plus_history, U_history, R_vals, k_e_psi_e_vals, k_e_b_e_vals, k_e_psi_plus_vals, k_e_b_plus_vals, heat_flux_psi_e_b_e_vals, heat_flux_psi_e_b_plus_vals, b_e_psi_plus_vals, b_e_b_plus_vals, psi_plus_b_plus_vals, switch_times
    return phi_e_history, phi_plus_history, U_history, R_vals

class Simulation:
    def __init__(self, epsilon, N_0_squared, r_m, k, m, m_u, dt, total_time):
        self.epsilon = epsilon
        self.N_0_squared = N_0_squared
        self.r_m = r_m
        self.k = k
        self.m = m
        self.m_u = m_u
        self.dt = dt
        self.total_time = total_time
        self.num_steps = int(total_time / dt)
        self.k_e_square = k**2 + m**2
        self.k_plus_square = k**2 + (m + m_u)**2

        self.W_e = np.array([[-1, (k / self.k_e_square)], [-k * N_0_squared, -1]])
        self.W_plus = np.array([[-1, -k / self.k_plus_square], [k * N_0_squared, -1]])
        self.L_e_plus = np.array([[(-k / (2 * self.k_e_square)) * (self.k_plus_square - m_u**2), 0],
                                  [0, k / 2]])
        self.L_plus_e = np.array([[(-k / (2 * self.k_plus_square)) * (m_u**2 - self.k_e_square), 0],
                                  [0, -k / 2]])

        self.phi_e_history = np.zeros((self.num_steps, 2))
        self.phi_plus_history = np.zeros((self.num_steps, 2))
        self.U_history = np.zeros(self.num_steps)
        self.R_vals = np.zeros(self.num_steps)
        self.k_e_psi_e_vals = np.zeros(self.num_steps)
        self.k_e_b_e_vals = np.zeros(self.num_steps)
        self.k_e_psi_plus_vals = np.zeros(self.num_steps)
        self.k_e_b_plus_vals = np.zeros(self.num_steps)
        self.heat_flux_psi_e_b_e_vals = np.zeros(self.num_steps)
        self.heat_flux_psi_e_b_plus_vals = np.zeros(self.num_steps)
        self.b_e_psi_plus_vals = np.zeros(self.num_steps)
        self.b_e_b_plus_vals = np.zeros(self.num_steps)
        self.psi_plus_b_plus_vals = np.zeros(self.num_steps)

        self.eta_batch = self.generate_eta_batch()

    def generate_eta_batch(self):
        return np.random.normal(0, 1, size=(self.num_steps, 1))

    def simulate(self):
        self.phi_e_history, self.phi_plus_history, self.U_history, self.R_vals, self.k_e_psi_e_vals, self.k_e_b_e_vals, self.k_e_psi_plus_vals, self.k_e_b_plus_vals, self.heat_flux_psi_e_b_e_vals, self.heat_flux_psi_e_b_plus_vals, self.b_e_psi_plus_vals, self.b_e_b_plus_vals, self.psi_plus_b_plus_vals, self.switch_times = simulate_numba(
            self.num_steps, self.k_e_square, self.W_e, self.L_e_plus, self.W_plus, self.L_plus_e, self.eta_batch, self.dt, self.epsilon, self.r_m, self.k_plus_square)
        return len(self.switch_times), [t * self.dt for t in self.switch_times]

    # def extract_reversal_data(self, window_size=5000):
    #     reversal_data = {}
    #     step_units = window_size
    #     last_reversal_index = -step_units  

    #     for switch_time in self.switch_times:
    #         index = switch_time
    #         if index - last_reversal_index < step_units:
    #             continue  
            

    #         if index >= step_units and index + step_units < self.num_steps:
    #             pre_reversal_positive = np.all(self.U_history[index - step_units:index] > 0)
    #             post_reversal_negative = np.all(self.U_history[index:index + step_units] < 0)
                
    #             if not pre_reversal_positive or not post_reversal_negative:
    #                 continue 
                
    #             reversal_data[f"reversal_at_{switch_time}"] = {
    #                 'phi_e': self.phi_e_history[index - step_units:index + step_units],
    #                 'phi_plus': self.phi_plus_history[index - step_units:index + step_units],
    #                 'U': self.U_history[index - step_units:index + step_units],
    #                 'R': self.R_vals[index - step_units:index + step_units],
    #                 'k_e_psi_e':self.k_e_psi_e_vals[index - step_units:index + step_units],
    #                 'k_e_b_e':self.k_e_b_e_vals[index - step_units:index + step_units],
    #                 'k_e_psi_plus':self.k_e_psi_plus_vals[index - step_units:index + step_units],
    #                 'k_e_b_plus':self.k_e_b_plus_vals[index - step_units:index + step_units],
    #                 'heat_flux_psi_e_b_e':self.heat_flux_psi_e_b_e_vals[index - step_units:index + step_units],
    #                 'heat_flux_psi_e_b_plus':self.heat_flux_psi_e_b_plus_vals[index - step_units:index + step_units],
    #                 'b_e_psi_plus':self.b_e_psi_plus_vals[index - step_units:index + step_units],
    #                 'b_e_b_plus':self.b_e_b_plus_vals[index - step_units:index + step_units],
    #                 'psi_plus_b_plus':self.psi_plus_b_plus_vals[index - step_units:index + step_units],
    #                 'eta': self.eta_batch[index - step_units:index + step_units]
    #             }
    #         last_reversal_index = index
    #     return reversal_data
    
    # def compress(self):
    #     self.phi_e_history = self.phi_e_history[9::10]
    #     self.phi_plus_history = self.phi_plus_history[9::10]
    #     self.U_history = self.U_history[9::10]
    #     self.R_vals = self.R_vals[9::10]
    #     self.k_e_psi_e_vals = self.k_e_psi_e_vals[9::10]
    #     self.k_e_b_e_vals = self.k_e_b_e_vals[9::10]
    #     self.k_e_psi_plus_vals = self.k_e_psi_plus_vals[9::10]
    #     self.k_e_b_plus_vals = self.k_e_b_plus_vals[9::10]
    #     self.heat_flux_psi_e_b_e_vals = self.heat_flux_psi_e_b_e_vals[9::10]
    #     self.heat_flux_psi_e_b_plus_vals = self.heat_flux_psi_e_b_plus_vals[9::10]
    #     self.b_e_psi_plus_vals = self.b_e_psi_plus_vals[9::10]
    #     self.b_e_b_plus_vals = self.b_e_b_plus_vals[9::10]
    #     self.psi_plus_b_plus_vals = self.psi_plus_b_plus_vals[9::10]
    #     self.eta_batch = self.eta_batch[9::10]

        

# def average_arrays(*arrays):
#     if not arrays:
#         raise ValueError("No arrays provided for averaging.")
    
#     np_arrays = [np.array(arr) for arr in arrays]
#     array_lengths = [len(arr) for arr in np_arrays]

#     if len(set(array_lengths)) != 1:
#         raise ValueError("All input arrays must have the same length.")
    
#     average_array = np.mean(np_arrays, axis=0)
    
#     return average_array

#This might still be useful to encapsulate the current aggregator into a function.

# def reversal_data_aggregator(window_size=5000):
#     phi_e_list = []
#     phi_plus_list = []
#     U_list = []
#     R_list = []
#     k_e_psi_e_list = []
#     k_e_b_e_list = []
#     k_e_psi_plus_list = []
#     k_e_b_plus_list = []
#     heat_flux_psi_e_b_e_list = []
#     heat_flux_psi_e_b_plus_list = []
#     b_e_psi_plus_list = []
#     b_e_b_plus_list = []
#     psi_plus_b_plus_list = []
#     eta_list = []

#     for sim in simulations:
#         reversal_data = sim.extract_reversal_data(window_size)
#         for key in reversal_data:
#             phi_e_list.append(reversal_data[key]['phi_e'])
#             phi_plus_list.append(reversal_data[key]['phi_plus'])
#             U_list.append(reversal_data[key]['U'])
#             R_list.append(reversal_data[key]['R'])
#             k_e_psi_e_list.append(reversal_data[key]['k_e_psi_e'])
#             k_e_b_e_list.append(reversal_data[key]['k_e_b_e'])
#             k_e_psi_plus_list.append(reversal_data[key]['k_e_psi_plus'])
#             k_e_b_plus_list.append(reversal_data[key]['k_e_b_plus'])
#             heat_flux_psi_e_b_e_list.append(reversal_data[key]['heat_flux_psi_e_b_e'])
#             heat_flux_psi_e_b_plus_list.append(reversal_data[key]['heat_flux_psi_e_b_plus'])
#             b_e_psi_plus_list.append(reversal_data[key]['b_e_psi_plus'])
#             b_e_b_plus_list.append(reversal_data[key]['b_e_b_plus'])
#             psi_plus_b_plus_list.append(reversal_data[key]['psi_plus_b_plus'])
#             eta_list.append(reversal_data[key]['eta'])



# def save_simulation_data(simulations, output_file='all_simulation_data.csv'):
#     nested_data = {
#         'simulation_index': [],
#         'U_history': [],
#         'R_vals': [],
#         'k_e_psi_e_vals': [],
#         'k_e_b_e_vals': [],
#         'k_e_psi_plus_vals': [],
#         'k_e_b_plus_vals': [],
#         'heat_flux_psi_e_b_e_vals': [],
#         'heat_flux_psi_e_b_plus_vals': [],
#         'b_e_psi_plus_vals': [],
#         'b_e_b_plus_vals': [],
#         'psi_plus_b_plus_vals': [],
#         'eta_batch': []
#     }

#     for idx, sim in enumerate(simulations):
#         nested_data['simulation_index'].append(idx)
#         nested_data['U_history'].append(sim.U_history.tolist())
#         nested_data['R_vals'].append(sim.R_vals.tolist())
#         nested_data['k_e_psi_e_vals'].append(sim.k_e_psi_e_vals.tolist())
#         nested_data['k_e_b_e_vals'].append(sim.k_e_b_e_vals.tolist())
#         nested_data['k_e_psi_plus_vals'].append(sim.k_e_psi_plus_vals.tolist())
#         nested_data['k_e_b_plus_vals'].append(sim.k_e_b_plus_vals.tolist())
#         nested_data['heat_flux_psi_e_b_e_vals'].append(sim.heat_flux_psi_e_b_e_vals.tolist())
#         nested_data['heat_flux_psi_e_b_plus_vals'].append(sim.heat_flux_psi_e_b_plus_vals.tolist())
#         nested_data['b_e_psi_plus_vals'].append(sim.b_e_psi_plus_vals.tolist())
#         nested_data['b_e_b_plus_vals'].append(sim.b_e_b_plus_vals.tolist())
#         nested_data['psi_plus_b_plus_vals'].append(sim.psi_plus_b_plus_vals.tolist())
#         nested_data['eta_batch'].append(sim.eta_batch.flatten().tolist())

#     df = pd.DataFrame(nested_data)
#     df.to_csv(output_file, index=False)
#     print(f"All simulation data saved to {output_file}")

# if __name__ == "__main__":
#     # simulations = []
#     compositeHalfList = []
#     epsilon = 0.12394270273516043
#     N_0_squared = 318.8640217310387
#     r_m = 0.1
#     k = 2 * np.pi * 6
#     m = 2 * np.pi * 3
#     m_u = 2 * np.pi * 7
#     dt = 0.001
#     total_time = 200

#     phi_e_list = []
#     phi_plus_list = []
#     U_list = []
#     R_list = []
#     k_e_psi_e_list = []
#     k_e_b_e_list = []
#     k_e_psi_plus_list = []
#     k_e_b_plus_list = []
#     heat_flux_psi_e_b_e_list = []
#     heat_flux_psi_e_b_plus_list = []
#     b_e_psi_plus_list = []
#     b_e_b_plus_list = []
#     psi_plus_b_plus_list = []
#     eta_list = []

#     for i in range(3800):
#         print(f"Running iteration {i}")
#         sim = Simulation(epsilon, N_0_squared, r_m, k, m, m_u, dt, total_time)
#         sim.simulate()
#         reversal_data = sim.extract_reversal_data(window_size=5000)
#         for key in reversal_data:
#             phi_e_list.append(reversal_data[key]['phi_e'][9::10])
#             phi_plus_list.append(reversal_data[key]['phi_plus'][9::10])
#             U_list.append(reversal_data[key]['U'][9::10])
#             R_list.append(reversal_data[key]['R'][9::10])
#             k_e_psi_e_list.append(reversal_data[key]['k_e_psi_e'][9::10])
#             k_e_b_e_list.append(reversal_data[key]['k_e_b_e'][9::10])
#             k_e_psi_plus_list.append(reversal_data[key]['k_e_psi_plus'][9::10])
#             k_e_b_plus_list.append(reversal_data[key]['k_e_b_plus'][9::10])
#             heat_flux_psi_e_b_e_list.append(reversal_data[key]['heat_flux_psi_e_b_e'][9::10])
#             heat_flux_psi_e_b_plus_list.append(reversal_data[key]['heat_flux_psi_e_b_plus'][9::10])
#             b_e_psi_plus_list.append(reversal_data[key]['b_e_psi_plus'][9::10])
#             b_e_b_plus_list.append(reversal_data[key]['b_e_b_plus'][9::10])
#             psi_plus_b_plus_list.append(reversal_data[key]['psi_plus_b_plus'][9::10])
#             eta_list.append(reversal_data[key]['eta'][9::10])
#         # del sim

#     conn = psycopg2.connect("dbname=simulations_data user=simulationuser password=simulations2024 host=localhost")
#     cur = conn.cursor()

#     # psi_e = [item[0] for item in phi_e_list]
#     # b_e = [item[1] for item in phi_e_list]
#     # psi_plus = [item[0] for item in phi_plus_list]
#     # b_plus = [item[1] for item in phi_plus_list]

#     for i in range(len(U_list)):

#         psi_e_json = json.dumps(phi_e_list[i][:, 0].tolist())
#         b_e_json = json.dumps(phi_e_list[i][:, 1].tolist())
#         psi_plus_json = json.dumps(phi_plus_list[i][:, 0].tolist())
#         b_plus_json = json.dumps(phi_plus_list[i][:, 1].tolist())
#         U_list_json = json.dumps(U_list[i].tolist())
#         R_list_json = json.dumps(R_list[i].tolist())
#         k_e_psi_e_list_json = json.dumps(k_e_psi_e_list[i].tolist())
#         k_e_b_e_list_json = json.dumps(k_e_b_e_list[i].tolist())
#         k_e_psi_plus_list_json = json.dumps(k_e_psi_plus_list[i].tolist())
#         k_e_b_plus_list_json = json.dumps(k_e_b_plus_list[i].tolist())
#         heat_flux_psi_e_b_e_list_json = json.dumps(heat_flux_psi_e_b_e_list[i].tolist())
#         heat_flux_psi_e_b_plus_list_json = json.dumps(heat_flux_psi_e_b_plus_list[i].tolist())
#         b_e_psi_plus_list_json = json.dumps(b_e_psi_plus_list[i].tolist())
#         b_e_b_plus_list_json = json.dumps(b_e_b_plus_list[i].tolist())
#         psi_plus_b_plus_list_json = json.dumps(psi_plus_b_plus_list[i].tolist())
#         eta_list_json = json.dumps(eta_list[i].tolist())

#     ##The variables in the for loop have weird names. They are not lists.

#         cur.execute("""
#         INSERT INTO composite_data (eps, n_0_squared, psi_e, b_e, psi_plus, b_plus, U_list, R_list, k_e_psi_e_list, 
#                                 k_e_b_e_list, k_e_psi_plus_list, k_e_b_plus_list, heat_flux_psi_e_b_e_list, 
#                                 heat_flux_psi_e_b_plus_list, b_e_psi_plus_list, b_e_b_plus_list, 
#                                 psi_plus_b_plus_list, eta_list) 
#         VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
#         """, (epsilon, N_0_squared, psi_e_json, b_e_json, psi_plus_json, b_plus_json, U_list_json, R_list_json, 
#             k_e_psi_e_list_json, k_e_b_e_list_json, k_e_psi_plus_list_json, k_e_b_plus_list_json, 
#             heat_flux_psi_e_b_e_list_json, heat_flux_psi_e_b_plus_list_json, b_e_psi_plus_list_json, 
#             b_e_b_plus_list_json, psi_plus_b_plus_list_json, eta_list_json))

#     conn.commit()
#     cur.close()
#     conn.close()
#     print("Inserted batch into database")

#     # plot_composite_analysis("totalPlot", epsilon, N_0_squared, phi_e_list, phi_plus_list, U_list, R_list, k_e_psi_e_list, k_e_b_e_list, k_e_psi_plus_list, k_e_b_plus_list, heat_flux_psi_e_b_e_list, heat_flux_psi_e_b_plus_list, b_e_psi_plus_list, b_e_b_plus_list, psi_plus_b_plus_list, eta_list, dt,2000)
#     # save_simulation_data(simulations)