import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from tqdm import tqdm  # For progress bars
from pprint import pprint
import math
from scipy.optimize import minimize_scalar


# ------------ SYSTEM PARAMETERS ------------

# Room/area parameters
ROOM_SIZE_X = 10  # meters
ROOM_SIZE_Y = 10  # meters

# Number of users
NUM_USERS = 20

# Number of RF base stations
NUM_RF_BS = 1

# Number of VLC access points
NUM_VLC_AP = 4

# RF network parameters
RF_TRANSMIT_POWER_MAX = 1.0  # Watts
RF_BANDWIDTH_MAX = 20e6  # Hz (20 MHz)
RF_NOISE_DENSITY = 1e-13  # W/Hz
RF_FIXED_COST = 5.0  # monetary units
RF_VARIABLE_COST_COEFF = 0.1  # monetary units per Mbps
RF_ENERGY_COST_COEFF = 2.0  # monetary units per Watt
RF_CIRCUIT_POWER = 0.05  # Watts
RF_FREQ = 2.4e9  # Hz (2.4 GHz)
RF_MAX_CAPACITY = RF_BANDWIDTH_MAX * np.log2(1 + RF_TRANSMIT_POWER_MAX / (RF_NOISE_DENSITY * RF_BANDWIDTH_MAX))

# VLC network parameters
VLC_TRANSMIT_POWER_MAX = 0.5  # Watts
VLC_BANDWIDTH_MAX = 100e6  # Hz (100 MHz)
VLC_NOISE_DENSITY = 1e-14  # W/Hz
VLC_FIXED_COST = 3.0  # monetary units
VLC_VARIABLE_COST_COEFF = 0.05  # monetary units per Mbps
VLC_ENERGY_COST_COEFF = 1.0  # monetary units per Watt
VLC_CIRCUIT_POWER = 0.05  # Watts
VLC_RESPONSIVITY = 0.5  # A/W
VLC_MAX_CAPACITY = VLC_BANDWIDTH_MAX * np.log2(1 + VLC_TRANSMIT_POWER_MAX / (VLC_NOISE_DENSITY * VLC_BANDWIDTH_MAX))

# Speed of light
SPEED_OF_LIGHT = 3e8  # m/s

# ------------ USER CLASS ------------

class User:
    def __init__(self, user_id, x, y, alpha_values=None, duty_cycle=None):
        self.user_id = user_id
        self.x = x
        self.y = y

        if duty_cycle is None:
            self.duty_cycle = 0.75
        
        self.duty_cycle = duty_cycle
        
        # Default user preferences if not provided
        if alpha_values is None:
            # Weights for: data rate, cost, delay, reliability 
            self.alpha = np.array([1.0, 0.8, 0.5, 0.6])
        else:
            self.alpha = np.array(alpha_values)
        
        # Normalizing weights
        self.alpha = self.alpha / np.sum(self.alpha)
        
        # Current network selection (0: RF, 1: VLC)
        self.selected_network = None
        self.utilities = {"RF": 0, "VLC": 0}
        self.data_rates = {"RF": 0, "VLC": 0}
    
    def calculate_rf_channel_gain(self, rf_bs_positions):
        # Simple distance-based path loss model for RF
        best_gain = -float('inf')
        
        for bs_pos in rf_bs_positions:
            distance = np.sqrt((self.x - bs_pos[0])**2 + (self.y - bs_pos[1])**2)
            # Path loss exponent of 3.5 for indoor environment
            path_loss_db = 20 * np.log10(RF_FREQ) + 35 * np.log10(distance) - 147.5
            channel_gain = 10**(-path_loss_db/10)
            best_gain = max(best_gain, channel_gain)
        
        return best_gain
    
    def calculate_vlc_channel_gain(self, vlc_ap_positions):
        # VLC channel model (simplified Lambertian model)
        best_gain = -float('inf')
        
        # Assuming a height difference of 2.5m between AP and user device
        height_diff = 2.5
        
        for ap_pos in vlc_ap_positions:
            horizontal_distance = np.sqrt((self.x - ap_pos[0])**2 + (self.y - ap_pos[1])**2)
            distance = np.sqrt(horizontal_distance**2 + height_diff**2)
            
            # Angle of irradiance
            phi = np.arctan(horizontal_distance / height_diff)
            
            # Lambertian order (typical LED semi-angle at half power = 60 degrees)
            m = -np.log(2) / np.log(np.cos(np.radians(60)))
            
            # Detector area (typical value)
            A = 1e-4  # m^2
            
            # Optical filter gain
            g_of = 1.0
            
            # Concentrator gain
            g_oc = 1.0
            
            # Lambertian model
            if phi <= np.pi/2:  # If within FOV
                channel_gain = (m+1) * A * g_of * g_oc * np.cos(phi)**(m+1) / (2 * np.pi * distance**2)
                best_gain = max(best_gain, channel_gain)
        
        return best_gain
    
    def calculate_utility(self, network_params):
        """Calculate utility for both networks and make selection"""
        rf_params = network_params["RF"]
        vlc_params = network_params["VLC"]
        
        # Calculate RF utility
        rf_channel_gain = self.calculate_rf_channel_gain(rf_params["bs_positions"])
        rf_snr = rf_params["tx_power"] * rf_channel_gain / (RF_NOISE_DENSITY * rf_params["bandwidth"])
        rf_data_rate = rf_params["bandwidth"] * np.log2(1 + rf_snr) / 1e6  # Mbps
        
        rf_cost = RF_FIXED_COST + RF_VARIABLE_COST_COEFF * rf_data_rate
        
        # Simple propagation delay (distance/speed of light)
        min_distance_rf = min([np.sqrt((self.x - bs[0])**2 + (self.y - bs[1])**2) for bs in rf_params["bs_positions"]])
        rf_prop_delay = min_distance_rf / SPEED_OF_LIGHT
        
        # Transmission delay (assume 1500 byte packet)
        rf_trans_delay = 1500 * 8 / (rf_data_rate * 1e6)
        
        # Queue delay (simplified model based on load)
        rf_queue_delay = 0.001 * rf_params["load"]  # seconds
        
        rf_delay = rf_prop_delay + rf_trans_delay + rf_queue_delay
        
        # Reliability based on SNR
        rf_reliability = min(1.0, 1 - np.exp(-0.5 * rf_snr))
        
        # Calculate VLC utility
        vlc_channel_gain = self.calculate_vlc_channel_gain(vlc_params["ap_positions"])
        
        # Check if VLC is available (has line of sight)
        if vlc_channel_gain > 0:
            vlc_snr = (VLC_RESPONSIVITY * vlc_params["tx_power"] * vlc_channel_gain) / (VLC_NOISE_DENSITY * vlc_params["bandwidth"])
            vlc_data_rate = vlc_params["bandwidth"] * np.log2(1 + vlc_snr) / 1e6  # Mbps
            
            vlc_cost = VLC_FIXED_COST + VLC_VARIABLE_COST_COEFF * vlc_data_rate
            
            # Simple propagation delay (distance/speed of light)
            min_distance_vlc = min([np.sqrt((self.x - ap[0])**2 + (self.y - ap[1])**2 + 2.5**2) for ap in vlc_params["ap_positions"]])
            vlc_prop_delay = min_distance_vlc / SPEED_OF_LIGHT
            
            # Transmission delay (assume 1500 byte packet)
            vlc_trans_delay = 1500 * 8 / (vlc_data_rate * 1e6)
            
            # Queue delay (simplified model based on load)
            vlc_queue_delay = 0.001 * vlc_params["load"]  # seconds
            
            vlc_delay = vlc_prop_delay + vlc_trans_delay + vlc_queue_delay
            
            # Reliability based on SNR and line-of-sight probability
            vlc_reliability = min(1.0, (1 - np.exp(-0.5 * vlc_snr)) * 0.9)  # 0.9 factor for potential blockage
        else:
            # No line of sight, VLC not available
            vlc_data_rate = 0
            vlc_cost = float('inf')
            vlc_delay = float('inf')
            vlc_reliability = 0
        
        # Store data rates for capacity calculations
        self.data_rates["RF"] = rf_data_rate
        self.data_rates["VLC"] = vlc_data_rate

        qos_term = math.log(1+5*self.duty_cycle)
        
        # Calculate utilities using the alpha weights
        utility_rf = (self.alpha[0] * rf_data_rate - 
                     self.alpha[1] * rf_cost * self.duty_cycle - 
                     self.alpha[2] * rf_delay + 
                     self.alpha[3] * rf_reliability +
                     self.alpha[3] * qos_term)
        
        utility_vlc = (self.alpha[0] * vlc_data_rate - 
                      self.alpha[1] * vlc_cost * self.duty_cycle - 
                      self.alpha[2] * vlc_delay + 
                      self.alpha[3] * vlc_reliability +
                      self.alpha[3] * self.duty_cycle)
        
        self.utilities["RF"] = utility_rf
        self.utilities["VLC"] = utility_vlc
        
        # Select network with higher utility
        if utility_rf >= utility_vlc:
            self.selected_network = "RF"
            return "RF", utility_rf
        else:
            self.selected_network = "VLC"
            return "VLC", utility_vlc

# ------------ PROVIDER CLASS ------------

class Provider:
    def __init__(self):
        # Provider's strategy variables (pricing, resource allocation, etc.)
        self.rf_price = RF_FIXED_COST
        self.vlc_price = VLC_FIXED_COST
        self.rf_tx_power = RF_TRANSMIT_POWER_MAX * 0.5
        self.vlc_tx_power = VLC_TRANSMIT_POWER_MAX * 0.5
        self.rf_bandwidth = RF_BANDWIDTH_MAX * 0.5
        self.vlc_bandwidth = VLC_BANDWIDTH_MAX * 0.5
        self.cost_per_rf_bandwidth = 1.5  # monetary units
        self.cost_per_vlc_bandwidth = 1.0  # monetary units
        
        # Base station and access point positions
        self.rf_bs_positions = [(ROOM_SIZE_X/2, ROOM_SIZE_Y/2)]
        self.vlc_ap_positions = [
            (ROOM_SIZE_X/4, ROOM_SIZE_Y/4),
            (ROOM_SIZE_X/4, 3*ROOM_SIZE_Y/4),
            (3*ROOM_SIZE_X/4, ROOM_SIZE_Y/4),
            (3*ROOM_SIZE_X/4, 3*ROOM_SIZE_Y/4)
        ]
        
        # Initial network loads
        self.rf_load = 0.5
        self.vlc_load = 0.5
        
    def get_network_params(self):
        """Return current network parameters"""
        return {
            "RF": {
                "tx_power": self.rf_tx_power,
                "bandwidth": self.rf_bandwidth,
                "price": self.rf_price,
                "bs_positions": self.rf_bs_positions,
                "load": self.rf_load
            },
            "VLC": {
                "tx_power": self.vlc_tx_power,
                "bandwidth": self.vlc_bandwidth,
                "price": self.vlc_price,
                "ap_positions": self.vlc_ap_positions,
                "load": self.vlc_load
            }
        }
    
    def calculate_utility(self, users):
        """Calculate provider's utility based on user selections"""
        # Count how many users selected each network
        n_rf = sum(1 for user in users if user.selected_network == "RF")
        n_vlc = sum(1 for user in users if user.selected_network == "VLC")

        vlc_energy_cost_per_unit = (VLC_CIRCUIT_POWER + self.vlc_tx_power)*VLC_ENERGY_COST_COEFF
        rf_energy_cost_per_unit = (RF_CIRCUIT_POWER + self.rf_tx_power)*RF_ENERGY_COST_COEFF

        rf_energy_cost = 0
        vlc_energy_cost = 0

        for user in users:
            if user.selected_network=="RF":
                rf_energy_cost += rf_energy_cost_per_unit*user.duty_cycle
            else:
                vlc_energy_cost += vlc_energy_cost_per_unit*user.duty_cycle


        vlc_bandwidth_cost = self.vlc_bandwidth * self.cost_per_vlc_bandwidth
        rf_bandwidth_cost = self.rf_bandwidth * self.cost_per_rf_bandwidth

        if n_rf==0:
            provider_rf_cost = rf_bandwidth_cost/RF_BANDWIDTH_MAX
        else:
            provider_rf_cost = rf_bandwidth_cost/RF_BANDWIDTH_MAX + rf_energy_cost/n_rf

        if n_vlc==0:
            provider_vlc_cost = vlc_bandwidth_cost/VLC_BANDWIDTH_MAX
        else:
            provider_vlc_cost = vlc_bandwidth_cost/VLC_BANDWIDTH_MAX + vlc_energy_cost/n_vlc
        
        # Update network loads based on user selections
        if n_rf > 0:
            self.rf_load = n_rf / len(users)
        else:
            self.rf_load = 0.1  # minimal load
            
        if n_vlc > 0:
            self.vlc_load = n_vlc / len(users)
        else:
            self.vlc_load = 0.1  # minimal load
        
        # Calculate revenue
        revenue_rf = n_rf * self.rf_price
        revenue_vlc = n_vlc * self.vlc_price
        
        # Calculate costs
        cost_rf = n_rf * provider_rf_cost
        cost_vlc = n_vlc * provider_vlc_cost
        
        # Calculate load balance penalty
        load_balance_penalty = 0.5 * abs(n_rf - n_vlc)
        
        # Calculate total utility
        utility = (revenue_rf - cost_rf) + (revenue_vlc - cost_vlc) - load_balance_penalty

        print("Provider utility:", utility)
        
        return utility
        
    def calculate_network_capacity(self, users):
        """Calculate allocated and unallocated capacity for both networks"""
        # Sum of data rates for users on each network
        rf_allocated = sum([user.data_rates["RF"] for user in users if user.selected_network == "RF"])
        vlc_allocated = sum([user.data_rates["VLC"] for user in users if user.selected_network == "VLC"])
        
        # Maximum theoretical capacity
        rf_max = RF_BANDWIDTH_MAX * np.log2(1 + self.rf_tx_power / (RF_NOISE_DENSITY * RF_BANDWIDTH_MAX)) / 1e6  # Mbps
        vlc_max = VLC_BANDWIDTH_MAX * np.log2(1 + self.vlc_tx_power / (VLC_NOISE_DENSITY * VLC_BANDWIDTH_MAX)) / 1e6  # Mbps
        
        # Unallocated capacity
        rf_unallocated = max(0, rf_max - rf_allocated)
        vlc_unallocated = max(0, vlc_max - vlc_allocated)
        
        return {
            "RF": {
                "allocated": rf_allocated,
                "unallocated": rf_unallocated,
                "total": rf_max
            },
            "VLC": {
                "allocated": vlc_allocated,
                "unallocated": vlc_unallocated,
                "total": vlc_max
            }
        }

# ------------ STACKELBERG GAME ------------

class StackelbergGame:
    def __init__(self, num_users=NUM_USERS, users=None):
        self.provider = Provider()

        if users:
            self.users = users
        else:
            # Generate random user positions
            self.users = []
            for i in range(num_users):
                x = np.random.uniform(0, ROOM_SIZE_X)
                y = np.random.uniform(0, ROOM_SIZE_Y)
                
                # Generate random preference weights
                alpha_values = np.random.uniform(0.3, 1.0, 5)
                duty_cycle = np.random.rand()
                
                self.users.append(User(i, x, y, alpha_values, duty_cycle=duty_cycle))
    
    def users_response(self):
        """Calculate users' best responses to provider's strategy"""
        network_params = self.provider.get_network_params()
        
        for user in self.users:
            user.calculate_utility(network_params)
    
    def provider_objective(self, strategy_vars):
        """Objective function for provider optimization"""
        # Update provider's strategy variables
        self.provider.rf_price = strategy_vars[0]
        self.provider.vlc_price = strategy_vars[1]
        self.provider.rf_tx_power = strategy_vars[2]
        self.provider.vlc_tx_power = strategy_vars[3]
        self.provider.rf_bandwidth = strategy_vars[4]
        self.provider.vlc_bandwidth = strategy_vars[5]
        
        # Calculate users' responses
        # Leader predicts the followers best response.
        # self.users_response()
        
        # Based on the followers' best response, leader's utility is found and maximised.
        # Calculate provider's utility
        utility = self.provider.calculate_utility(self.users)
        
        # We want to maximize utility, so return negative for minimization
        return -utility
    
    def user_objective(self, user):
        def negative_utility(duty_cycle):
            # We want to maximize utility, so minimize the negative utility
            user.duty_cycle = duty_cycle
            _, utility = user.calculate_utility(self.provider.get_network_params())
            return -utility  # negate for maximization

        result = minimize_scalar(negative_utility, bounds=(0, 1), method='bounded')
        if result.success:
            optimal_duty = result.x
            user.duty_cycle = optimal_duty
            best_network, max_utility = user.calculate_utility(self.provider.get_network_params())
            return optimal_duty, best_network, max_utility
        else:
            raise RuntimeError("Duty cycle optimization failed.")

    
    
    def verify_stackelberg_equilibrium(self, tolerance=1e-4):
        """
        Verify that the current solution is a Stackelberg equilibrium.
        
        Args:
            provider_strategy: The provider's strategy to verify
            tolerance: Tolerance for utility differences
            
        Returns:
            is_equilibrium: Boolean indicating whether it's a Stackelberg equilibrium
            verification_details: Dictionary with details of verification
        """
        print("\nVerifying Stackelberg Equilibrium:")
        
        
        # Store original strategy to restore later
        original_strategy = [self.provider.rf_price, self.provider.vlc_price, self.provider.rf_tx_power, self.provider.vlc_tx_power, \
                             self.provider.rf_bandwidth, self.provider.vlc_bandwidth]

        # Let users respond to this strategy
        self.users_response()
        
        # Calculate baseline utility
        baseline_utility = self.provider.calculate_utility(self.users)
        print(f"Baseline provider utility: {baseline_utility:.4f}")
        
        # Check if this is locally optimal by slightly perturbing each parameter
        is_provider_optimal = True
        provider_verification = {}
        perturbation = 0.01  # Small perturbation
        
        param_names = ['RF Price', 'VLC Price', 'RF Power', 'VLC Power', 'RF Bandwidth', 'VLC Bandwidth']
        
        for i, param_name in enumerate(param_names):
            # Try increasing the parameter
            perturbed_strategy = original_strategy.copy()
            perturbed_strategy[i] *= (1 + perturbation)
            
            # Apply perturbed strategy
            if i == 0:
                self.provider.rf_price = perturbed_strategy[i]
            elif i == 1:
                self.provider.vlc_price = perturbed_strategy[i]
            elif i == 2:
                self.provider.rf_tx_power = perturbed_strategy[i]
            elif i == 3:
                self.provider.vlc_tx_power = perturbed_strategy[i]
            elif i == 4:
                self.provider.rf_bandwidth = perturbed_strategy[i]
            elif i == 5:
                self.provider.vlc_bandwidth = perturbed_strategy[i]
            
            # Let users respond
            self.users_response()
            
            # Calculate utility with increased parameter
            increased_utility = self.provider.calculate_utility(self.users)
            
            # Try decreasing the parameter
            perturbed_strategy = original_strategy.copy()
            perturbed_strategy[i] *= (1 - perturbation)
            
            # Apply perturbed strategy
            if i == 0:
                self.provider.rf_price = perturbed_strategy[i]
            elif i == 1:
                self.provider.vlc_price = perturbed_strategy[i]
            elif i == 2:
                self.provider.rf_tx_power = perturbed_strategy[i]
            elif i == 3:
                self.provider.vlc_tx_power = perturbed_strategy[i]
            elif i == 4:
                self.provider.rf_bandwidth = perturbed_strategy[i]
            elif i == 5:
                self.provider.vlc_bandwidth = perturbed_strategy[i]
            
            # Let users respond
            self.users_response()
            
            # Calculate utility with decreased parameter
            decreased_utility = self.provider.calculate_utility(self.users)
            
            # Check if baseline is better than both perturbations
            is_locally_optimal = (baseline_utility >= increased_utility - tolerance and 
                                baseline_utility >= decreased_utility - tolerance)
            
            provider_verification[param_name] = {
                'locally_optimal': is_locally_optimal,
                'baseline': baseline_utility,
                'increased': increased_utility,
                'decreased': decreased_utility
            }
            
            if not is_locally_optimal:
                is_provider_optimal = False
                print(f"  Provider strategy not optimal for {param_name}:")
                print(f"    Baseline: {baseline_utility:.4f}, Increased: {increased_utility:.4f}, Decreased: {decreased_utility:.4f}")
                print("Original Strategy: ", original_strategy)
                break

        return is_provider_optimal

    """
    def solve(self, num_iter=100):
        # Solve the Stackelberg game
        # Initial strategy values - use diverse starting points to avoid local minima
        initial_strategy = [
            RF_FIXED_COST * 1.5,          # RF price - start higher than fixed cost
            VLC_FIXED_COST * 1.5,         # VLC price - start higher than fixed cost
            RF_TRANSMIT_POWER_MAX * 0.7,  # RF tx power - different from 0.5
            VLC_TRANSMIT_POWER_MAX * 0.6, # VLC tx power - different from 0.5
            RF_BANDWIDTH_MAX * 0.4,       # RF bandwidth - different from 0.5
            VLC_BANDWIDTH_MAX * 0.8       # VLC bandwidth - different from 0.5
        ]
        
        # Strategy bounds
        bounds = [
            (RF_FIXED_COST, 10.0),                                  # RF price bounds (min must be ≥ fixed cost)
            (VLC_FIXED_COST, 10.0),                                 # VLC price bounds (min must be ≥ fixed cost)
            (0.1 * RF_TRANSMIT_POWER_MAX, RF_TRANSMIT_POWER_MAX),   # RF tx power bounds
            (0.1 * VLC_TRANSMIT_POWER_MAX, VLC_TRANSMIT_POWER_MAX), # VLC tx power bounds
            (0.1 * RF_BANDWIDTH_MAX, RF_BANDWIDTH_MAX),             # RF bandwidth bounds
            (0.1 * VLC_BANDWIDTH_MAX, VLC_BANDWIDTH_MAX)            # VLC bandwidth bounds
        ]
        
        print("Starting optimization with initial strategy:", initial_strategy)
        print("Bounds:", bounds)
        
        cur_strategy = initial_strategy
        prev_utility = float('-inf')
        best_strategy = None
        best_utility = float('-inf')
        i = 0
        
        while i < num_iter:
            print(f"\nIteration {i+1}/{num_iter}")
            
            # Update provider's strategy
            self.provider.rf_price = cur_strategy[0]
            self.provider.vlc_price = cur_strategy[1]
            self.provider.rf_tx_power = cur_strategy[2]
            self.provider.vlc_tx_power = cur_strategy[3]
            self.provider.rf_bandwidth = cur_strategy[4]
            self.provider.vlc_bandwidth = cur_strategy[5]
            
            print(f"Current strategy: {[round(x, 6) for x in cur_strategy]}")
            
            # Calculate user responses to current provider strategy
            self.users_response()
            
            # Check user choices for debugging
            user_choices = [user.selected_network for user in self.users]
            rf_users = user_choices.count('RF')
            vlc_users = user_choices.count('VLC')
            print(f"User choices - RF: {rf_users}, VLC: {vlc_users}, None: {len(user_choices) - rf_users - vlc_users}")
            
            # Use different optimization methods to avoid local minima
            # Try with a tighter tolerance first
            result = minimize(
                self.provider_objective,
                cur_strategy,
                bounds=bounds,
                method='L-BFGS-B',
                options={'ftol': 1e-8, 'gtol': 1e-8}
            )
            
            if not result.success:
                print("L-BFGS-B optimization failed, trying with SLSQP...")
                # Try another method if L-BFGS-B fails
                result = minimize(
                    self.provider_objective,
                    cur_strategy,
                    bounds=bounds,
                    method='SLSQP',
                    options={'ftol': 1e-8}
                )
                
                if not result.success:
                    print("All optimization attempts failed")
                    return False, 0  # Optimization failed
            
            # Get new optimal strategy
            new_strategy = result.x
            current_utility = -result.fun
            
            print(f"Optimization result: {result.success}, Function value: {-result.fun}")
            print(f"New strategy: {[round(x, 6) for x in new_strategy]}")
            
            # Check if we've reached equilibrium (strategy hasn't changed significantly)
            strategy_change = sum(abs(new_strategy[j] - cur_strategy[j]) for j in range(len(cur_strategy)))
            utility_change = abs(current_utility - prev_utility)
            
            print(f"Strategy change: {strategy_change}, Utility change: {utility_change}")
            
            # Track best strategy seen so far
            if current_utility > best_utility:
                best_utility = current_utility
                best_strategy = new_strategy.copy()
                print(f"New best strategy found with utility: {best_utility}")
            
            # More lenient convergence criteria to allow exploration
            if i > 0 and strategy_change < 1e-4 and utility_change < 1e-4:
                print("Convergence detected, breaking loop")
                if best_strategy is not None:
                    cur_strategy = best_strategy  # Use the best strategy found
                else:
                    cur_strategy = new_strategy
                break
                
            # Update for next iteration
            cur_strategy = new_strategy
            prev_utility = current_utility
            i += 1
            
        # Final update of provider's strategy with the optimal values
        self.provider.rf_price = cur_strategy[0]
        self.provider.vlc_price = cur_strategy[1]
        self.provider.rf_tx_power = cur_strategy[2]
        self.provider.vlc_tx_power = cur_strategy[3]
        self.provider.rf_bandwidth = cur_strategy[4]
        self.provider.vlc_bandwidth = cur_strategy[5]
        
        # Calculate final user responses
        self.users_response()
        
        # Get final user allocations for debug info
        user_choices = [user.selected_network for user in self.users]
        rf_users = user_choices.count('RF')
        vlc_users = user_choices.count('VLC')
        no_network = len(user_choices) - rf_users - vlc_users
        
        print("\nFinal solution:")
        print(f"Strategy: {[round(x, 6) for x in cur_strategy]}")
        print(f"Provider utility: {prev_utility}")
        print(f"User allocations - RF: {rf_users}, VLC: {vlc_users}, None: {no_network}")
        
        # Check if the provider objective function actually depends on user responses
        # This is a sanity check to detect if there's an implementation issue
        self.users_response()  # Calculate user responses for current strategy
        utility1 = -self.provider_objective(cur_strategy)
        
        # Change user responses artificially (for testing purposes only)
        for user in self.users:
            if user.selected_network == 'RF':
                user.chosen_network = 'VLC'
            elif user.selected_network == 'VLC':
                user.chosen_network = 'RF'
        
        utility2 = -self.provider_objective(cur_strategy)
        
        if abs(utility1 - utility2) < 1e-6:
            print("\nWARNING: Provider objective function may not be properly considering user responses!")
            print("This could explain why optimization converges immediately.")
        
        # Restore correct user responses
        self.users_response()
        
        return True, prev_utility  # Return success and optimal utility 
    """
    
    def solve(self, num_iter = 100):
        # Solve the Stackelberg game
        # Initial strategy values
        initial_strategy = [
            RF_FIXED_COST,                # RF price
            VLC_FIXED_COST,               # VLC price
            RF_TRANSMIT_POWER_MAX * 0.5,  # RF tx power
            VLC_TRANSMIT_POWER_MAX * 0.5, # VLC tx power
            RF_BANDWIDTH_MAX * 0.5,       # RF bandwidth
            VLC_BANDWIDTH_MAX * 0.5       # VLC bandwidth
        ]
        
        # Strategy bounds
        bounds = [
            (1.0, 10.0),                          # RF price bounds
            (1.0, 10.0),                          # VLC price bounds
            (0.1 * RF_TRANSMIT_POWER_MAX, RF_TRANSMIT_POWER_MAX),    # RF tx power bounds
            (0.1 * VLC_TRANSMIT_POWER_MAX, VLC_TRANSMIT_POWER_MAX),  # VLC tx power bounds
            (0.1 * RF_BANDWIDTH_MAX, RF_BANDWIDTH_MAX),              # RF bandwidth bounds
            (0.1 * VLC_BANDWIDTH_MAX, VLC_BANDWIDTH_MAX)             # VLC bandwidth bounds
        ]

        cur_strategy = initial_strategy

        i = 0
        while i<=num_iter:
            for user in self.users:
                self.user_objective(user=user)

            # Solve optimization problem
            result = minimize(
                self.provider_objective,
                cur_strategy,
                bounds=bounds,
                method='L-BFGS-B'
            )

            optimal_strategy = result.x
            self.provider.rf_price = optimal_strategy[0]
            self.provider.vlc_price = optimal_strategy[1]
            self.provider.rf_tx_power = optimal_strategy[2]
            self.provider.vlc_tx_power = optimal_strategy[3]
            self.provider.rf_bandwidth = optimal_strategy[4]
            self.provider.vlc_bandwidth = optimal_strategy[5]

            cur_strategy = optimal_strategy

            i+=1

            # if self.verify_stackelberg_equilibrium() or i>num_iter:
            #     break
        
        if result.success:
            # Update provider's strategy with optimal values
            optimal_strategy = result.x
            self.provider.rf_price = optimal_strategy[0]
            self.provider.vlc_price = optimal_strategy[1]
            self.provider.rf_tx_power = optimal_strategy[2]
            self.provider.vlc_tx_power = optimal_strategy[3]
            self.provider.rf_bandwidth = optimal_strategy[4]
            self.provider.vlc_bandwidth = optimal_strategy[5]
            
            # Calculate final user responses
            self.users_response()
            
            return True, -result.fun  # Return success and optimal utility
        else:
            return False, 0  # Optimization failed
    

    """
    def solve(self, num_iterations=1000, learning_rate=0.1, tolerance=1e-2):
        # Solve the Stackelberg game by sequentially updating provider's strategy and users' responses.
        converged = False
        last_utility = None
        for iteration in tqdm(range(num_iterations), desc="Solving Stackelberg Game"):
            # 1. Users respond to current provider strategy
            self.users_response()
            # 2. Calculate current provider utility
            current_utility = self.provider.calculate_utility(self.users)
            if last_utility is not None:
                utility_change = abs(current_utility - last_utility)
                if utility_change < tolerance:
                    # Converged
                    converged = True
                    break
            last_utility = current_utility

            # 3. Save the current parameters
            current_params = self.provider.get_network_params()
            # 4. Slightly perturb each provider parameter to estimate gradient
            gradients = {}
            delta = 1e-3  # Small change for finite differences
            for param in ['rf_price', 'vlc_price', 'rf_tx_power', 'vlc_tx_power', 'rf_bandwidth', 'vlc_bandwidth']:
                # Save original value
                original_value = getattr(self.provider, param)

                # Perturb positively
                setattr(self.provider, param, original_value + delta)
                self.users_response()
                utility_plus = self.provider.calculate_utility(self.users)

                # Perturb negatively
                setattr(self.provider, param, original_value - delta)
                self.users_response()
                utility_minus = self.provider.calculate_utility(self.users)

                # Reset to original
                setattr(self.provider, param, original_value)

                # Estimate gradient
                gradient = (utility_plus - utility_minus) / (2 * delta)
                gradients[param] = gradient
            
            # 5. Update provider parameters using the estimated gradients
            for param, grad in gradients.items():
                updated_value = getattr(self.provider, param) + learning_rate * grad
                # Optionally clip updated values within valid ranges
                setattr(self.provider, param, max(0.0, updated_value))  # no negative prices, power etc

        # After training is done, get final users' responses
        self.users_response()
        final_utility = self.provider.calculate_utility(self.users)
        return converged, final_utility
    """ 

    def get_results(self):
        """Get the results of the game"""
        # Count network selections
        n_rf = sum(1 for user in self.users if user.selected_network == "RF")
        n_vlc = sum(1 for user in self.users if user.selected_network == "VLC")
        
        # Get optimal strategy
        optimal_strategy = {
            "RF": {
                "price": self.provider.rf_price,
                "tx_power": self.provider.rf_tx_power,
                "bandwidth": self.provider.rf_bandwidth
            },
            "VLC": {
                "price": self.provider.vlc_price,
                "tx_power": self.provider.vlc_tx_power,
                "bandwidth": self.provider.vlc_bandwidth
            }
        }
        
        # Calculate average utilities
        avg_utility_rf = np.mean([user.utilities["RF"] for user in self.users])
        avg_utility_vlc = np.mean([user.utilities["VLC"] for user in self.users])
        
        provider_utility = self.provider.calculate_utility(self.users)
        
        # Calculate network capacity
        capacity = self.provider.calculate_network_capacity(self.users)
        
        results = {
            "optimal_strategy": optimal_strategy,
            "network_selection": {
                "RF": n_rf,
                "VLC": n_vlc
            },
            "avg_user_utility": {
                "RF": avg_utility_rf,
                "VLC": avg_utility_vlc
            },
            "provider_utility": provider_utility,
            "capacity": capacity
        }
        
        return results
    
    def visualize_results(self):
        """Visualize the results"""
        plt.figure(figsize=(10, 8))
        
        # Plot room boundaries
        plt.plot([0, ROOM_SIZE_X, ROOM_SIZE_X, 0, 0], 
                 [0, 0, ROOM_SIZE_Y, ROOM_SIZE_Y, 0], 'k-', linewidth=2)
        
        # Plot RF base stations
        for bs_pos in self.provider.rf_bs_positions:
            plt.plot(bs_pos[0], bs_pos[1], 'bs', markersize=10, label='RF BS')
        
        # Plot VLC access points
        for ap_pos in self.provider.vlc_ap_positions:
            plt.plot(ap_pos[0], ap_pos[1], 'r^', markersize=10, label='VLC AP')
        
        # Plot users
        for user in self.users:
            if user.selected_network == "RF":
                plt.plot(user.x, user.y, 'bo', alpha=0.7)
            else:  # VLC
                plt.plot(user.x, user.y, 'ro', alpha=0.7)
        
        # Remove duplicate labels
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())
        
        plt.xlabel('X position (m)')
        plt.ylabel('Y position (m)')
        plt.title('Network Selection Results (Blue: RF, Red: VLC)')
        plt.grid(True)
        plt.axis('equal')
        
        plt.tight_layout()
        plt.show()

# ------------ ADDITIONAL VISUALIZATIONS ------------

def plot_snr_vs_distance():
    """Plot SNR vs. distance for both RF and VLC networks"""
    # Distance range
    distances = np.linspace(0.1, 10, 100)
    
    # Calculate SNR for RF
    rf_snr_values = []
    for d in distances:
        # Simple path loss model for RF
        path_loss_db = 20 * np.log10(RF_FREQ) + 35 * np.log10(d) - 147.5
        channel_gain = 10**(-path_loss_db/10)
        snr = RF_TRANSMIT_POWER_MAX * channel_gain / (RF_NOISE_DENSITY * RF_BANDWIDTH_MAX)
        snr_db = 10 * np.log10(snr)
        rf_snr_values.append(snr_db)
    
    # Calculate SNR for VLC
    vlc_snr_values = []
    height = 2.5  # Ceiling height
    for d in distances:
        if d > 10:  # Limit to reasonable distances
            vlc_snr_values.append(float('nan'))
            continue
            
        # Calculate 3D distance
        distance_3d = np.sqrt(d**2 + height**2)
        
        # Angle of irradiance
        phi = np.arctan(d / height)
        
        # Lambertian order
        m = -np.log(2) / np.log(np.cos(np.radians(60)))
        
        # Detector area
        A = 1e-4  # m^2
        
        # Calculate channel gain using Lambertian model
        if phi <= np.pi/2:  # Within FOV
            channel_gain = (m+1) * A * 1.0 * 1.0 * np.cos(phi)**(m+1) / (2 * np.pi * distance_3d**2)
            snr = (VLC_RESPONSIVITY * VLC_TRANSMIT_POWER_MAX * channel_gain) / (VLC_NOISE_DENSITY * VLC_BANDWIDTH_MAX)
            snr_db = 10 * np.log10(snr)
            vlc_snr_values.append(snr_db)
        else:
            vlc_snr_values.append(float('nan'))
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(distances, rf_snr_values, 'b-', linewidth=2, label='RF')
    plt.plot(distances, vlc_snr_values, 'r-', linewidth=2, label='VLC')
    plt.xlabel('Horizontal Distance (m)')
    plt.ylabel('SNR (dB)')
    plt.title('SNR vs. Distance for RF and VLC Networks')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def simulate_increasing_users(start_val=5, max_users=25, inc=1):
    """Simulate system with increasing number of users and plot results"""
    user_counts = list(range(start_val, max_users + 1, inc))
    
    # Results containers
    rf_capacities = []
    vlc_capacities = []
    rf_unallocated = []
    vlc_unallocated = []
    rf_user_utilities = []
    vlc_user_utilities = []
    total_user_utilities = []
    avg_provider_utilities = []
    rf_allocated = []
    vlc_allocated = []

    users = []
    for i in range(start_val):
        x = np.random.uniform(0, ROOM_SIZE_X)
        y = np.random.uniform(0, ROOM_SIZE_Y)
        
        # Generate random preference weights
        alpha_values = np.random.uniform(0.1, 1.0, 4)

        duty_cycle = np.random.rand()
        
        users.append(User(i, x, y, alpha_values, duty_cycle=duty_cycle))
    
    for num_users in tqdm(user_counts, desc="Simulating increasing users"):
        # Run stackelberg game with this many users
        game = StackelbergGame(num_users, users)
        success, _ = game.solve()
        
        results = game.get_results()
        
        # Store capacity results
        rf_capacities.append(results["capacity"]["RF"]["total"])
        vlc_capacities.append(results["capacity"]["VLC"]["total"])
        rf_unallocated.append(results["capacity"]["RF"]["unallocated"])
        vlc_unallocated.append(results["capacity"]["VLC"]["unallocated"])
        rf_allocated.append(results["capacity"]["RF"]["allocated"])
        vlc_allocated.append(results["capacity"]["VLC"]["allocated"])
        
        # Calculate throughput per user
        n_rf_users = results["network_selection"]["RF"]
        n_vlc_users = results["network_selection"]["VLC"]

        rf_user_utilities.append(results["avg_user_utility"]["RF"]*n_rf_users)
        vlc_user_utilities.append(results["avg_user_utility"]["VLC"]*n_vlc_users)
        total_user_utilities.append(results["avg_user_utility"]["RF"]*n_rf_users + results["avg_user_utility"]["VLC"]*n_vlc_users)
        avg_provider_utilities.append(results["provider_utility"])
        
        # Generate random user positions
        for i in range(inc):
            x = np.random.uniform(0, ROOM_SIZE_X)
            y = np.random.uniform(0, ROOM_SIZE_Y)
            
            # Generate random preference weights
            alpha_values = np.random.uniform(0.1, 1.0, 4)

            duty_cycle = np.random.rand()
            
            users.append(User(i, x, y, alpha_values, duty_cycle=duty_cycle))
    
    # Plot 1: Unallocated available capacity vs number of users
    plt.figure(figsize=(10, 6))
    plt.plot(user_counts, rf_unallocated, 'b-', marker='o', label='RF Unallocated')
    plt.plot(user_counts, vlc_unallocated, 'r-', marker='s', label='VLC Unallocated')
    plt.xlabel('Number of Users')
    plt.ylabel('Unallocated Capacity (Mbps)')
    plt.title('Unallocated Available Capacity vs. Number of Users')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Plot 2: Total User utility vs number of users
    plt.figure(figsize=(10, 6))
    plt.plot(user_counts, total_user_utilities, 'b-', marker='o', label='Total User Utility')
    plt.plot(user_counts, rf_user_utilities, 'r-', marker='o', label='RF User Utility')
    plt.plot(user_counts, vlc_user_utilities, 'g-', marker='o', label='VLC User Utility')
    plt.xlabel('Number of Users')
    plt.ylabel('User Utility')
    plt.title('User Utility vs. Number of Users')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # Plot 3: Allocated capacity vs number of users
    plt.figure(figsize=(10, 6))
    plt.plot(user_counts, rf_allocated, 'b-', marker='o', label='RF Allocated')
    plt.plot(user_counts, vlc_allocated, 'r-', marker='s', label='VLC Allocated')
    plt.xlabel('Number of Users')
    plt.ylabel('Allocated Capacity (Mbps)')
    plt.title('Allocated Capacity vs. Number of Users')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Plot 4: Provider utility vs number of users
    plt.figure(figsize=(10, 6))
    plt.plot(user_counts, avg_provider_utilities, 'b-', marker='o', label='Provider utility')
    plt.xlabel('Number of Users')
    plt.ylabel('Average Provider Utility')
    plt.title('Average Provider Utility vs. Number of Users')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

# ------------ MAIN FUNCTION ------------

def run_stackelberg_game(num_users=NUM_USERS, visualize_all=True):
    """Run the Stackelberg game and display results"""
    # Initialize the game
    game = StackelbergGame(num_users)
    
    print("Solving Stackelberg game...")
    success, optimal_utility = game.solve()
    
    if success:
        print("Game solved successfully!")
        results = game.get_results()
        
        print("\nOPTIMAL PROVIDER STRATEGY:")
        print(f"RF Price: {results['optimal_strategy']['RF']['price']:.2f}")
        print(f"VLC Price: {results['optimal_strategy']['VLC']['price']:.2f}")
        print(f"RF Transmit Power: {results['optimal_strategy']['RF']['tx_power']:.2f} W")
        print(f"VLC Transmit Power: {results['optimal_strategy']['VLC']['tx_power']:.2f} W")
        print(f"RF Bandwidth: {results['optimal_strategy']['RF']['bandwidth']/1e6:.2f} MHz")
        print(f"VLC Bandwidth: {results['optimal_strategy']['VLC']['bandwidth']/1e6:.2f} MHz")
        
        print("\nNETWORK SELECTION RESULTS:")
        print(f"RF Users: {results['network_selection']['RF']}")
        print(f"VLC Users: {results['network_selection']['VLC']}")
        
        print("\nAVERAGE USER UTILITY:")
        print(f"RF: {results['avg_user_utility']['RF']:.4f}")
        print(f"VLC: {results['avg_user_utility']['VLC']:.4f}")
        
        print("\nPROVIDER UTILITY:")
        print(f"{results['provider_utility']:.4f}")
        
        print("\nNETWORK CAPACITY:")
        print(f"RF Total: {results['capacity']['RF']['total']:.2f} Mbps")
        print(f"RF Allocated: {results['capacity']['RF']['allocated']:.2f} Mbps")

        # Visualize results
        game.visualize_results()
        
        return game, results
    else:
        print("Failed to solve!")
        return None, None

if __name__ == "__main__":
    game, results = run_stackelberg_game()
    pprint(results)
    plot_snr_vs_distance()
    simulate_increasing_users()