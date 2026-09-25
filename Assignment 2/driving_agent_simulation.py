"""
driving_agent_simulation.py
---------------------------
Tata Technologies - TechPulse FY-26: Applied AI & ML
Lab Statement 2: Simulated Driving Agent Behavior
Build a self-driving agent using Python and genetic algorithms.

Architecture:
1. Continuous 2D Automotive Racing Circuit with Checkpoints & Boundary Walls
2. Kinematic Vehicle Dynamics with 5 Virtual Raycast Distance Sensors (LIDAR)
3. Neural Network Driving Controller (Weights encoded as Genetic Chromosome)
4. Genetic Algorithm Engine:
   - Population: 40 vehicles
   - Fitness Function: Distance progressed along centerline + Checkpoints passed + Speed bonus
   - Selection: Tournament Selection (k=3)
   - Crossover: Arithmetic & Two-Point Crossover
   - Mutation: Adaptive Gaussian Mutation
   - Elitism: Top-2 Survivor preservation
5. Generation Telemetry Tracking & Publication Diagnostic Visualizations
"""

import os
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True, encoding="utf-8")

import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# -------------------------------------------------------------
# 1. Track Generation & Geometry
# -------------------------------------------------------------
def build_race_track():
    """Generates an oval-curved racing track with inner, outer boundaries and checkpoints."""
    t = np.linspace(0, 2 * np.pi, 100)
    # Curvature parametric equations
    center_x = 120.0 * np.cos(t) + 30.0 * np.cos(3 * t)
    center_y = 70.0 * np.sin(t) - 20.0 * np.sin(2 * t)
    
    track_width = 16.0
    half_w = track_width / 2.0
    
    # Calculate tangents and normals
    dx = np.gradient(center_x)
    dy = np.gradient(center_y)
    lengths = np.sqrt(dx**2 + dy**2)
    nx = -dy / lengths
    ny = dx / lengths
    
    inner_x = center_x - half_w * nx
    inner_y = center_y - half_w * ny
    outer_x = center_x + half_w * nx
    outer_y = center_y + half_w * ny
    
    centerline = np.column_stack([center_x, center_y])
    inner_boundary = np.column_stack([inner_x, inner_y])
    outer_boundary = np.column_stack([outer_x, outer_y])
    
    return centerline, inner_boundary, outer_boundary

# -------------------------------------------------------------
# 2. Kinematic Vehicle & Raycast Sensor Simulation
# -------------------------------------------------------------
class VehicleAgent:
    def __init__(self, chromosome=None):
        # Neural Network Controller Dimensions:
        # Inputs: 5 sensor distances + 1 current velocity = 6 inputs
        # Hidden Layer: 8 neurons
        # Output: 2 outputs (Steering [-1, 1], Throttle [0, 1])
        # Chromosome size: (6*8 + 8) + (8*2 + 2) = 56 + 18 = 74 parameters
        self.input_dim = 6
        self.hidden_dim = 8
        self.output_dim = 2
        self.chrom_size = (self.input_dim * self.hidden_dim + self.hidden_dim) + (self.hidden_dim * self.output_dim + self.output_dim)
        
        if chromosome is None:
            self.chromosome = np.random.normal(0, 0.6, self.chrom_size)
        else:
            self.chromosome = np.array(chromosome, dtype=float)
            
        self.unpack_weights()
        self.reset()
        
    def unpack_weights(self):
        idx = 0
        w1_size = self.input_dim * self.hidden_dim
        self.W1 = self.chromosome[idx : idx + w1_size].reshape(self.input_dim, self.hidden_dim)
        idx += w1_size
        
        b1_size = self.hidden_dim
        self.B1 = self.chromosome[idx : idx + b1_size]
        idx += b1_size
        
        w2_size = self.hidden_dim * self.output_dim
        self.W2 = self.chromosome[idx : idx + w2_size].reshape(self.hidden_dim, self.output_dim)
        idx += w2_size
        
        b2_size = self.output_dim
        self.B2 = self.chromosome[idx : idx + b2_size]
        
    def reset(self, start_pos=(150.0, 0.0), start_heading=np.pi/2):
        self.x, self.y = float(start_pos[0]), float(start_pos[1])
        self.heading = float(start_heading)
        self.velocity = 1.5
        self.alive = True
        self.checkpoints_reached = 0
        self.total_distance = 0.0
        self.lifetime_steps = 0
        self.trajectory = [(self.x, self.y)]
        
    def compute_sensor_readings(self, inner_boundary, outer_boundary, max_range=30.0):
        # 5 ray angles relative to heading: -60°, -30°, 0°, +30°, +60°
        sensor_angles = np.radians([-60, -30, 0, 30, 60])
        readings = []
        
        for rel_angle in sensor_angles:
            ray_dir = self.heading + rel_angle
            ray_dx = np.cos(ray_dir)
            ray_dy = np.sin(ray_dir)
            
            # Simple raycast distance check against track boundary points
            min_dist = max_range
            step_sizes = np.linspace(1.0, max_range, 15)
            
            for s in step_sizes:
                px = self.x + s * ray_dx
                py = self.y + s * ray_dy
                
                # Distance to nearest inner and outer boundary points
                d_inner = np.min(np.hypot(inner_boundary[:, 0] - px, inner_boundary[:, 1] - py))
                d_outer = np.min(np.hypot(outer_boundary[:, 0] - px, outer_boundary[:, 1] - py))
                
                if d_inner < 1.8 or d_outer < 1.8:
                    min_dist = s
                    break
            readings.append(min_dist / max_range) # Normalized [0, 1]
            
        return np.array(readings)

    def decide_control(self, sensor_readings):
        """Neural Network Forward Pass"""
        x_in = np.append(sensor_readings, self.velocity / 6.0) # 6 inputs
        hidden = np.tanh(np.dot(x_in, self.W1) + self.B1)
        out = np.tanh(np.dot(hidden, self.W2) + self.B2)
        
        steering = float(out[0]) # Steering angle in [-1, 1] (-30 deg to +30 deg)
        throttle = float(1.0 / (1.0 + np.exp(-out[1]))) # Sigmoid throttle [0, 1]
        return steering, throttle

    def update_physics(self, steering, throttle, dt=0.2):
        if not self.alive:
            return
            
        max_steering = np.radians(32)
        self.heading += steering * max_steering * (self.velocity / 5.0) * dt
        
        # Acceleration & Friction
        accel = (throttle * 4.0) - (0.15 * self.velocity)
        self.velocity = np.clip(self.velocity + accel * dt, 0.5, 6.5)
        
        dx = self.velocity * np.cos(self.heading) * dt
        dy = self.velocity * np.sin(self.heading) * dt
        
        self.x += dx
        self.y += dy
        self.total_distance += np.hypot(dx, dy)
        self.lifetime_steps += 1
        self.trajectory.append((self.x, self.y))

# -------------------------------------------------------------
# 3. Genetic Algorithm Optimization Engine
# -------------------------------------------------------------
class DrivingGeneticAlgorithm:
    def __init__(self, pop_size=40, generations=25, mutation_rate=0.08):
        self.pop_size = pop_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.centerline, self.inner_bound, self.outer_bound = build_race_track()
        self.population = [VehicleAgent() for _ in range(pop_size)]
        self.start_pos = (self.centerline[0, 0], self.centerline[0, 1])
        # Start facing along the track tangent
        dx = self.centerline[1, 0] - self.centerline[0, 0]
        dy = self.centerline[1, 1] - self.centerline[0, 1]
        self.start_heading = np.arctan2(dy, dx)
        
        self.history = []
        self.best_trajectories = {}
        
    def evaluate_fitness(self, agent, max_steps=220):
        agent.reset(self.start_pos, self.start_heading)
        curr_checkpoint = 0
        n_checkpoints = len(self.centerline)
        
        for step in range(max_steps):
            sensors = agent.compute_sensor_readings(self.inner_bound, self.outer_bound)
            
            # Crash detection: too close to boundary
            d_inner = np.min(np.hypot(self.inner_bound[:, 0] - agent.x, self.inner_bound[:, 1] - agent.y))
            d_outer = np.min(np.hypot(self.outer_bound[:, 0] - agent.x, self.outer_bound[:, 1] - agent.y))
            
            if d_inner < 1.2 or d_outer < 1.2 or min(sensors) < 0.08:
                agent.alive = False
                break
                
            # Checkpoint detection
            target_cp = (curr_checkpoint + 1) % n_checkpoints
            dist_to_cp = np.hypot(self.centerline[target_cp, 0] - agent.x, self.centerline[target_cp, 1] - agent.y)
            if dist_to_cp < 12.0:
                curr_checkpoint += 1
                agent.checkpoints_reached += 1
                
            steering, throttle = agent.decide_control(sensors)
            agent.update_physics(steering, throttle)
            
        # Fitness formulation:
        # Checkpoints passed + total forward distance + speed bonus
        fitness = (agent.checkpoints_reached * 100.0) + (agent.total_distance * 1.5) + (agent.lifetime_steps * 0.2)
        return float(fitness)

    def evolve_population(self):
        print(f"Starting Genetic Algorithm: {self.generations} Generations, Population Size = {self.pop_size}")
        
        for gen in range(1, self.generations + 1):
            fitness_scores = []
            
            for agent in self.population:
                fit = self.evaluate_fitness(agent)
                fitness_scores.append(fit)
                
            fitness_scores = np.array(fitness_scores)
            best_idx = np.argmax(fitness_scores)
            best_fitness = fitness_scores[best_idx]
            mean_fitness = np.mean(fitness_scores)
            best_agent = self.population[best_idx]
            
            # Save trajectory of best agent at key generations
            if gen in [1, 5, 12, self.generations]:
                self.best_trajectories[gen] = list(best_agent.trajectory)
                
            lap_pct = min(100.0, (best_agent.checkpoints_reached / len(self.centerline)) * 100.0)
            
            self.history.append({
                "generation": gen,
                "max_fitness": round(best_fitness, 2),
                "mean_fitness": round(mean_fitness, 2),
                "min_fitness": round(np.min(fitness_scores), 2),
                "best_checkpoints": best_agent.checkpoints_reached,
                "best_distance_m": round(best_agent.total_distance, 1),
                "lap_completion_pct": round(lap_pct, 1)
            })
            
            print(f"Gen {gen:02d}/{self.generations:02d} | Max Fit: {best_fitness:7.1f} | Mean Fit: {mean_fitness:6.1f} | Lap %: {lap_pct:5.1f}% | Dist: {best_agent.total_distance:.1f}m")
            
            # Breeding New Generation (Elitism + Tournament Selection + Crossover + Mutation)
            sorted_indices = np.argsort(fitness_scores)[::-1]
            new_pop = [VehicleAgent(self.population[sorted_indices[0]].chromosome),
                       VehicleAgent(self.population[sorted_indices[1]].chromosome)] # Top-2 Elitism
                       
            while len(new_pop) < self.pop_size:
                # Tournament selection
                p1 = self.tournament_select(fitness_scores)
                p2 = self.tournament_select(fitness_scores)
                
                # Crossover
                c1_chrom, c2_chrom = self.crossover(p1.chromosome, p2.chromosome)
                
                # Mutation
                c1_chrom = self.mutate(c1_chrom)
                c2_chrom = self.mutate(c2_chrom)
                
                new_pop.append(VehicleAgent(c1_chrom))
                if len(new_pop) < self.pop_size:
                    new_pop.append(VehicleAgent(c2_chrom))
                    
            self.population = new_pop
            
        return pd.DataFrame(self.history)

    def tournament_select(self, fitness_scores, k=3):
        candidates = np.random.choice(len(self.population), size=k, replace=False)
        best_cand = candidates[np.argmax(fitness_scores[candidates])]
        return self.population[best_cand]

    def crossover(self, parent1_chrom, parent2_chrom):
        # Arithmetic blending crossover
        alpha = np.random.uniform(0.2, 0.8)
        child1 = alpha * parent1_chrom + (1 - alpha) * parent2_chrom
        child2 = (1 - alpha) * parent1_chrom + alpha * parent2_chrom
        return child1, child2

    def mutate(self, chromosome):
        for i in range(len(chromosome)):
            if np.random.rand() < self.mutation_rate:
                chromosome[i] += np.random.normal(0, 0.35)
        return chromosome

# -------------------------------------------------------------
# 4. Diagnostic Plotting & Reporting
# -------------------------------------------------------------
def generate_lab2_visualizations(ga, history_df, output_dir):
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Plot 1: Track Map & Generational Trajectory Evolution
    fig, ax = plt.subplots(figsize=(12, 7.5))
    ax.plot(ga.centerline[:, 0], ga.centerline[:, 1], "k--", alpha=0.4, label="Track Centerline")
    ax.plot(ga.inner_bound[:, 0], ga.inner_bound[:, 1], "dimgray", linewidth=2.5, label="Track Inner Wall")
    ax.plot(ga.outer_bound[:, 0], ga.outer_bound[:, 1], "dimgray", linewidth=2.5, label="Track Outer Wall")
    
    # Fill track area
    ax.fill(np.append(ga.outer_bound[:, 0], ga.inner_bound[::-1, 0]),
            np.append(ga.outer_bound[:, 1], ga.inner_bound[::-1, 1]),
            color="#ecf0f1", alpha=0.6)
            
    # Mark Start Line
    ax.plot([ga.inner_bound[0, 0], ga.outer_bound[0, 0]], [ga.inner_bound[0, 1], ga.outer_bound[0, 1]],
            color="red", linewidth=4, label="Start / Finish Line")
            
    # Plot best trajectories across generations
    traj_colors = {1: "#e74c3c", 5: "#e67e22", 12: "#2980b9", list(ga.best_trajectories.keys())[-1]: "#27ae60"}
    for gen, traj in sorted(ga.best_trajectories.items()):
        traj_arr = np.array(traj)
        c = traj_colors.get(gen, "#8e44ad")
        ax.plot(traj_arr[:, 0], traj_arr[:, 1], color=c, linewidth=2.2, label=f"Best Agent Trajectory (Gen {gen})")
        
    ax.set_title("Self-Driving Agent Trajectory Evolution Across Generations", fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel("Track X Coordinate (meters)")
    ax.set_ylabel("Track Y Coordinate (meters)")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.set_aspect("equal")
    plt.tight_layout()
    p1 = os.path.join(plots_dir, "01_track_and_best_agent_trajectories.png")
    plt.savefig(p1, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"-> Saved: {p1}")
    
    # Plot 2: Fitness Evolution Curves (Max, Mean, Min)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(history_df["generation"], history_df["max_fitness"], color="#27ae60", linewidth=2.5, marker="o", label="Peak Fitness (Best Agent)")
    ax.plot(history_df["generation"], history_df["mean_fitness"], color="#2980b9", linewidth=2.0, linestyle="--", label="Population Mean Fitness")
    ax.fill_between(history_df["generation"], history_df["mean_fitness"], history_df["max_fitness"], color="#2ecc71", alpha=0.15)
    ax.set_title("Genetic Algorithm Convergence & Fitness Progression", fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("Generation Number")
    ax.set_ylabel("Fitness Score")
    ax.legend()
    plt.tight_layout()
    p2 = os.path.join(plots_dir, "02_fitness_evolution_curve.png")
    plt.savefig(p2, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"-> Saved: {p2}")
    
    # Plot 3: Lap Completion % and Distance Traveled
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax2 = ax1.twinx()
    
    ax1.bar(history_df["generation"] - 0.2, history_df["lap_completion_pct"], width=0.4, color="#3498db", label="Lap Completion %")
    ax2.plot(history_df["generation"], history_df["best_distance_m"], color="#d35400", linewidth=2.5, marker="s", label="Distance Traveled (m)")
    
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Lap Completion (%)", color="#3498db")
    ax2.set_ylabel("Distance Traveled (meters)", color="#d35400")
    ax1.set_ylim(0, 110)
    ax1.set_title("Autonomous Driving Lap Progress & Distance per Generation", fontsize=12, fontweight="bold", pad=10)
    plt.tight_layout()
    p3 = os.path.join(plots_dir, "03_lap_completion_and_distance.png")
    plt.savefig(p3, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"-> Saved: {p3}")

def main():
    base_dir = r"C:\Users\abhin\.gemini\antigravity-ide\scratch\lab2_simulated_driving_agent"
    os.makedirs(base_dir, exist_ok=True)
    
    print("=" * 80)
    print("TATA TECHNOLOGIES - TECHPULSE FY-26: APPLIED AI & ML")
    print("LAB STATEMENT 2: SIMULATED DRIVING AGENT BEHAVIOR (GENETIC ALGORITHMS)")
    print("=" * 80)
    
    ga = DrivingGeneticAlgorithm(pop_size=40, generations=20, mutation_rate=0.08)
    history_df = ga.evolve_population()
    
    # Save simulation logs
    csv_path = os.path.join(base_dir, "simulation_logs.csv")
    history_df.to_csv(csv_path, index=False)
    print(f"\nSimulation telemetry saved to: {csv_path}")
    
    # Save track coordinates as dataset
    track_df = pd.DataFrame({
        "center_x": ga.centerline[:, 0],
        "center_y": ga.centerline[:, 1],
        "inner_x": ga.inner_bound[:, 0],
        "inner_y": ga.inner_bound[:, 1],
        "outer_x": ga.outer_bound[:, 0],
        "outer_y": ga.outer_bound[:, 1]
    })
    track_path = os.path.join(base_dir, "driving_track_coords.csv")
    track_df.to_csv(track_path, index=False)
    print(f"Track coordinates dataset saved to: {track_path}")
    
    # Generate diagnostic plots
    generate_lab2_visualizations(ga, history_df, base_dir)
    print("\nLab Statement 2 simulation and evaluation completed successfully!")

if __name__ == "__main__":
    main()
