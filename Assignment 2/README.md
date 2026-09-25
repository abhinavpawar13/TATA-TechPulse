[README.md](https://github.com/user-attachments/files/32665569/README.md)
# Tata Technologies - TechPulse FY-26: Applied AI & ML
## Lab Statement 2: Simulated Driving Agent Behavior

**Curriculum Unit:** Unit 1 – Introduction to AI & ML (Evolutionary Algorithms, Agent Behavior & Autonomous Driving)  
**Lab Statement 2:** *Build a self-driving agent using Python and genetic algorithms.*  
**Domain Focus:** Autonomous Vehicle Control, Collision Avoidance & Neuroevolutionary Path Planning  

---

## 📌 1. Overview & Objectives

In autonomous automotive systems, self-driving agents must navigate dynamic track environments, negotiate high-speed curves, and avoid boundary collisions using limited sensor inputs. When gradient-based reinforcement learning is slow or unstable, **Genetic Algorithms (GAs)** combined with neural network policy controllers provide a powerful evolutionary optimization technique.

This lab implements an end-to-end continuous self-driving simulation where a population of autonomous vehicles evolves to master a multi-curve racing circuit using **Python**, **NumPy**, and **Matplotlib**.

### Key Learning Outcomes:
1. **Agent Architecture**: Model an autonomous vehicle with kinematic state dynamics (position, velocity, heading angle, steering limits).
2. **Raycast Distance Sensing (Virtual LIDAR)**: Implement 5 virtual range sensors measuring distances to track boundaries at $\{-60^\circ, -30^\circ, 0^\circ, +30^\circ, +60^\circ\}$.
3. **Neural Policy Chromosome Encoding**: Map weights and biases of a multi-layer controller into a continuous real-valued genetic chromosome ($74$ genes).
4. **Multi-Objective Fitness Function**: Reward forward circuit progression along checkpoints, velocity, and distance while penalizing wall collisions.
5. **Evolutionary Operators**: Apply **Tournament Selection** ($k=3$), **Arithmetic Crossover**, **Gaussian Mutation**, and **Elitism** across generations.
6. **Diagnostic Evaluation**: Analyze generational trajectory improvements, fitness convergence curves, and lap completion statistics.

---

## 📁 2. Project Directory Structure

```text
lab2_simulated_driving_agent/
│
├── README.md                                      # Comprehensive lab manual & technical documentation
├── driving_agent_simulation.py                    # Complete autonomous vehicle simulation & GA engine
├── lab2_simulated_driving_agent.ipynb             # Interactive Jupyter Notebook
├── driving_track_coords.csv                       # Track boundaries & centerline coordinates dataset
├── simulation_logs.csv                            # Generational telemetry log (fitness, lap %, distance)
├── requirements.txt                               # Minimal project dependencies
│
└── plots/                                         # Diagnostic visualizations (200 DPI)
    ├── 01_track_and_best_agent_trajectories.png    # Track map showing best vehicle trajectories across generations
    ├── 02_fitness_evolution_curve.png             # Generational convergence (Peak, Mean & Minimum fitness)
    └── 03_lap_completion_and_distance.png         # Lap progress (%) and distance traveled per generation
```

---

## 🔬 3. Mathematical Formulations & Agent Dynamics

### 3.1 Kinematic Vehicle Dynamics
The vehicle state $\mathbf{s}_t = [x_t, y_t, \theta_t, v_t]^T$ evolves via discrete kinematic equations:

$$\theta_{t+1} = \theta_t + \delta_t \cdot \delta_{\max} \cdot \left(\frac{v_t}{v_{\text{ref}}}\right) \cdot \Delta t$$
$$x_{t+1} = x_t + v_t \cos(\theta_{t+1}) \Delta t$$
$$y_{t+1} = y_t + v_t \sin(\theta_{t+1}) \Delta t$$

Where $\delta_t \in [-1, 1]$ is the steering action, $\delta_{\max} = 32^\circ$, and $\Delta t = 0.2\text{ s}$.

### 3.2 Neural Policy Forward Pass
The agent's decision engine is a neural network mapping 6 inputs (5 LIDAR ranges + normalized speed) to steering and throttle:

$$\mathbf{h} = \tanh(\mathbf{W}_1 \mathbf{x} + \mathbf{b}_1)$$
$$\text{Steering } \delta = \tanh(\mathbf{w}_{\text{steer}}^T \mathbf{h} + b_{\text{steer}})$$
$$\text{Throttle } a = \sigma(\mathbf{w}_{\text{throt}}^T \mathbf{h} + b_{\text{throt}}) = \frac{1}{1 + e^{-z}}$$

- **Chromosome Size**: $(6 \times 8 + 8) + (8 \times 2 + 2) = 56 + 18 = 74 \text{ real-valued parameters}$.

### 3.3 Fitness Function
$$\text{Fitness} = 100 \cdot N_{\text{checkpoints}} + 1.5 \cdot d_{\text{traveled}} + 0.2 \cdot t_{\text{steps}}$$

Collisions with inner or outer boundary walls immediately terminate the agent's run.

### 3.4 Genetic Operators
- **Selection**: Tournament Selection ($k=3$).
- **Crossover**: Arithmetic Blending:
  $$\mathbf{c}_1 = \alpha \mathbf{p}_1 + (1 - \alpha) \mathbf{p}_2, \quad \mathbf{c}_2 = (1 - \alpha) \mathbf{p}_1 + \alpha \mathbf{p}_2, \quad \alpha \sim U(0.2, 0.8)$$
- **Mutation**: Adaptive Gaussian Perturbation ($p_m = 0.08, \sigma = 0.35$).
- **Elitism**: Top-2 fittest agents are preserved untouched into the subsequent generation.

---

## 🚀 4. How to Run

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Run the Simulation & Evolutionary Optimization
```bash
python driving_agent_simulation.py
```

### Step 3: Open the Interactive Jupyter Notebook
```bash
jupyter notebook lab2_simulated_driving_agent.ipynb
```
