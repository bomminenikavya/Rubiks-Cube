System C Hybrid Model: RL + DeepCubeA

Introduction:

Our project creates an AI system that can solve a 2×2 Rubik’s Cube by actually learning how to solve it. Instead of giving it fixed rules, we teach the AI using deep learning, reinforcement learning, and a smart search method. The neural network learns to estimate how close any cube is to being solved. Then, a reinforcement learning model learns which moves are helpful through trial and error. Finally, a best-first search uses what the AI has learned to find the shortest path to a solution.

Architecture:

The system models the 2×2×2 Rubik’s Cube using a compact corner-based representation. A Deep Cube A-style neural network estimates how far a state is from being solved, while a DQN agent learns which moves are most effective. A Best-First Search algorithm uses these learned predictions to explore the most promising states. All components—the cube model, heuristic network, RL agent, replay buffer, and search engine—work together to turn a scrambled cube into a complete solution efficiently.

Usage 

This project smartly solves the 2×2×2 Rubik’s Cube by bringing together deep learning, reinforcement learning, and heuristic search. It shows how an AI system can learn cube patterns on its own, understand how close a state is to being solved, decide the best moves, and find an efficient solution path. Overall, this project helps in understanding how different AI techniques can work together, evaluate solver performance, and apply learned strategies to other real-world problem-solving and decision-making tasks.

Cube Simulation (cube2x2.py):

Represents the cube using 8 corner pieces
Each corner has a position (perm) and orientation (orient)
Provides move operations like U, U’, R, F’, etc.
Converts cube state into one-hot encoding for ML models

Supervised Learning Model (DCA):   
 
The supervised model learns to predict how many moves away a cube state is from being solved. Training pipeline:
Generate dataset
python generate_supervised.py

Train the model

python train_supervised.py

Outputs:

supervised_data.bin
dca.bin (trained model)

Reinforcement Learning (DQN):

The RL agent learns which moves reduce the cube distance-to-solve.

Training:

python train_rl.py

Produces:

dqn.bin (trained Q-network)


Solver System (solve.py):

The solver uses:

A* Search
DCA model predictions as heuristic
Cube simulation to expand states

Outputs include:

Scramble depth
Scramble sequence
Number of moves in solution
Solution path
Time taken

Expected Output:
=== SOLVER START ===
Scramble depth: 6
Scramble sequence: ['L', "B'", "D'", 'L', "F'", 'L']

=== SOLVED ===
Moves used: 12
Solution path: ['U', 'R', 'F', ...]
Time taken: 0.012 seconds
