import numpy as np

# Define the chess environment
n_states = 64  # Number of squares on the chessboard
n_actions = 8  # Number of possible moves

# Define the rewards for each state (square)
rewards = np.zeros(n_states)
rewards[63] = 10  # Assign a higher reward for the terminal state (last square)

# Define the Q-table
Q = np.zeros((n_states, n_actions))

# Define the learning parameters
alpha = 0.5  # Learning rate
gamma = 0.9  # Discount factor
n_episodes = 1000

# Run the Q-Learning algorithm
for episode in range(n_episodes):
    state = np.random.randint(0, n_states-1)
    done = False

    while not done:
        # Choose an action based on the current state and Q-values
        action = np.argmax(Q[state, :] + np.random.randn(1, n_actions)/(episode+1))

        # Simulate the next state and get the reward
        next_state = state + action
        next_state = np.clip(next_state, 0, n_states-1)
        reward = rewards[next_state]

        # Update the Q-value for the current state and action
        Q[state, action] = (1 - alpha) * Q[state, action] + alpha * (reward + gamma * np.max(Q[next_state, :]))

        state = next_state
        done = (state == n_states-1)

# Print the learned Q-values
print("Learned Q-values:")
print(Q)
