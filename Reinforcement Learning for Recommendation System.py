import numpy as np

# Initialize the recommendation system
num_users = 5
num_items = 10
q_table = np.zeros((num_users, num_items))

# Define the reward matrix
rewards = np.array([
    [5, 2, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 4, 0, 3, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 2, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 3, 4],
    [0, 0, 5, 0, 0, 0, 0, 0, 0, 0]
])

# Define the learning parameters
num_episodes = 100
alpha = 0.8
gamma = 0.95
epsilon = 0.2

# Reinforcement Learning algorithm
for episode in range(num_episodes):
    state = np.random.randint(num_users)  # Initial state is a random user
    done = False

    while not done:
        if np.random.uniform() < epsilon:
            action = np.random.randint(num_items)  # Explore by choosing a random item
        else:
            action = np.argmax(q_table[state])  # Exploit by choosing the item with maximum Q-value

        next_state = np.random.randint(num_users)  # Transition to a random next state (user)

        # Update Q-table
        q_table[state, action] += alpha * (
                rewards[state, action] + gamma * np.max(q_table[next_state]) - q_table[state, action])

        state = next_state

        # Check if episode is finished
        if np.random.uniform() < 0.1:
            done = True

# Print the learned Q-table
print("Learned Q-table:")
print(q_table)
