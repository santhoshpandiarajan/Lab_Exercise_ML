from hmmlearn import hmm
import numpy as np

# Define the number of hidden states and observation symbols
n_states = 3
n_observations = 2

# Create a GaussianHMM model
model = hmm.GaussianHMM(n_components=n_states, n_iter=100)

# Generate some example data
X = np.array([[0], [1], [0], [1], [0]])

# Reshape the data to match the expected format
X = X.reshape(-1, 1)

# Fit the model to the data using the Baum-Welch algorithm
model.fit(X)

# Get the estimated transition matrix, means, and covariances
transition_matrix = model.transmat_
means = model.means_
covariances = model.covars_

print("Transition Matrix:")
print(transition_matrix)
print("Means:")
print(means)
print("Covariances:")
print(covariances)
