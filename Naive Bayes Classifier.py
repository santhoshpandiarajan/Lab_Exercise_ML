from sklearn.naive_bayes import GaussianNB

# Create a Gaussian Naive Bayes classifier
classifier = GaussianNB()

# Generate some example data
X = [[1, 2], [2, 1], [3, 4], [4, 3]]
y = [0, 0, 1, 1]

# Fit the classifier to the data
classifier.fit(X, y)

# Predict the class labels for new samples
new_samples = [[1.5, 2.5], [3.5, 3.5]]
predicted_labels = classifier.predict(new_samples)

print("Predicted Labels:", predicted_labels)
