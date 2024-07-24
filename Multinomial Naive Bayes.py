from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Load the 20 Newsgroups dataset
categories = ['alt.atheism', 'sci.space']
data_train = fetch_20newsgroups(subset='train', categories=categories)
data_test = fetch_20newsgroups(subset='test', categories=categories)

# Extract features from the text data using the bag-of-words model
vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(data_train.data)
X_test = vectorizer.transform(data_test.data)

# Create the Naive Bayes classifier
classifier = MultinomialNB()

# Train the classifier
classifier.fit(X_train, data_train.target)

# Make predictions on the test data
y_pred = classifier.predict(X_test)

# Compute accuracy, precision, and recall
accuracy = accuracy_score(data_test.target, y_pred)
precision = precision_score(data_test.target, y_pred)
recall = recall_score(data_test.target, y_pred)

# Print the evaluation metrics
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
