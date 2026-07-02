import pandas as pd
import re

# Load dataset
data = pd.read_csv("dataset/labeled_data.csv")

# Keep only tweet and class columns
data = data[['tweet','class']]

# Function to clean tweets
def clean_text(text):
    text = re.sub(r"http\S+", "", text)   # remove links
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # remove special characters
    text = text.lower()   # convert to lowercase
    return text

# Apply cleaning
data['tweet'] = data['tweet'].apply(clean_text)

print("Cleaned Tweets:")
print(data.head())

from sklearn.feature_extraction.text import TfidfVectorizer

# Convert text to TF-IDF features
vectorizer = TfidfVectorizer(stop_words='english')

X = vectorizer.fit_transform(data['tweet'])
y = data['class']

print("Feature shape:", X.shape)

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)
# Evaluate model
print("\nModel Accuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Test with a custom tweet
test_tweet = ["I hate you so much"]

# Clean the tweet
test_tweet_clean = [clean_text(t) for t in test_tweet]

# Convert to TF-IDF
test_vector = vectorizer.transform(test_tweet_clean)

# Predict
prediction = model.predict(test_vector)

print("\nPrediction:", prediction)

if prediction[0] == 0:
    print("Hate Speech")
elif prediction[0] == 1:
    print("Offensive Language")
else:
    print("Neither")
    
import pickle

# Save trained model
pickle.dump(model, open("model/hate_model.pkl", "wb"))

# Save vectorizer
pickle.dump(vectorizer, open("model/vectorizer.pkl", "wb"))

print("Model saved successfully!")

