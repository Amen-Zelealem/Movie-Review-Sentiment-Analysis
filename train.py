import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

df = pd.read_csv("imdb_reviews_sample.csv")  

# Encode sentiment as 0/1
df['label'] = df['sentiment'].map({'negative': 0, 'positive': 1})

# Split data
X_train, X_test, y_train, y_test = train_test_split(df['review'], df['label'], test_size=0.2, random_state=42)

# Build pipeline
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', LogisticRegression(max_iter=1000))
])

# Train
pipeline.fit(X_train, y_train)

# Save model
joblib.dump(pipeline, 'model.joblib')
print("Model trained and saved as 'model.joblib'")