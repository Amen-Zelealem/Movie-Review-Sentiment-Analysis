# 🎬 Movie Review Sentiment Analysis (Mini Project)

A simple yet effective sentiment analysis system trained on IMDb movie reviews using classic NLP techniques in Python. Built as part of a job application project to demonstrate a complete machine learning pipeline from preprocessing to deployment.

---

## 🔍 Project Overview

- **Task**: Binary sentiment classification (positive or negative).
- **Model**: Logistic Regression trained on TF-IDF vectors.
- **Tools**: Python, Scikit-learn, Flask (optional).
- **Data**: IMDb review samples (e.g., 5,000 entries).

---

## ✨ Features

✅ Train a sentiment classifier using TF-IDF + Logistic Regression  
✅ Predict sentiment from the command line with confidence scores  
✅ Bonus: Flask web app for interactive predictions

---

## 🚀 Getting Started

### 1. Clone the Repository

```
git clone https://github.com/Amen-Zelealem/Movie-Review-Sentiment-Analysis.git
cd Movie-Review-Sentiment-Analysis
```

### 2. Create and Activate Virtual Environment (Optional)
```
python -m venv venv
# On Linux/macOS:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

### 3. Install Dependencies
```
pip install -r requirements.txt
```

### Dataset
The project expects a dataset named:
`imdb_reviews_sample.csv`

This CSV file should be placed in the project root and must include at least the following columns:
- review: The movie review text
- sentiment: Either positive or negative

### Training the Model

To train the sentiment classifier and save the model to disk:
```
python train.py
```

This will:
- Load and preprocess the dataset
- Train a TF-IDF + Logistic Regression pipeline
- Save the model as model.joblib

### Making Predictions (Command-Line)
Predict the sentiment of any movie review using:
```
python predict.py "This movie was absolutely amazing!"
```
Example Output:
```
Prediction: positive (Confidence: 88.97%)
```


### Web App (Bonus)
An optional Flask-based UI is available to test predictions via the browser.
Running the App
```
cd app
python app.py
```
Then visit: http://127.0.0.1:5000 or 

**You Can Visit the Deployed Version** using https://movie-review-sentiment-analysis-g0du.onrender.com/

### Requirements
- Python 3.7+
- scikit-learn
- Flask (for web api)
- pandas, joblib

Install everything via:
```
pip install -r requirements.txt
```

### Project Structure
```
Movie-Review-Sentiment-Analysis/
├── .vscode/
│   └── settings.json
├── app/
│   └── templates
│       └── index.html          # Index page
│       └── result.html         # Result Page
│   └── app.py                  # Flask UI
├── scripts/
│   └── load_data.py            # Script to load data from Kaggle
├── tests/
│   └── test_load_data.py       # Unit test for load_data
├── predict.py                  # Command-line prediction
├── train.py                    # Model training script
├── imdb_reviews_sample.csv
├── Procfile
├── requirements.txt
└── README.md
```

## **Sample Screenshots**
![SentimentForm](/screenshots/SentimentForm.png)

---

![Result](/screenshots/Result.png)


### Acknowledgements
- IMDb dataset (for demonstration purposes)
- scikit-learn for fast and easy ML
- Flask for lightweight web interface

### Contact
[Amen-Zelealem](mailto:amenzelealem@gmail.com)
