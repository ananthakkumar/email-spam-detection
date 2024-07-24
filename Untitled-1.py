"""
import pandas as pd
from flask import Flask, request, render_template
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from nltk.corpus import stopwords
import string
import joblib
import os

# Define file paths
DATASET_PATH = 'C:/Users/anant_l1e8kp/OneDrive/Documents/jesper apps/email spam/spamham.csv'
MODEL_PATH = 'spam_classifier.pkl'
FEEDBACK_PATH = 'C:/Users/anant_l1e8kp/OneDrive/Documents/jesper apps/email spam/feedback.csv'

# Load and preprocess the dataset
emails = pd.read_csv(DATASET_PATH, encoding='ISO-8859-1')

def preprocess_text(text):
    """Preprocess the input text by converting to lowercase, removing punctuation, and filtering stopwords."""
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    tokens = text.split()
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    return ' '.join(tokens)

emails['email'] = emails['email'].apply(preprocess_text)
X_train, X_test, y_train, y_test = train_test_split(emails['email'], emails['label'], test_size=0.2, random_state=42)

pipeline = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('classifier', MultinomialNB())
])
pipeline.fit(X_train, y_train)

# Save initial model
joblib.dump(pipeline, MODEL_PATH)

# Create Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Predict whether the input email text is spam or ham."""
    try:
        if request.method == 'POST':
            email_text = request.form['email_text']
            processed_email_text = preprocess_text(email_text)
            prediction = pipeline.predict([processed_email_text])
            result = 'Spam' if prediction[0] == 'spam' else 'Ham'
            return render_template('index.html', prediction_text=f'The email is classified as: {result}', 
                                   email_text=email_text, prediction=prediction[0])
    except Exception as e:
        print(f"Error in predict route: {str(e)}")
        return render_template('error.html', error_message=str(e))

@app.route('/feedback', methods=['POST'])
def feedback():
    """Handle user feedback for email classification and retrain the model."""
    try:
        if request.method == 'POST':
            feedback = request.form['feedback']
            reason = request.form['reason']
            email_text = request.form['email_text']
            prediction = request.form['prediction']
            
            # Save feedback to a file or database with UTF-8 encoding
            with open(FEEDBACK_PATH, 'a', encoding='utf-8', errors='ignore') as f:
                f.write(f'"{email_text}","{prediction}","{feedback}","{reason}"\n')
            
            # Retrain model with feedback
            integrate_feedback_and_retrain()
            
            return render_template('index.html', prediction_text='Thank you for your feedback!')
    except Exception as e:
        print(f"Error in feedback route: {str(e)}")
        return render_template('error.html', error_message=str(e))

def integrate_feedback_and_retrain():
    """Integrate user feedback into the dataset and retrain the model."""
    global pipeline, emails
    
    try:
        # Load feedback data, skip lines with errors
        feedback_df = pd.read_csv(FEEDBACK_PATH, encoding='utf-8', error_bad_lines=False)
    except Exception as e:
        print(f"Error loading feedback.csv: {str(e)}")
        return
    
    # Filter incorrect feedback
    incorrect_feedback = feedback_df[feedback_df['feedback'] == 'no'][['email_text']]
    
    # Rename columns for merging
    incorrect_feedback.columns = ['email']
    incorrect_feedback['label'] = 'ham'  # Assuming the user provides the correct label in the 'reason' column
    
    # Append corrected emails to the original dataset
    emails = pd.concat([emails, incorrect_feedback], ignore_index=True)
    
    # Save the updated dataset to disk
    emails.to_csv(DATASET_PATH, index=False, encoding='ISO-8859-1')

    # Preprocess the text again
    emails['email'] = emails['email'].apply(preprocess_text)
    
    # Split the updated dataset
    X_train, X_test, y_train, y_test = train_test_split(emails['email'], emails['label'], test_size=0.2, random_state=42)
    
    # Retrain the pipeline
    pipeline.fit(X_train, y_train)
    
    # Save the updated model to disk
    joblib.dump(pipeline, MODEL_PATH)
    
    print("Model retrained and saved.")

if __name__ == '__main__':
    # Load the initial model
    pipeline = joblib.load(MODEL_PATH)
    app.run(debug=True)
"""