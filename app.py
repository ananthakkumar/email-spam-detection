import pandas as pd
from flask import Flask, request, render_template
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from nltk.corpus import stopwords
import string
import joblib
import logging
import os

# Define file paths
DATASET_PATH = 'C:/Users/anant_l1e8kp/OneDrive/Documents/jesper apps/email spam/spamham.csv'
MODEL_PATH = 'spam_classifier.pkl'
FEEDBACK_PATH = 'C:/Users/anant_l1e8kp/OneDrive/Documents/jesper apps/email spam/feedback.csv'

# Set up logging
logging.basicConfig(level=logging.DEBUG, filename='app.log', filemode='a', 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Load and preprocess the dataset
emails = pd.read_csv(DATASET_PATH, encoding='ISO-8859-1')

# Print the column names to check for any issues
print(emails.columns)

# Rename the column to a more manageable name
emails.rename(columns={'ï»¿email': 'email'}, inplace=True)

def preprocess_text(text):
    """Preprocess the input text by converting to lowercase, removing punctuation, and filtering stopwords."""
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    tokens = text.split()
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    return ' '.join(tokens)

# Drop rows with missing values in the email column
emails.dropna(subset=['email'], inplace=True)

# Ensure all values in the email column are strings
emails['email'] = emails['email'].astype(str)

# Preprocess the email column
emails['email'] = emails['email'].apply(preprocess_text)

# Split dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(emails['email'], emails['label'], test_size=0.2, random_state=42)

# Create a pipeline
pipeline = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('classifier', MultinomialNB())
])

# Fit the pipeline on training data
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
        logging.error(f"Error in predict route: {str(e)}")
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
            
            # Integrate feedback into the dataset and retrain the model
            integrate_feedback_and_retrain(email_text, prediction, feedback, reason)
            
            return render_template('index.html', prediction_text='Thank you for your feedback!')
    except Exception as e:
        logging.error(f"Error in feedback route: {str(e)}")
        return render_template('error.html', error_message=str(e))

def integrate_feedback_and_retrain(email_text, prediction, feedback, reason):
    """Integrate user feedback into the dataset and retrain the model."""
    global pipeline, emails

    try:
        # Append feedback to the dataset
        new_data = {'email': [email_text], 'label': [reason]}
        new_emails = pd.DataFrame(new_data)
        
        # Preprocess new email text
        new_emails['email'] = new_emails['email'].apply(preprocess_text)
        
        # Append corrected emails to the original dataset
        emails = pd.concat([emails, new_emails], ignore_index=True)
        
        # Debug: Print the size of the dataset before and after appending feedback
        logging.debug(f"Size of original dataset: {len(emails) - len(new_emails)}")
        logging.debug(f"Size of dataset after adding feedback: {len(emails)}")
        
        # Save the updated dataset to disk with UTF-8 encoding
        emails.to_csv(DATASET_PATH, index=False, encoding='utf-8-sig')
        
        # Preprocess the text again
        emails['email'] = emails['email'].apply(preprocess_text)
        
        # Split the updated dataset
        X_train, X_test, y_train, y_test = train_test_split(emails['email'], emails['label'], test_size=0.2, random_state=42)
        
        # Retrain the pipeline
        pipeline.fit(X_train, y_train)
        
        # Save the updated model to disk
        #joblib.dump(pipeline, MODEL_PATH)
        
       # logging.debug("Model retrained and saved.")
        
    except Exception as e:
        logging.error(f"Error integrating feedback and retraining model: {str(e)}")

if __name__ == '__main__':
    # Load the initial model
    pipeline = joblib.load(MODEL_PATH)
    app.run(debug=True)
