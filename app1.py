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

# Check the structure of your emails DataFrame
emails = pd.read_csv(DATASET_PATH, encoding='ISO-8859-1')
print(emails.columns)  # Print columns to verify structure

# Ensure 'email' column exists before applying preprocessing
if 'email' in emails.columns:
    emails['email'] = emails['email'].apply(preprocess_text)
else:
    print("Column 'email' not found in DataFrame.")

# Further processing and model training...
