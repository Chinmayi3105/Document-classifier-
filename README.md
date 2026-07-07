# Document Classifier

A machine learning-based document classification system that classifies text documents into predefined categories using Natural Language Processing and Machine Learning techniques.

## Features

- Classifies documents based on their content
- Performs text preprocessing and vectorization
- Uses a trained machine learning model for prediction
- Provides an interactive interface using Streamlit
- Uses saved model files for quick classification

## Technologies Used

- Python
- Scikit-learn
- Pandas
- Joblib
- Streamlit

## Project Structure

Document Classifier

├── app.py              - Streamlit application  
├── train.py            - Model training script  
├── model.pkl           - Trained classification model  
├── vectorizer.pkl      - Text vectorizer  
├── requirements.txt    - Required Python libraries  
└── README.md           - Project documentation  

## How to Run the Project

Install the required dependencies:

pip install -r requirements.txt

Run the Streamlit application:

python -m streamlit run app.py

After running the command, the application will open in your browser. Enter document text and get the predicted category.

## About the Project

This project uses Natural Language Processing and Machine Learning techniques to automatically classify documents. The trained model is integrated with a Streamlit application to provide an easy-to-use document classification system.