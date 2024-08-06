import streamlit as st
from joblib import load
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Load your logistic regression model and CountVectorizer
lr_loaded = load('logistic_regression_model.joblib')
cv_loaded = load('count_vectorizer.joblib')

# Streamlit application starts here
def main():
    # Title of your web app
    st.title("Spam/Ham Prediction App")

    # Sidebar for navigation
    st.sidebar.title("Options")
    option = st.sidebar.selectbox("Choose how to input data", ["Enter text", "Upload file"])

    if option == "Enter text":
        # Text box for user input
        user_input = st.text_input("Enter a sentence to check if it's spam or ham:")

        # Predict button
        if st.button('Predict'):
            if user_input:  # Check if the input is not empty
                predict_and_display([user_input])  # Single sentence prediction
            else:
                st.error("Please enter a sentence for prediction.")
    else:  # Option to upload file
        uploaded_file = st.file_uploader("Choose a file", type=['txt', 'csv'])
        if uploaded_file is not None:
            if uploaded_file.type == "text/csv" or uploaded_file.name.endswith('.csv'):
                data = pd.read_csv(uploaded_file)
            else:  # Assume text file
                data = pd.read_table(uploaded_file, header=None, names=['text'])

            # Check if the file has content
            if not data.empty:
                sentences = data['text'].tolist()
                predict_and_display(sentences)  # File-based prediction

def predict_and_display(sentences):
    # Transform the sentences
    transformed_sentences = cv_loaded.transform(sentences)

    # Make predictions
    results = lr_loaded.predict(transformed_sentences)

    # Combine the inputs and predictions into a DataFrame
    results_df = pd.DataFrame({
        'Input': sentences,
        'Prediction': results
    })

    # Tabulate and display the results
    with st.expander("Show/Hide Prediction Table"):
        st.table(results_df)

    # Display histogram of predictions
    st.write("Histogram of Predictions:")
    fig, ax = plt.subplots()
    prediction_counts = pd.Series(results).value_counts().sort_index()
    prediction_counts.plot(kind='bar', ax=ax)
    ax.set_title("Number of Spam and Ham Predictions")
    ax.set_xlabel("Category")
    ax.set_ylabel("Count")
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))  # Ensure y-axis has integer ticks
    st.pyplot(fig)
if __name__ == '__main__':
    main()