
# Cyberbullying Detection using Machine Learning

## Overview

This project focuses on detecting various types of cyberbullying on the internet using machine learning and deep learning models. The objective is to classify harmful content into predefined categories and enable real-time detection through a backend Flask API and Chrome Extension integration.

## Classes Detected

- Cyberstalking
- Doxing
- Sexual Harassment
- Revenge Porn
- Slut Shaming
- Non-Cyberbullying

## Dataset

- **Source**: IEEE Dataport
- **Augmentation**: Back Translation was used to improve model generalization.
- **Class Extension**: Added a 6th class - "Non-Cyberbullying" by combining IEEE dataset samples and synthetic examples generated via GPT.

## Models Used

- Naive Bayes
- Support Vector Machine (SVM)
- Logistic Regression
- Bi-LSTM (Optimized using Adam optimizer)
- Ensemble Learning (Hard Voting and Soft Voting)

## Evaluation Metrics

- Confusion Matrix
- Precision-Recall Curve
- Accuracy, Precision, Recall, F1-Score

## Backend API (Flask)

- **/predict** endpoint accepts an array of texts and returns the predicted class, confidence score, and a flag indicating if it's cyberbullying.
- Model is saved as `model.h5` and loaded using `model_utils/predictor.py`.

## How to Run the Backend

1. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

2. Run the Flask server:

    ```bash
    python app.py
    ```

3. Open in browser:

    ```
    http://127.0.0.1:5000/
    ```

## Folder Structure

```
📁 root/
│
├── model.h5                  # Trained ensemble model
├── app.py                    # Flask backend API
├── model_utils/
│   └── predictor.py          # Functions to load model/tokenizer and make predictions
├── requirements.txt
└── README.md
```

## Future Work

- Improve multilingual support
- Handle noisy/abusive data more effectively
- Real-time monitoring and reporting system
