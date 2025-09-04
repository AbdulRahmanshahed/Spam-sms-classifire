# 📩 Spam SMS Classifier (NLP with Scikit-learn)

This project is part of my AI/ML internship (Week 2). It builds a **Spam SMS Classifier** using **Python, Scikit-learn, and NLP techniques**. The dataset is the UCI SMS Spam Collection (5,574 SMS messages labeled as ham or spam). The project flow is: load and clean dataset → preprocess text (lowercasing, stopword removal, TF-IDF) → train Naive Bayes classifier → evaluate with accuracy, precision, recall, F1 score → save model and vectorizer with Joblib → test messages interactively via `predict.py`. 

### 🔹 Tech Stack
Python · pandas · numpy · scikit-learn · matplotlib · seaborn · joblib  

### 🔹 Files
- `spam_classifier.py` → training code  
- `predict.py` → script to test SMS messages  
- `spam_model.pkl` → trained model  
- `vectorizer.pkl` → TF-IDF vectorizer  

### 🔹 How to Run
```bash
pip install pandas numpy scikit-learn matplotlib seaborn joblib
python spam_classifier.py     # trains and saves model
python predict.py             # test messages interactively

<img width="1200" height="683" alt="image" src="https://github.com/user-attachments/assets/5e172542-64f7-48ec-a03c-9eb1b5993d41" />

Results: Achieved ~97% accuracy with confusion matrix and classification report generated during training. Open-source and free to use for educational purposes.
