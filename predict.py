# predict.py

import joblib

# Load model & vectorizer
model = joblib.load("spam_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

print("🔹 Spam SMS Classifier Ready! (type 'exit' to quit)\n")

while True:
    message = input("Enter a message: ")

    # Exit condition
    if message.lower() == "exit":
        print("👋 Exiting classifier.")
        break

    # Preprocess and predict
    message_tfidf = vectorizer.transform([message])
    prediction = model.predict(message_tfidf)

    if prediction[0] == 1:
        print("🚨 Spam Message\n")
    else:
        print("✅ Ham Message (Not Spam)\n")
