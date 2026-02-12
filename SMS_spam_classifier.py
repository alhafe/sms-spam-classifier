# ===============================
# SMS Spam Classifier Project
# Author: Hafe
# ===============================

import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns


# -------------------------------
# 1. Load Dataset
# -------------------------------
df = pd.read_table(
    "SMSSpamCollection.txt",
    sep="\t",
    header=None,
    names=["label", "sms_message"]
)

print("Dataset shape:", df.shape)
print(df.head())


# -------------------------------
# 2. Label Encoding
# -------------------------------
df["label"] = df["label"].map({"ham": 0, "spam": 1})


# -------------------------------
# 3. Text Cleaning Function
# -------------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"\d+", "", text)        # remove numbers
    text = re.sub(r"[^\w\s]", "", text)    # remove punctuation
    return text

df["sms_message"] = df["sms_message"].apply(clean_text)


# -------------------------------
# 4. Train-Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    df["sms_message"],
    df["label"],
    test_size=0.2,
    random_state=42
)

print(f"Training samples: {X_train.shape[0]}")
print(f"Testing samples: {X_test.shape[0]}")


# -------------------------------
# 5. Vectorization (Bag of Words)
# -------------------------------
vectorizer = CountVectorizer(stop_words="english")
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

print("Vocabulary size:", X_train_vec.shape[1])


# -------------------------------
# 6. Train Naive Bayes Model
# -------------------------------
model = MultinomialNB()
model.fit(X_train_vec, y_train)


# -------------------------------
# 7. Predictions
# -------------------------------
predictions = model.predict(X_test_vec)


# -------------------------------
# 8. Evaluation Metrics
# -------------------------------
accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions)
recall = recall_score(y_test, predictions)
f1 = f1_score(y_test, predictions)

print("\nModel Performance:")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")


# -------------------------------
# 9. Confusion Matrix
# -------------------------------
cm = confusion_matrix(y_test, predictions)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Ham", "Spam"])
disp.plot()
plt.title("Confusion Matrix")
plt.show()


# -------------------------------
# 10. Data Distribution Plot
# -------------------------------
sns.countplot(x=df["label"])
plt.title("Spam vs Ham Distribution")
plt.show()


# -------------------------------
# 11. Test on Custom Messages
# -------------------------------
test_messages = [
    "You won $1000! Send your bank details now",
    "Hello, are we meeting today?",
    "Congratulations! Claim your free prize now"
]

test_vec = vectorizer.transform(test_messages)
test_predictions = model.predict(test_vec)

for msg, pred in zip(test_messages, test_predictions):
    label = "SPAM" if pred == 1 else "HAM"
    print(f"\nMessage: {msg}")
    print("Prediction:", label)