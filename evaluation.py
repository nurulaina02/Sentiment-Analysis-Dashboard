from sklearn.metrics import classification_report, accuracy_score
import pandas as pd

def evaluate(true_labels, predicted_labels):
    print("Accuracy:", accuracy_score(true_labels, predicted_labels))
    print(classification_report(true_labels, predicted_labels))
