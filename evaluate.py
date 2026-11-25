#!/usr/bin/env python3
"""
Generate evaluation plots – clean version (no warnings)
"""
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support
import joblib
import warnings
warnings.filterwarnings('ignore')

# Load data
with open('gesture_data.pkl', 'rb') as f:
    X_raw, y_raw = pickle.load(f)

model = joblib.load('best_model.pkl')
scaler = joblib.load('scaler.pkl')
le = joblib.load('label_encoder.pkl')

X = np.array(X_raw)
y = le.transform(y_raw)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_test_scaled = scaler.transform(X_test)
y_pred = model.predict(X_test_scaled)

# 1. Accuracy bar chart (top 10 models) – FIX: use hue to avoid deprecation
df = pd.read_csv('model_ranking.csv')
plt.figure(figsize=(12,6))
# If the DataFrame has a 'Test' column (numeric)
if 'Test' in df.columns:
    # Use hue with legend=False to silence the FutureWarning
    sns.barplot(data=df.head(10), x='Model', y='Test', hue='Model', palette='viridis', legend=False)
    plt.xticks(rotation=45)
    plt.title('Top 10 Models by Test Accuracy')
    plt.tight_layout()
    plt.savefig('accuracy_chart.png')
    print("Saved accuracy_chart.png")
else:
    print("model_ranking.csv missing 'Test' column, skipping accuracy chart")

# 2. Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.title('Confusion Matrix – Best Model')
plt.savefig('confusion_matrix.png')
print("Saved confusion_matrix.png")

# 3. Classification report – handle zero_division warnings
# Compute per-class precision/recall/f1 explicitly with zero_division=0
precision, recall, fscore, _ = precision_recall_fscore_support(y_test, y_pred, average=None, zero_division=0)
classes = le.classes_
# Format as table
report_lines = ["Classification Report (zero_division=0):\n"]
for i, cls in enumerate(classes):
    report_lines.append(f"{cls:15s} precision={precision[i]:.3f} recall={recall[i]:.3f} f1={fscore[i]:.3f}")
# Also add macro averages
macro_prec = precision.mean()
macro_rec = recall.mean()
macro_f1 = fscore.mean()
report_lines.append(f"{'macro avg':15s} precision={macro_prec:.3f} recall={macro_rec:.3f} f1={macro_f1:.3f}")
with open('classification_report.txt', 'w') as f:
    f.write('\n'.join(report_lines))
print("Saved classification_report.txt (clean)")

# 4. Learning curve – skip if model is KNN (to avoid n_neighbors issues)
model_name = type(model).__name__
if model_name == 'KNeighborsClassifier':
    print("Learning curve skipped for KNN to avoid n_neighbors errors (KNN requires >5 samples per CV fold).")
else:
    try:
        from sklearn.model_selection import learning_curve
        train_sizes, train_scores, test_scores = learning_curve(
            model, X_test_scaled, y_test, cv=3,
            train_sizes=np.linspace(0.2, 1.0, 4),
            scoring='accuracy', n_jobs=1
        )
        train_mean = np.mean(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        plt.figure()
        plt.plot(train_sizes, train_mean, 'o-', label='Train')
        plt.plot(train_sizes, test_mean, 'o-', label='CV')
        plt.legend(); plt.grid(); plt.title('Learning Curve')
        plt.savefig('learning_curve.png')
        print("Saved learning_curve.png")
    except Exception as e:
        print(f"Learning curve failed: {e} (skipped)")

print("\n✅ Evaluation complete. Ready for demo.")
