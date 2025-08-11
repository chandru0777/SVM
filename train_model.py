import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib

# 1) Load CSV
df = pd.read_csv("plant_features_50.csv")

# 2) Features and target
X = df[['mean_R','mean_G','mean_B','std_gray','edge_density','hue_mean','area_ratio']].values
y_raw = df['label'].values

# 3) Encode labels
le = LabelEncoder()
y = le.fit_transform(y_raw)  # e.g. Healthy->0, Early_Blight->1, Late_Blight->2

# 4) Train / Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 5) Scale features (important for SVM)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# 6) Train SVM (with probability estimates)
model = SVC(kernel='rbf', C=5.0, gamma='scale', probability=True, random_state=42)
model.fit(X_train_s, y_train)

# 7) Evaluate
y_pred = model.predict(X_test_s)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred, target_names=le.classes_))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# 8) Save model, scaler, label encoder
joblib.dump(model, "svm_plant_model.pkl")
joblib.dump(scaler, "svm_scaler.pkl")
joblib.dump(le, "label_encoder.pkl")
print("Saved: svm_plant_model.pkl, svm_scaler.pkl, label_encoder.pkl")
