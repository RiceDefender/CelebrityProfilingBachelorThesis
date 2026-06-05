import numpy as np
import joblib
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

# ==========================================
# 1. LOAD AND PREPARE THE DATA
# ==========================================
print("Loading data arrays...")
features = np.load('./Testing/celeb_files/features.npy')
labels = np.load('./Testing/celeb_files/labels.npy')

X = features[:, 1:] # Drop celeb_id

y_by = np.array([int(y[0]) for y in labels])         # Birth Year
y_gender = np.array([int(y[1]) for y in labels])     # Gender
y_occupation = np.array([int(y[2]) for y in labels]) # Occupation

X_train, X_test, y_train_occ, y_test_occ, y_train_gen, y_test_gen, y_train_by, y_test_by = train_test_split(
    X, y_occupation, y_gender, y_by, test_size=0.2, random_state=42
)

# Scale the features AND SAVE THE SCALER
print("Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create directory for saving models if it doesn't exist
os.makedirs("trained_models", exist_ok=True)
joblib.dump(scaler, "trained_models/bert_scaler.joblib")
print("Saved scaler to trained_models/bert_scaler.joblib")

# ==========================================
# 2. CUSTOM METRICS (Omitted for brevity, keep your original functions here)
# ==========================================

# ==========================================
# 3. TRAIN MODELS & SAVE THEM
# ==========================================
print("\nTraining Logistic Regression Models...")
lr_params = {'max_iter': 1000, 'random_state': 42}

# --- OCCUPATION ---
print("\n--- OCCUPATION PREDICTION ---")
model_occ = LogisticRegression(**lr_params)
model_occ.fit(X_train_scaled, y_train_occ)
joblib.dump(model_occ, "trained_models/occupation_model.joblib")
print("Saved occupation_model.joblib")

# --- GENDER ---
print("\n--- GENDER PREDICTION ---")
model_gen = LogisticRegression(**lr_params)
model_gen.fit(X_train_scaled, y_train_gen)
joblib.dump(model_gen, "trained_models/gender_model.joblib")
print("Saved gender_model.joblib")

# --- BIRTH YEAR ---
print("\n--- BIRTH YEAR PREDICTION ---")
model_by = LogisticRegression(**lr_params)
model_by.fit(X_train_scaled, y_train_by)
joblib.dump(model_by, "trained_models/birth_year_model.joblib")
print("Saved birth_year_model.joblib")