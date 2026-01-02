# ==============================
# AI-Based Network Intrusion Detection System
# Machine Learning Model
# ==============================

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ------------------------------
# 1. DATA GENERATION (Simulation)
# ------------------------------
def generate_data(samples=5000):
    np.random.seed(42)

    data = {
        "Destination_Port": np.random.randint(1, 65535, samples),
        "Flow_Duration": np.random.randint(1, 100000, samples),
        "Total_Fwd_Packets": np.random.randint(1, 200, samples),
        "Packet_Length_Mean": np.random.uniform(10, 1500, samples),
        "Active_Mean": np.random.uniform(1, 1000, samples),
        "Label": np.random.choice([0, 1], samples, p=[0.7, 0.3])  
        # 0 = Benign, 1 = Malicious
    }

    df = pd.DataFrame(data)

    # Attack behavior pattern
    df.loc[df["Label"] == 1, "Total_Fwd_Packets"] += np.random.randint(50, 300)
    df.loc[df["Label"] == 1, "Flow_Duration"] = np.random.randint(1, 2000)

    return df


# ------------------------------
# 2. LOAD DATA
# ------------------------------
df = generate_data()

X = df.drop("Label", axis=1)
y = df["Label"]

# ------------------------------
# 3. TRAIN-TEST SPLIT
# ------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ------------------------------
# 4. MODEL CREATION
# ------------------------------
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

# ------------------------------
# 5. TRAIN MODEL
# ------------------------------
model.fit(X_train, y_train)

# ------------------------------
# 6. EVALUATION
# ------------------------------
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("\nModel Accuracy:", accuracy * 100, "%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# ------------------------------
# 7. PREDICTION FUNCTION
# ------------------------------
def predict_intrusion(packet_data):
    """
    packet_data format:
    [Destination_Port, Flow_Duration, Total_Fwd_Packets, Packet_Length_Mean, Active_Mean]
    """
    packet_data = np.array(packet_data).reshape(1, -1)
    result = model.predict(packet_data)

    if result[0] == 1:
        return "⚠ MALICIOUS TRAFFIC DETECTED"
    else:
        return "✅ BENIGN TRAFFIC"


# ------------------------------
# 8. TEST MODEL MANUALLY
# ------------------------------
sample_packet = [80, 500, 300, 1200, 50]
print("\nSample Prediction:", predict_intrusion(sample_packet))
