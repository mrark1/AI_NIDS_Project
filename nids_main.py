# =========================================
# AI-Based Network Intrusion Detection System
# Streamlit Dashboard
# =========================================

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------------------
# PAGE CONFIG
# -----------------------------------------
st.set_page_config(
    page_title="AI-NIDS Dashboard",
    layout="wide"
)

st.title("🛡️ AI-Based Network Intrusion Detection System")
st.markdown("""
This dashboard uses **Machine Learning (Random Forest Algorithm)** to detect  
**Malicious Network Traffic** in real time.
""")

# -----------------------------------------
# DATA GENERATION (SIMULATED)
# -----------------------------------------
@st.cache_data
def load_data():
    np.random.seed(42)
    samples = 5000

    data = {
        "Destination_Port": np.random.randint(1, 65535, samples),
        "Flow_Duration": np.random.randint(1, 100000, samples),
        "Total_Fwd_Packets": np.random.randint(1, 200, samples),
        "Packet_Length_Mean": np.random.uniform(10, 1500, samples),
        "Active_Mean": np.random.uniform(1, 1000, samples),
        "Label": np.random.choice([0, 1], samples, p=[0.7, 0.3])
    }

    df = pd.DataFrame(data)

    # Introduce attack patterns
    df.loc[df["Label"] == 1, "Total_Fwd_Packets"] += np.random.randint(50, 300)
    df.loc[df["Label"] == 1, "Flow_Duration"] = np.random.randint(1, 2000)

    return df


df = load_data()

# -----------------------------------------
# SIDEBAR CONTROLS
# -----------------------------------------
st.sidebar.header("⚙️ Model Controls")

train_size = st.sidebar.slider(
    "Training Data (%)", 60, 90, 80
)

n_estimators = st.sidebar.slider(
    "Number of Trees", 50, 300, 100
)

# -----------------------------------------
# DATA SPLIT
# -----------------------------------------
X = df.drop("Label", axis=1)
y = df["Label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=(100 - train_size) / 100,
    random_state=42
)

# -----------------------------------------
# MODEL TRAINING
# -----------------------------------------
st.divider()
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("🔁 Model Training")

    if st.button("Train Model Now"):
        with st.spinner("Training the model..."):
            model = RandomForestClassifier(
                n_estimators=n_estimators,
                random_state=42,
                n_jobs=-1
            )
            model.fit(X_train, y_train)
            st.session_state["model"] = model
            st.success("✅ Model Trained Successfully!")

if "model" in st.session_state:
    model = st.session_state["model"]

# -----------------------------------------
# MODEL PERFORMANCE
# -----------------------------------------
with col2:
    st.subheader("📊 Performance Metrics")

    if "model" in st.session_state:
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        m1, m2, m3 = st.columns(3)
        m1.metric("Accuracy", f"{acc*100:.2f}%")
        m2.metric("Total Records", len(df))
        m3.metric("Threats Detected", np.sum(y_pred))

        st.markdown("### Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)

        fig, ax = plt.subplots(figsize=(4, 3))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Reds", ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)
    else:
        st.warning("⚠️ Please train the model first.")

# -----------------------------------------
# LIVE TRAFFIC SIMULATOR
# -----------------------------------------
st.divider()
st.subheader("🚦 Live Traffic Simulator")

st.markdown("Enter packet details to test the AI model:")

c1, c2, c3, c4, c5 = st.columns(5)

port = c1.number_input("Destination Port", 1, 65535, 80)
duration = c2.number_input("Flow Duration", 1, 100000, 500)
packets = c3.number_input("Total Packets", 1, 500, 120)
pkt_len = c4.number_input("Packet Length Mean", 1, 1500, 600)
active = c5.number_input("Active Mean", 1, 1000, 50)

if st.button("Analyze Traffic"):
    if "model" in st.session_state:
        input_data = np.array(
            [[port, duration, packets, pkt_len, active]]
        )
        prediction = model.predict(input_data)

        if prediction[0] == 1:
            st.error("🚨 MALICIOUS TRAFFIC DETECTED!")
            st.write("High packet rate & abnormal duration indicate a possible attack.")
        else:
            st.success("✅ BENIGN TRAFFIC (Safe)")
    else:
        st.error("Please train the model before testing.")

# -----------------------------------------
# FOOTER
# -----------------------------------------
st.markdown("---")
st.markdown(
    "Developed by **Arpit Raj Katiyar** | Internship Final Project"
)
