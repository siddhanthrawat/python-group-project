import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor

st.title("\U0001F9A0 SwasthyaNet AI - Disease Outbreak Predictor")
st.subheader("AI-driven early warning system for infectious disease surveillance")

st.sidebar.header("Simulate or Upload Data")
data_mode = st.sidebar.radio("Choose Input Mode:", ("Simulate Synthetic Data", "Upload CSV"))

if data_mode == "Simulate Synthetic Data":
    days = np.arange(30)
    fever_cases = np.random.poisson(lam=50, size=30)
    rash_cases = np.random.poisson(lam=5, size=30)
    platelet_alerts = np.random.poisson(lam=2, size=30)

    data = pd.DataFrame({
        'day': days,
        'fever_cases': fever_cases,
        'rash_cases': rash_cases,
        'platelet_alerts': platelet_alerts
    })

else:
    uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
    else:
        st.warning("Please upload a CSV file with columns: 'day', 'fever_cases', 'rash_cases', 'platelet_alerts'")
        st.stop()

st.write("### Clinical Data Preview:")
st.dataframe(data.head())

# Normalize features
scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(data[['fever_cases', 'rash_cases', 'platelet_alerts']])

# Create features and labels using sliding window
X, y = [], []
window_size = 5
for i in range(len(scaled_features) - window_size):
    X.append(scaled_features[i:i+window_size].flatten())
    y.append(scaled_features[i+window_size][0])  # Predict fever_cases
X, y = np.array(X), np.array(y)

# Train Random Forest model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# Predict next fever case
last_sequence = scaled_features[-window_size:].flatten().reshape(1, -1)
predicted_scaled = model.predict(last_sequence)[0]
predicted_combined = np.array([[predicted_scaled, 0, 0]])  # Pad zeros to match scaler input
predicted_fever = scaler.inverse_transform(predicted_combined)[0][0]

st.success(f"Predicted Fever Cases for Day {int(data['day'].max()) + 1}: {predicted_fever:.2f}")

# Alert logic
if predicted_fever > data['fever_cases'].iloc[-1] * 1.2:
    st.error("\U0001F6A8 ALERT: Potential Outbreak Predicted!")
else:
    st.info("\U0001F44D No major outbreak trend detected.")

# Plot
st.write("### Fever Cases vs Predicted")
fig, ax = plt.subplots()
ax.plot(data['day'], data['fever_cases'], label='Actual Fever Cases', marker='o')
ax.plot([data['day'].iloc[-1] + 1], [predicted_fever],
        label='Predicted Next Day', marker='X', markersize=10, color='red')
ax.set_xlabel("Day")
ax.set_ylabel("Fever Cases")
ax.legend()
ax.grid(True)
st.pyplot(fig)

st.caption("\U0001F4A1 SwasthyaNet AI - Smart Surveillance for a Healthier Future")
