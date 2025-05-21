import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# ----------- Load and Prepare Data -----------
data = pd.read_csv("Nifty.csv", index_col=1)
data.index = pd.to_datetime(data.index, format="%d-%b-%y")

data['Return'] = data['Close'].pct_change()
data['Volatility'] = data['Return'].rolling(window=21).std()
data.dropna(inplace=True)

# ----------- Parameters -----------
initial_training = 60  # Wait for enough data to compute volatility
k = 4  # Number of clusters

squared_errors = []
direction_correct = 0
total_predictions = 0

predicted_returns = []
predicted_volatilities = []
actual_returns = []
actual_volatilities = []

# ----------- Backtest Loop -----------
for i in range(initial_training, len(data)):
    # Use all data up to current point
    train_data = data.iloc[:i][['Return', 'Volatility']]
    
    scaler = StandardScaler()
    scaled_train = scaler.fit_transform(train_data)

    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    try:
        kmeans.fit(scaled_train)
    except:
        continue

    # Predict next point's regime
    last_point = data.iloc[i][['Return', 'Volatility']].values.reshape(1, -1)
    last_scaled = scaler.transform(last_point)
    cluster = kmeans.predict(last_scaled)[0]
    predicted_scaled = kmeans.cluster_centers_[cluster]
    predicted = scaler.inverse_transform(predicted_scaled.reshape(1, -1))[0]

    predicted_return = predicted[0]
    predicted_volatility = predicted[1]
    actual_return = data['Return'].iloc[i]
    actual_volatility = data['Volatility'].iloc[i]

    predicted_returns.append(predicted_return)
    predicted_volatilities.append(predicted_volatility)
    actual_returns.append(actual_return)
    actual_volatilities.append(actual_volatility)

    # Error metrics
    error = (predicted_return - actual_return) ** 2 + (predicted_volatility - actual_volatility) ** 2
    squared_errors.append(error)

    if np.sign(predicted_return) == np.sign(actual_return):
        direction_correct += 1
    total_predictions += 1

# ----------- Metrics -----------
combined_mse = np.mean(squared_errors)
rmse = np.sqrt(combined_mse)

mean_return = np.mean(np.abs(data['Return']))
mean_volatility = np.mean(np.abs(data['Volatility']))

percent_error_return = (rmse / mean_return) * 100
percent_error_volatility = (rmse / mean_volatility) * 100
directional_accuracy = (direction_correct / total_predictions) * 100

print("=== Backtest Evaluation (Using All Past Data) ===")
print(f"Combined MSE (Return + Volatility): {combined_mse:.8f}")
print(f"Root MSE (RMSE): {rmse:.8f}")
print(f"Percent Error (Return): {percent_error_return:.2f}%")
print(f"Percent Error (Volatility): {percent_error_volatility:.2f}%")
print(f"Directional Accuracy: {directional_accuracy:.2f}%")

# ----------- Plotting Actual vs Predicted Returns -----------
plt.figure(figsize=(12, 6))
plt.plot(data.index[initial_training:], actual_returns, label='Actual Return', alpha=0.6)
plt.plot(data.index[initial_training:], predicted_returns, label='Predicted Return', alpha=0.6)
plt.title("Actual vs Predicted Returns (Cumulative KMeans Training)")
plt.xlabel("Date")
plt.ylabel("Return")
plt.legend()
plt.grid(True, linestyle="dotted", alpha=0.5)
plt.show()
