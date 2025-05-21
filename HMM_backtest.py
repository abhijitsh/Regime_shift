import pandas as pd
import numpy as np
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler

# Load and preprocess data
data = pd.read_csv("Nifty.csv", index_col=1)
data['Return'] = data['Close'].pct_change()
data['Volatility'] = data['Return'].rolling(window=21).std()
data.dropna(inplace=True)

lookback = 60  # minimum required length before starting
squared_errors = []

for i in range(lookback, len(data)):
    train_data = data.iloc[:i][['Return', 'Volatility']]  # growing window

    scaler = StandardScaler()
    scaled_train = scaler.fit_transform(train_data)

    model = hmm.GaussianHMM(n_components=3, covariance_type="full", n_iter=1000, random_state=42)
    try:
        model.fit(scaled_train)
    except:
        continue

    last_scaled = scaled_train[-1].reshape(1, -1)
    last_state = model.predict(last_scaled)[0]
    state_means = model.means_[last_state]
    predicted = scaler.inverse_transform(state_means.reshape(1, -1))[0]

    predicted_return = predicted[0]
    predicted_volatility = predicted[1]

    actual_return = data['Return'].iloc[i]
    actual_volatility = data['Volatility'].iloc[i]

    error = (predicted_return - actual_return) ** 2 + (predicted_volatility - actual_volatility) ** 2
    squared_errors.append(error)

# Calculate combined MSE
combined_mse = np.mean(squared_errors)
rmse = np.sqrt(combined_mse)

# Calculate mean absolute values for scaling
mean_return = np.mean(np.abs(data['Return']))
mean_volatility = np.mean(np.abs(data['Volatility']))

# Approximate percentage error relative to mean magnitudes
percent_error_return = (rmse / mean_return) * 100
percent_error_volatility = (rmse / mean_volatility) * 100

print(f"Combined MSE (Return + Volatility): {combined_mse:.8f}")
print(f"Root MSE (RMSE): {rmse:.8f}")
print(f"Approximate % error wrt mean return: {percent_error_return:.2f}%")
print(f"Approximate % error wrt mean volatility: {percent_error_volatility:.2f}%")
