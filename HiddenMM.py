# Install required libraries (run this in your terminal or notebook)
# pip install yfinance hmmlearn numpy pandas matplotlib scikit-learn 

import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler
import matplotlib.dates as mdates
import matplotlib.patches as mpatches

##### ---->  Download historical data for NIFTY 50 Using Yfinance, but not always used due to Limited trials  <----

# data = yf.download("^NSEI", start="2022-01-01", end="2023-01-01", auto_adjust = False)
# print(data)

#### -----> Data taken from https://niftyindices.com/reports/historical-data  <----
data = pd.read_csv("Nifty.csv", index_col=1)
##### Arrange data in increasing order of date-time

# Calculate returns and volatility
data['Return'] = data['Close'].pct_change()
data['Volatility'] = data['Return'].rolling(window=21).std()  # 1 month effective rolling volatility
# print("volatility is", data['Volatility'].head(30))
data = data.dropna()  # Drop missing values
# print(data)

# Prepare features for HMM (returns and volatility)
features = data[['Return', 'Volatility']]
# print(features)

# Standardize features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)


############## SELECTION OF MODEL BASED ON THE LOWER AIC/BIC

min_aic = float('inf')
min_bic = float('inf')
model = None
comp =3

for n in range(3,5):
    m = hmm.GaussianHMM(n_components=n, covariance_type="full", n_iter=1000, random_state=42)
    m.fit(scaled_features)
    
    log_likelihood = m.score(scaled_features)
    num_params = n**2 + 2 * n * scaled_features.shape[1] + n  # Corrected formula
    num_samples = len(scaled_features)

    aic = 2 * num_params - 2 * log_likelihood
    bic = num_params * np.log(num_samples) - 2 * log_likelihood

    if(aic <= min_aic and bic <= min_bic):
        comp =n
        model = m


# Train the Hidden Markov Model
# model = hmm.GaussianHMM(n_components=5, covariance_type="full", n_iter=1000, random_state=42)
# model.fit(scaled_features)

state_means = pd.DataFrame(model.means_, columns=['Return Mean', 'Volatility Mean'])
# print("The stats for each states (in Z-form) are:\n",state_means)

# Get mean and standard deviation from StandardScaler (Z form, So volatility is negative)
original_means = scaler.mean_  # [Return Mean, Volatility Mean]
original_stds = scaler.scale_  # [Return Std, Volatility Std]

# Convert back to original values
real_means = (state_means * original_stds) + original_means
real_means.columns = ['Return Mean (Actual)', 'Volatility Mean (Actual)']

print("The actual stats for each states are:\n", real_means.applymap(lambda x: f"{x:.2%}"))

# Predict hidden states (regimes)


# print(scaled_features, scaled_features.shape)
hidden_states = model.predict(scaled_features)
# print(hidden_states)

#Sort states based on return (descending) and volatility (ascending)
sorted_states = real_means.sort_values(by=['Return Mean (Actual)', 'Volatility Mean (Actual)'], ascending=[False, True]).index.tolist()
# print(sorted_states)


# Map states to regimes based on mean return and volatility

state_to_regime= {}

for i, state in enumerate(sorted_states):
    if real_means.loc[state, 'Return Mean (Actual)'] >= 0:
        if i==0:
            state_to_regime[state]= "Bullish"
        elif real_means.loc[sorted_states[i+1], 'Return Mean (Actual)'] >0:
            state_to_regime[state]= "Mild Bullish"
        else :
            state_to_regime[state]= "Neutral"

             
    elif real_means.loc[state, 'Return Mean (Actual)'] < 0:
        if i == len(sorted_states) -1:
            state_to_regime[state]= "Bearish"
        elif state_to_regime[sorted_states[i-1]] == "Neutral":
            state_to_regime[state] = "Mild Bearish"
        else:
            state_to_regime[state] = "Neutral"


######### Adding Regimes and Hidden states to the Database

data['Regime'] = [state_to_regime[state] for state in hidden_states]
data['Hidden States']= hidden_states
data.to_excel("Regiems_model.xlsx")


# Predict the next day's regime probabilities using the transition matrix

# Get transition probability matrix
trans_mat = model.transmat_

# Get last observed regime
last_state = hidden_states[-1]  

# Compute probabilities for next day's regime

next_day_probs = trans_mat[last_state]  

# Convert to readable format
prob_dict = {state_to_regime[i]: prob for i, prob in enumerate(next_day_probs)}

# Print results

print(f"The nature of market for the last day was {state_to_regime[last_state]}\n ")
print("Predicted Regime Probabilities for the Next Day:")
for regime, prob in prob_dict.items():
    print(f"{regime}: {prob:.2%}")


# Define regime colors
regime_colors = {
    "Bullish": "green",
    "Mild Bullish": "yellow",
    "Neutral": "blue",
    "Bearish": "red",
    "Mild Bearish": "pink"
}

# Convert index to datetime for proper plotting
data.index = pd.to_datetime(data.index, format="%d-%b-%y")  # Adjust format as per your CSV

# Plot Price Line
plt.figure(figsize=(14, 8))
plt.plot(data.index, data['Close'], label="NIFTY 50 Price", alpha=0.7, color="blue")

# Fill regimes continuously without gaps
prev_date = data.index[0]  # Start from the first date
prev_regime = data['Regime'].iloc[0]  # Get the first regime

for date, regime in zip(data.index, data['Regime']):
    if regime != prev_regime:  # Regime change detected
        plt.axvspan(prev_date, date, color=regime_colors[prev_regime], alpha=0.3)
        prev_date = date  # Update start date for the next region
        prev_regime = regime  # Update previous regime

# Fill the last regime region (to avoid cutting off at the last transition)
plt.axvspan(prev_date, data.index[-1], color=regime_colors[prev_regime], alpha=0.3)

# Highlight regime change dates on x-axis
regime_changes = data.index[data['Regime'] != data['Regime'].shift(1)]
for change_date in regime_changes:
    plt.axvline(x=change_date, color='black', linestyle='dotted', alpha=0.8)
    plt.text(change_date, data['Close'].max(), change_date.strftime('%Y-%m-%d'), 
             rotation=45, fontsize=8, verticalalignment='top')

# Format x-axis
plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=2))  # Show every 2 months
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))  # Format as 'YYYY-MM'

plt.title("NIFTY 50 Price with Market Regimes (HMM)")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.grid(True, linestyle='dotted', alpha=0.5)
plt.xticks(rotation=45)

# ------> **Legend 1: Regime Colors**
patches = [mpatches.Patch(color=color, label=regime) for regime, color in regime_colors.items()]
plt.legend(handles=patches, title="Market Regimes", loc="upper left", fontsize=10)

# ------> **Legend 2: Actual Return & Volatility per State**
real_means_legend = "\n".join(
    [f"{state_to_regime[i]}: Return {ret:.2%}, Volatility {vol:.2%}" for i, (ret, vol) in enumerate(real_means.values)]
)

# Add text box for real mean values
plt.gcf().text(0.75, 0.85, real_means_legend, fontsize=10, 
               bbox=dict(facecolor='white', alpha=0.3, edgecolor='black'))

plt.show()

