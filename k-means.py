import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# -------------- DATA PREP -----------------
data = pd.read_csv("Nifty.csv", index_col=1)
data.index = pd.to_datetime(data.index, format="%d-%b-%y")
data['Return'] = data['Close'].pct_change()
data['Volatility'] = data['Return'].rolling(window=21).std()
data.dropna(inplace=True)

features = data[['Return', 'Volatility']]
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# -------------- AUTOMATED ELBOW -----------
k_min, k_max = 1, 10
inertias = []

for k in range(k_min, k_max + 1):
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    km.fit(scaled_features)
    inertias.append(km.inertia_)

# Try to locate the "knee" (elbow) automatically
try:
    from kneed import KneeLocator
    kl = KneeLocator(
        range(k_min, k_max + 1),
        inertias,
        curve="convex",
        direction="decreasing",
    )
    optimal_k = kl.elbow
except ImportError:
    # Fallback: largest relative drop heuristic
    pct_drop = -np.diff(inertias) / inertias[:-1]
    elbow_candidates = np.where(pct_drop < 0.10)[0]
    optimal_k = elbow_candidates[0] + k_min if len(elbow_candidates) else 4  # default to 4

print(f"Optimal k selected by Elbow method: {optimal_k}")

# OPTIONAL: Plot the Elbow curve for verification
plt.figure(figsize=(8, 5))
plt.plot(range(k_min, k_max + 1), inertias, "o-")
plt.axvline(optimal_k, color="red", linestyle="--", label=f"Elbow @ k={optimal_k}")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia (WCSS)")
plt.title("Elbow Method – Automatic Knee Detection")
plt.legend()
plt.tight_layout()
plt.show()

# ---------- K-MEANS REGIME CLUSTERING ----------
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
data["Cluster"] = kmeans.fit_predict(scaled_features)

cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)
cluster_info = pd.DataFrame(cluster_centers, columns=["Return", "Volatility"])

# Label regimes by sorted return/volatility
cluster_info["Regime"] = ""
sorted_clusters = cluster_info.sort_values(
    by=["Return", "Volatility"], ascending=[False, True]
).index
labels = ["Bullish", "Mild Bullish", "Neutral", "Bearish"][: len(sorted_clusters)]

for label, cluster_idx in zip(labels, sorted_clusters):
    cluster_info.at[cluster_idx, "Regime"] = label

cluster_to_regime = cluster_info["Regime"].to_dict()
data["Regime"] = data["Cluster"].map(cluster_to_regime)

# Save to Excel
data.to_excel("KMeans_Regimes.xlsx")

# --------- PLOTTING REGIMES -------------
regime_colors = {
    "Bullish": "green",
    "Mild Bullish": "yellow",
    "Neutral": "blue",
    "Bearish": "red"
}

plt.figure(figsize=(14, 8))
plt.plot(data.index, data['Close'], label="NIFTY 50", alpha=0.7, color="black")

# Regime shading
prev_date = data.index[0]
prev_regime = data['Regime'].iloc[0]
for date, regime in zip(data.index, data['Regime']):
    if regime != prev_regime:
        plt.axvspan(prev_date, date, color=regime_colors.get(prev_regime, "gray"), alpha=0.3)
        prev_date = date
        prev_regime = regime
plt.axvspan(prev_date, data.index[-1], color=regime_colors.get(prev_regime, "gray"), alpha=0.3)

# Mark regime changes
regime_changes = data.index[data['Regime'] != data['Regime'].shift(1)]
for change_date in regime_changes:
    plt.axvline(x=change_date, color='black', linestyle='dotted', alpha=0.7)
    plt.text(change_date, data['Close'].max(), change_date.strftime('%Y-%m-%d'), 
             rotation=45, fontsize=8, verticalalignment='top')

plt.title("NIFTY 50 Market Regimes Using K-Means Clustering")
plt.xlabel("Date")
plt.ylabel("Price")
plt.grid(True, linestyle='dotted', alpha=0.5)
plt.xticks(rotation=45)
plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=2))
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

# Add legends
patches = [mpatches.Patch(color=color, label=regime) for regime, color in regime_colors.items()]
plt.legend(handles=patches, title="Market Regimes", loc="upper left")

# Add regime stats
text_box = "\n".join(
    [f"{row['Regime']}: Return {row['Return']:.2%}, Vol {row['Volatility']:.2%}" 
     for idx, row in cluster_info.iterrows()]
)
plt.gcf().text(0.75, 0.85, text_box, fontsize=10, 
               bbox=dict(facecolor='white', alpha=0.3, edgecolor='black'))

plt.show()
