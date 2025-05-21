# Regime Shift Detection in Financial Markets

This repository presents algorithms to detect market regime shifts using **Hidden Markov Models (HMM)** and **K-Means clustering**.  
By analyzing historical financial data, the models aim to identify distinct market states—such as bullish, bearish, or sideways—facilitating informed investment decisions.

---

## 🧠 Overview

Financial markets transition through regimes characterized by different return and volatility patterns. Identifying these shifts enables better asset allocation and risk management.

This project implements:

- **Hidden Markov Models (HMM):** Probabilistic model for uncovering latent regimes.
- **K-Means Clustering:** Unsupervised learning to categorize time periods into market states.

---

## 📁 Repository Structure

| File                   | Description                                                  |
|------------------------|--------------------------------------------------------------|
| `HiddenMM.py`          | HMM-based regime detection script.                           |
| `HMM_backtest.py`      | Backtests strategy using HMM regime classifications.         |
| `K-means.py`           | K-Means-based regime detection script.                       |
| `K-backtest.py`        | Backtests strategy using K-Means regime classifications.     |
| `Nifty.csv`            | Historical Nifty 50 index data used for model training.      |
| `HMM_regime.xlsx`      | Excel output of HMM regime predictions.                      |
| `KMeans_Regimes.xlsx`  | Excel output of K-Means regime predictions.                  |
| `README.md`            | Project documentation.                                       |

---

## ⚙️ Getting Started

### 📦 Prerequisites

Install the required Python libraries:

```bash
pip install numpy pandas matplotlib hmmlearn scikit-learn yfinance
```

### 📥 Get the Data (`Nifty.csv`)

#### Option 1: Download from Official Website

1. Visit: [https://www.niftyindices.com/reports/historical-data](https://www.niftyindices.com/reports/historical-data)
2. Select:  
   - Index: `NIFTY 50`  
   - Time Range: Desired start and end dates  
3. Download the `.csv` file
4. Rename the file to `Nifty.csv` and place it in the project directory

#### Option 2: Use `yfinance` in Python

You can also generate `Nifty.csv` using the following snippet:

```python
import yfinance as yf

# Download historical data for NIFTY 50 (represented by ^NSEI)
df = yf.download("^NSEI", start="2010-01-01", end="2024-12-31")
df.to_csv("Nifty.csv")
```

---

## 🚀 Running the Models

### Hidden Markov Model (HMM)

```bash
python HiddenMM.py
python HMM_backtest.py
```

### K-Means Clustering

```bash
python K-means.py
python K-backtest.py
```

---

## 📊 Model Comparison

Backtesting results demonstrate that **HMM consistently outperforms K-Means** in regime classification, yielding lower Mean Squared Error (MSE):

| Model     | Mean Squared Error (MSE) |
|-----------|--------------------------|
| HMM       | **0.0163**               |
| K-Means   | 0.0279                   |

> You can verify these results by running the `HMM_backtest.py` and `K-backtest.py` scripts.

---

## 🔍 Methodology

### Hidden Markov Model (HMM)

- Uses market features (e.g., returns, volatility) to infer hidden states
- Learns transition probabilities between regimes
- Outputs smoothed predictions for better robustness

### K-Means Clustering

- Clusters historical data into `k` groups based on features
- Each cluster represents a distinct market regime
- Simpler and faster but less accurate in dynamic financial contexts

---

## 📈 Results & Insights

- HMM more accurately identifies shifts in volatility and returns
- K-Means works as a quick approximation but lags in noisy data
- The models provide a foundation for regime-based investment strategies

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

## 🙏 Acknowledgments

- [hmmlearn](https://github.com/hmmlearn/hmmlearn)
- [scikit-learn](https://scikit-learn.org/)
- [yfinance](https://github.com/ranaroussi/yfinance)
