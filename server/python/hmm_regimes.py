import sys
import json
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
import yfinance as yf
import warnings
warnings.filterwarnings("ignore")

def fetch_data(symbol="DOGE-USD", period="1y"):
    df = yf.download(symbol, period=period, progress=False)
    if df.empty:
        raise ValueError(f"No data found for {symbol}")
    # Handle multi-index columns from yfinance
    if isinstance(df.columns, pd.MultiIndex):
        closes = df['Close'][symbol].values.flatten()
        volumes = df['Volume'][symbol].values.flatten()
    else:
        closes = df['Close'].values.flatten()
        volumes = df['Volume'].values.flatten()
    return closes, volumes

def prepare_features(closes, volumes):
    # Calculate returns
    returns = np.diff(closes) / closes[:-1]
    # Calculate log returns
    log_returns = np.log(closes[1:] / closes[:-1])
    
    # Calculate volatility (e.g., 5-period rolling std dev)
    ret_series = pd.Series(log_returns)
    volatility = ret_series.rolling(window=5).std().fillna(0).values
    
    # Combine features (Returns, Volatility)
    # We drop the first element of closes/volumes to match the length of returns
    X = np.column_stack([log_returns, volatility])
    return X

def train_hmm(X, n_components=3):
    model = GaussianHMM(
        n_components=n_components, 
        covariance_type="diag", 
        n_iter=1000, 
        random_state=42,
        min_covar=1e-3
    )
    model.fit(X)
    return model

def interpret_regimes(model, X):
    # This is a heuristic way to interpret regimes based on their means and variances
    # Ensure we are working with 1D arrays of scalars
    means = np.array(model.means_[:, 0]).ravel()
    
    # Handle diagonal covariance shape (n_components, n_features)
    # We take the first feature (returns) variance
    variances = np.array(model.covars_[:, 0]).ravel()
    median_var = float(np.median(variances))
    
    regimes = {}
    for i in range(len(means)):
        m = float(means[i])
        v = float(variances[i])
        
        if m > 0 and v < median_var:
            regimes[i] = "TREND_BULL"
        elif m < 0 and v < median_var:
            regimes[i] = "TREND_BEAR"
        else:
            regimes[i] = "RANGE" # High variance or near-zero mean
            
    # Fallback to simple sorting if logic is ambiguous
    sorted_by_vol = np.argsort(variances)
    if len(regimes) < model.n_components:
        regimes[sorted_by_vol[0]] = "TREND_BULL" if float(means[sorted_by_vol[0]]) > 0 else "TREND_BEAR"
        regimes[sorted_by_vol[1]] = "RANGE"
        regimes[sorted_by_vol[-1]] = "HIGH_VOLATILITY"
        
    return regimes

def main():
    try:
        input_data = sys.stdin.read().strip()
        if input_data:
            data = json.loads(input_data)
            closes = np.array(data.get("closes", [])).flatten()
            volumes = np.array(data.get("volumes", [])).flatten()
            
            # Si los datos vienen incompletos por alguna razón, fallback a yfinance
            if len(closes) < 30:
                closes, volumes = fetch_data()
        else:
            closes, volumes = fetch_data()
            
        X = prepare_features(closes, volumes)
        
        # Train HMM with 3 states (e.g., Bull, Bear, Range/Volatile)
        n_components = 3
        model = train_hmm(X, n_components=n_components)
        
        # Predict the hidden states for the entire sequence
        hidden_states = model.predict(X)
        
        # Interpret what each state means
        regimes_map = interpret_regimes(model, X)
        
        current_state = hidden_states[-1]
        current_regime = regimes_map.get(current_state, "UNKNOWN")
        
        # Get probabilities for the last observation
        last_obs = X[-1].reshape(1, -1)
        # Using model.predict_proba is not directly available, we can use predict
        # Or evaluate probabilities using model.score_samples or similar.
        # For simplicity, we just output the regime.
        
        result = {
            "status": "success",
            "current_regime": current_regime,
            "current_state_id": int(current_state),
            "state_descriptions": regimes_map,
            "observations_count": len(X)
        }
        
        print(json.dumps(result))
        
    except Exception as e:
        sys.stderr.write(str(e))
        sys.exit(1)

if __name__ == "__main__":
    main()
