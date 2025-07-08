# run_simple_prediction.py (v5 - Τελική Έκδοση)
#
# Τελική, αυτόνομη έκδοση που ενσωματώνει όλες τις λειτουργίες:
# 1. Σύνδεση στο MT5.
# 2. Λήψη δεδομένων.
# 3. Έλεγχος κατάστασης (Bullish/Bearish/Sideways) με κατώφλι και σαφή ένδειξη του κεριού ελέγχου.
# 4. Υπολογισμός GARCH volatility.
# 5. Πρόβλεψη Monte Carlo για το επόμενο κερί με απλή ορολογία.

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import time
from arch import arch_model

# --- ΡΥΘΜΙΣΕΙΣ ---
SYMBOL = "GBPJPY"
TIMEFRAME = mt5.TIMEFRAME_M30
CANDLES_TO_DOWNLOAD = 200
MA_PERIOD = 9
# Κατώφλι για τον χαρακτηρισμό "Sideways".
NEUTRAL_THRESHOLD = 0.005


# --- 1. ΛΟΓΙΚΗ ΣΥΝΔΕΣΗΣ & ΛΗΨΗΣ ΔΕΔΟΜΕΝΩΝ ---

def connect_to_mt5():
    """Συνδέεται στο MT5. Επιστρέφει True αν επιτύχει, αλλιώς False."""
    print("--- Προσπάθεια σύνδεσης στο MetaTrader 5 ---")
    if not mt5.initialize():
        print(f"❌ Αποτυχία σύνδεσης: {mt5.last_error()}")
        return False
    print("✅ Επιτυχής σύνδεση στο MetaTrader 5.")
    return True


def fetch_data(symbol, timeframe, n_candles):
    """Κατεβάζει τα τελευταία Ν κεριά και τα επιστρέφει ως DataFrame."""
    print(f"--- Λήψη {n_candles} κεριών για {symbol}... ---")
    try:
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, n_candles)
        if rates is None or len(rates) == 0:
            print(f"⚠️ Δεν βρέθηκαν δεδομένα για {symbol}.")
            return pd.DataFrame()

        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
        df.set_index('time', inplace=True)
        df.columns = df.columns.str.lower()
        print(f"✅ Λήψη {len(df)} κεριών ολοκληρώθηκε.")
        return df
    except Exception as e:
        print(f"🚨 Σφάλμα κατά τη λήψη δεδομένων: {e}")
        return pd.DataFrame()


# --- 2. ΛΟΓΙΚΗ ΥΠΟΛΟΓΙΣΜΩΝ ---

def calculate_garch_volatility(data, p=1, q=1, dist='t'):
    """Υπολογίζει την GARCH volatility."""
    print(f"--- Υπολογισμός GARCH({p},{q}) Volatility... ---")
    if 'close' not in data.columns:
        return pd.Series(index=data.index, dtype=float)
    returns = 100 * np.log(data['close'] / data['close'].shift(1)).dropna()
    if returns.empty or len(returns) < p + q + 10:
        return pd.Series(index=data.index, dtype=float)
    try:
        am = arch_model(returns, vol='Garch', p=p, q=q, mean='Constant', dist=dist, rescale=False)
        res = am.fit(disp='off', show_warning=False)
        volatility_series = (res.conditional_volatility / 100.0).reindex(data.index).bfill()
        print("✅ Ο υπολογισμός GARCH Volatility ολοκληρώθηκε.")
        return volatility_series
    except Exception:
        return pd.Series(index=data.index, dtype=float)


def check_ma_status(df, period=MA_PERIOD, threshold=NEUTRAL_THRESHOLD):
    """Ελέγχει το crossover των EMA(Open) και EMA(Close) του τελευταίου ολοκληρωμένου κεριού."""
    print(f"--- Έλεγχος Κατάστασης Κινητών Μέσων Όρων (EMA {period})... ---")
    if 'open' not in df.columns or 'close' not in df.columns or len(df) < period:
        return "Άγνωστη"

    ema_open = df['open'].ewm(span=period, adjust=False).mean()
    ema_close = df['close'].ewm(span=period, adjust=False).mean()

    last_ema_open = ema_open.iloc[-1]
    last_ema_close = ema_close.iloc[-1]

    difference = last_ema_close - last_ema_open

    last_candle_time = df.index[-1].strftime('%Y-%m-%d %H:%M')
    print(f"  > Έλεγχος για το κερί που έκλεισε στις: {last_candle_time}")

    num_decimals = 3 if 'JPY' in SYMBOL.upper() else 5
    print(f"  > Τελευταία τιμή EMA(open):  {last_ema_open:.{num_decimals}f}")
    print(f"  > Τελευταία τιμή EMA(close): {last_ema_close:.{num_decimals}f}")
    print(f"  > Διαφορά (Close - Open):   {difference:.{num_decimals}f}")

    if difference > threshold:
        return "Bullish"
    elif difference < -threshold:
        return "Bearish"
    else:
        return "Sideways / Ουδέτερη"


def calculate_price_distribution(df_main, fallback_vol_window=12):
    """Εκτελεί την πρόβλεψη Monte Carlo για το επόμενο κερί."""
    print("--- Υπολογισμός Πρόβλεψης Κατανομής (Monte Carlo)... ---")
    if df_main is None or df_main.empty or 'close' not in df_main.columns or len(df_main) < 1:
        return None

    x0 = df_main['close'].iloc[-1]
    last_candle_time_pred = df_main.index[-1].strftime('%Y-%m-%d %H:%M')
    print(f"  > Τιμή εκκίνησης (x0) από το close του κεριού @ {last_candle_time_pred}: {x0:.5f}")

    sigma = np.nan
    if 'garch_forecast' in df_main.columns:
        garch_val = df_main['garch_forecast'].iloc[-1]
        if pd.notna(garch_val) and garch_val > 0:
            sigma = garch_val
            print(f"  > Χρήση μεταβλητότητας (σ) από GARCH: {sigma:.6f}")

    if pd.isna(sigma):
        print("  > Το GARCH απέτυχε, χρήση ιστορικής μεταβλητότητας...")
        if len(df_main) >= fallback_vol_window + 1:
            log_returns = np.log(df_main['close'] / df_main['close'].shift(1)).dropna()
            if len(log_returns) >= fallback_vol_window:
                sigma = log_returns.rolling(window=fallback_vol_window).std().iloc[-1]
                print(f"  > Χρήση ιστορικής μεταβλητότητας (σ): {sigma:.6f}")

    if pd.isna(sigma) or sigma <= 0: return None

    n_paths, mu = 50000, 0
    Z = np.random.randn(n_paths)
    final_prices = x0 * np.exp((mu - 0.5 * sigma ** 2) + sigma * Z)

    stats = {
        'mean': np.mean(final_prices),
        'low': np.percentile(final_prices, 5),
        'high': np.percentile(final_prices, 95),
        'x0': x0,
        'sigma': sigma
    }
    print("✅ Ο υπολογισμός της πρόβλεψης ολοκληρώθηκε.")
    return stats


# --- ΚΥΡΙΩΣ ΠΡΟΓΡΑΜΜΑ ---

if __name__ == "__main__":
    if connect_to_mt5():
        df_data = fetch_data(SYMBOL, TIMEFRAME, CANDLES_TO_DOWNLOAD)

        if not df_data.empty:
            ma_status = check_ma_status(df_data)
            df_data['garch_forecast'] = calculate_garch_volatility(df_data)
            prediction_stats = calculate_price_distribution(df_data)

            if prediction_stats:
                print("\n--- ΑΠΟΤΕΛΕΣΜΑΤΑ ---")
                num_decimals = 3 if 'JPY' in SYMBOL.upper() else 5
                print(f"Σύμβολο: {SYMBOL}")
                print("-" * 45)
                print(f"Κατάσταση Μέσων Όρων: {ma_status}")
                print("-" * 45)
                print(f"Τελευταία τιμή (x0):      {prediction_stats['x0']:.{num_decimals}f}")
                print(f"Μεταβλητότητα (σ):         {prediction_stats['sigma']:.6f}")
                print(f"Προβλεπόμενη Μέση Τιμή:    {prediction_stats['mean']:.{num_decimals}f}")
                print(f"Αναμενόμενο Low (P05):     {prediction_stats['low']:.{num_decimals}f}")
                print(f"Αναμενόμενο High (P95):    {prediction_stats['high']:.{num_decimals}f}")
                print("-" * 45)

        mt5.shutdown()
    print("\n👋 Το πρόγραμμα ολοκληρώθηκε.")