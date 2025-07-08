# accuracy_checker.py
#
# Αυτό το script εκτελεί έναν ιστορικό έλεγχο (backtest) για να μετρήσει
# την ακρίβεια του μοντέλου πρόβλεψης. Υπολογίζει το ποσοστό επιτυχίας
# για δύο διαφορετικά επίπεδα πιθανότητας: 90% και 50%.

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from arch import arch_model

# --- ΡΥΘΜΙΣΕΙΣ ---
SYMBOL = "GBPJPY"
TIMEFRAME = mt5.TIMEFRAME_M30
CANDLES_TO_DOWNLOAD = 300  # Περισσότερα κεριά για να έχουμε ιστορικό για τον έλεγχο
CANDLES_TO_BACKTEST = 100  # Πόσα από τα τελευταία κεριά θα ελέγξουμε


# --- ΛΟΓΙΚΗ ΣΥΝΔΕΣΗΣ & ΛΗΨΗΣ ΔΕΔΟΜΕΝΩΝ (Ίδια με πριν) ---
def connect_to_mt5():
    print("--- Προσπάθεια σύνδεσης στο MetaTrader 5 ---")
    if not mt5.initialize():
        print(f"❌ Αποτυχία σύνδεσης: {mt5.last_error()}")
        return False
    print("✅ Επιτυχής σύνδεση στο MetaTrader 5.")
    return True


def fetch_data(symbol, timeframe, n_candles):
    print(f"--- Λήψη {n_candles} κεριών για {symbol}... ---")
    try:
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, n_candles)
        if rates is None or len(rates) == 0: return pd.DataFrame()
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
        df.set_index('time', inplace=True)
        df.columns = df.columns.str.lower()
        print(f"✅ Λήψη {len(df)} κεριών ολοκληρώθηκε.")
        return df
    except Exception as e:
        print(f"🚨 Σφάλμα κατά τη λήψη δεδομένων: {e}")
        return pd.DataFrame()


# --- ΛΟΓΙΚΗ ΥΠΟΛΟΓΙΣΜΩΝ (Τροποποιημένη για πολλαπλά εύρη) ---

def calculate_garch_volatility(data, p=1, q=1, dist='t'):
    # Αυτή η συνάρτηση παραμένει ίδια
    returns = 100 * np.log(data['close'] / data['close'].shift(1)).dropna()
    if returns.empty or len(returns) < p + q + 10: return pd.Series(index=data.index, dtype=float)
    try:
        am = arch_model(returns, vol='Garch', p=p, q=q, mean='Constant', dist=dist, rescale=False)
        res = am.fit(disp='off', show_warning=False)
        volatility_series = (res.conditional_volatility / 100.0).reindex(data.index).bfill()
        return volatility_series
    except Exception:
        return pd.Series(index=data.index, dtype=float)


def calculate_price_distribution(df_main, fallback_vol_window=12):
    """
    Τροποποιήθηκε για να επιστρέφει ΚΑΙ το εύρος 50% (P25-P75).
    """
    if df_main is None or len(df_main) < 2: return None

    # Η πρόβλεψη γίνεται για το τελευταίο κερί του df_main,
    # χρησιμοποιώντας τα δεδομένα ΠΡΙΝ από αυτό.
    x0 = df_main['close'].iloc[-2]

    # Υπολογισμός sigma (GARCH ή ιστορικό)
    sigma = np.nan
    df_history = df_main.iloc[:-1]  # Δεδομένα μέχρι το προηγούμενο κερί
    if 'garch_forecast' in df_history.columns:
        garch_val = df_history['garch_forecast'].iloc[-1]
        if pd.notna(garch_val) and garch_val > 0:
            sigma = garch_val

    if pd.isna(sigma):
        if len(df_history) >= fallback_vol_window + 1:
            log_returns = np.log(df_history['close'] / df_history['close'].shift(1)).dropna()
            if len(log_returns) >= fallback_vol_window:
                sigma = log_returns.rolling(window=fallback_vol_window).std().iloc[-1]

    if pd.isna(sigma) or sigma <= 0: return None

    # Προσομοίωση Monte Carlo
    n_paths, mu = 50000, 0
    Z = np.random.randn(n_paths)
    final_prices = x0 * np.exp((mu - 0.5 * sigma ** 2) + sigma * Z)

    # Υπολογισμός στατιστικών για πολλαπλά επίπεδα
    stats = {
        'low_90': np.percentile(final_prices, 5),  # 5ο εκατοστημόριο
        'high_90': np.percentile(final_prices, 95),  # 95ο εκατοστημόριο
        'low_50': np.percentile(final_prices, 25),  # 25ο εκατοστημόριο
        'high_50': np.percentile(final_prices, 75)  # 75ο εκατοστημόριο
    }
    return stats


# --- ΚΥΡΙΩΣ ΛΟΓΙΚΗ BACKTEST ---

def run_accuracy_backtest(df_full):
    print("\n--- Εκκίνηση Ιστορικού Ελέγχου (Backtest) Ακρίβειας ---")
    if df_full is None or df_full.empty:
        print("❌ Δεν υπάρχουν δεδομένα για τον έλεγχο.")
        return

    # Προσθήκη στήλης GARCH σε όλα τα δεδομένα μία φορά
    df_full['garch_forecast'] = calculate_garch_volatility(df_full)
    print("✅ Ολοκληρώθηκε ο αρχικός υπολογισμός GARCH για όλα τα δεδομένα.")

    hits_90_percent = 0
    hits_50_percent = 0
    predictions_made = 0

    start_index = len(df_full) - CANDLES_TO_BACKTEST

    for i in range(start_index, len(df_full)):
        # Για κάθε κερί 'i', χρησιμοποιούμε όλα τα δεδομένα μέχρι και το 'i'
        # για να προβλέψουμε το 'i' (η συνάρτηση θα χρησιμοποιήσει τα δεδομένα μέχρι το i-1)
        df_slice = df_full.iloc[:i + 1]

        # Πραγματική τιμή κλεισίματος του κεριού που ελέγχουμε
        actual_close = df_slice['close'].iloc[-1]

        # Κάνουμε την πρόβλεψη
        pred_stats = calculate_price_distribution(df_slice)

        if pred_stats:
            predictions_made += 1

            # Έλεγχος για το εύρος 90%
            if pred_stats['low_90'] <= actual_close <= pred_stats['high_90']:
                hits_90_percent += 1

            # Έλεγχος για το εύρος 50%
            if pred_stats['low_50'] <= actual_close <= pred_stats['high_50']:
                hits_50_percent += 1

    print("\n--- ΑΠΟΤΕΛΕΣΜΑΤΑ BACKTEST ---")
    if predictions_made > 0:
        accuracy_90 = (hits_90_percent / predictions_made) * 100
        accuracy_50 = (hits_50_percent / predictions_made) * 100

        print(f"Σύνολο Κεριών που Ελέγχθηκαν: {predictions_made}")
        print("-" * 35)
        print(f"🎯 Ακρίβεια στο 90% Εύρος: {accuracy_90:.1f}%  ({hits_90_percent}/{predictions_made} επιτυχίες)")
        print(f"🎯 Ακρίβεια στο 50% Εύρος: {accuracy_50:.1f}%  ({hits_50_percent}/{predictions_made} επιτυχίες)")
        print("-" * 35)
        print("\n* Σημείωση: 'Ακρίβεια' σημαίνει το ποσοστό των φορών που η πραγματική τιμή κλεισίματος")
        print("  βρέθηκε μέσα στο προβλεπόμενο εύρος πιθανοτήτων.")
    else:
        print("Δεν ήταν δυνατή η εκτέλεση προβλέψεων για τον έλεγχο.")


# --- ΚΥΡΙΩΣ ΠΡΟΓΡΑΜΜΑ ---
if __name__ == "__main__":
    if connect_to_mt5():
        # Κατεβάζουμε τα δεδομένα μόνο μία φορά
        df_historical_data = fetch_data(SYMBOL, TIMEFRAME, CANDLES_TO_DOWNLOAD)

        # Εκτελούμε τον έλεγχο ακρίβειας
        run_accuracy_backtest(df_historical_data)

        # Αποσύνδεση
        mt5.shutdown()
    print("\n👋 Το πρόγραμμα ολοκληρώθηκε.")