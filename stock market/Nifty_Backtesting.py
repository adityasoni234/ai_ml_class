import pandas as pd

# Load Processed Data
processed_data = pd.read_csv("Nifty_Options_Strategy.pdf.csv", parse_dates=['time'])

# Parameters
ENTRY_OFFSET = 2
STOP_LOSS_OFFSET = 15
TARGET_OFFSET = 20

# Backtesting Function
def backtest_strategy(data):
    results = []
    active_trade = None

    for i, row in data.iterrows():
        if active_trade:
            # Check if the active trade hit stop loss or target
            if active_trade['type'] == 'call':
                if row['call_high'] >= active_trade['target_price']:
                    active_trade['exit_price'] = active_trade['target_price']
                    active_trade['exit_time'] = row['time']
                    active_trade['result'] = 'target_hit'
                    results.append(active_trade)
                    active_trade = None
                elif row['call_high'] <= active_trade['stop_loss']:
                    active_trade['exit_price'] = active_trade['stop_loss']
                    active_trade['exit_time'] = row['time']
                    active_trade['result'] = 'stop_loss'
                    results.append(active_trade)
                    active_trade = None

            elif active_trade['type'] == 'put':
                if row['put_high'] >= active_trade['target_price']:
                    active_trade['exit_price'] = active_trade['target_price']
                    active_trade['exit_time'] = row['time']
                    active_trade['result'] = 'target_hit'
                    results.append(active_trade)
                    active_trade = None
                elif row['put_high'] <= active_trade['stop_loss']:
                    active_trade['exit_price'] = active_trade['stop_loss']
                    active_trade['exit_time'] = row['time']
                    active_trade['result'] = 'stop_loss'
                    results.append(active_trade)
                    active_trade = None

            continue

        # Evaluate new trades
        if not active_trade and not pd.isna(row['call_high']):
            entry_price = row['call_high'] + ENTRY_OFFSET
            stop_loss = entry_price - STOP_LOSS_OFFSET
            target_price = entry_price + TARGET_OFFSET

            active_trade = {
                'type': 'call',
                'entry_time': row['time'],
                'entry_price': entry_price,
                'stop_loss': stop_loss,
                'target_price': target_price,
                'strike_price': row['strike_price_call'],
            }

        if not active_trade and not pd.isna(row['put_high']):
            entry_price = row['put_high'] + ENTRY_OFFSET
            stop_loss = entry_price - STOP_LOSS_OFFSET
            target_price = entry_price + TARGET_OFFSET

            active_trade = {
                'type': 'put',
                'entry_time': row['time'],
                'entry_price': entry_price,
                'stop_loss': stop_loss,
                'target_price': target_price,
                'strike_price': row['strike_price_put'],
            }

    return pd.DataFrame(results)

# Backtest the Strategy
backtest_results = backtest_strategy(processed_data)

# Save Results
backtest_results.to_csv("backtest_results.csv", index=False)

# Summary Metrics
total_trades = len(backtest_results)
wins = len(backtest_results[backtest_results['result'] == 'target_hit'])
losses = len(backtest_results[backtest_results['result'] == 'stop_loss'])
win_rate = wins / total_trades * 100 if total_trades > 0 else 0

print(f"Total Trades: {total_trades}")
print(f"Wins: {wins}")
print(f"Losses: {losses}")
print(f"Win Rate: {win_rate:.2f}%")
