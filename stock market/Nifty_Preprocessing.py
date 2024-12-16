import pandas as pd
import numpy as np

# Load Data
# Assuming spot_chart.csv and premium_chart.csv have columns like 'time', 'open', 'high', 'low', 'close'
spot_chart = pd.read_csv("spot_chart.csv", parse_dates=['time'])
premium_chart = pd.read_csv("premium_chart.csv", parse_dates=['time'])
# Step 1: Timeframe Selection
def select_timeframe(df):
    df['timeframe'] = np.where(df['time'].dt.time <= pd.to_datetime("10:30:00").time(), '1min', '3min')
    return df

spot_chart = select_timeframe(spot_chart)

# Step 2: Identify Swing Patterns
def find_swing_patterns(df, direction='up'):  # 'up' for red candles, 'down' for green candles
    swing_col = 'red_count' if direction == 'up' else 'green_count'
    
    df[swing_col] = 0
    consecutive = 0
    for i in range(1, len(df)):
        if direction == 'up' and df.loc[i, 'close'] < df.loc[i, 'open']:  # Red candle
            consecutive += 1
        elif direction == 'down' and df.loc[i, 'close'] > df.loc[i, 'open']:  # Green candle
            consecutive += 1
        else:
            consecutive = 0
        
        df.loc[i, swing_col] = consecutive

    return df

spot_chart = find_swing_patterns(spot_chart, direction='up')
spot_chart = find_swing_patterns(spot_chart, direction='down')

# Step 3: Mark Swing Highs/Lows
def mark_swing_levels(df, swing_col, level_type):  # 'high' for swing highs, 'low' for swing lows
    swing_level = []
    for i in range(len(df)):
        if df.loc[i, swing_col] >= 3:
            level = df.loc[max(i - 2, 0):i, level_type].max() if level_type == 'high' else df.loc[max(i - 2, 0):i, level_type].min()
            swing_level.append(level)
        else:
            swing_level.append(None)
    df[f'{level_type}_swing'] = swing_level
    return df

spot_chart = mark_swing_levels(spot_chart, 'red_count', 'high')
spot_chart = mark_swing_levels(spot_chart, 'green_count', 'low')

# Step 4: Option Strike Price Selection
def select_strike_price(swing_level, step=50):
    if pd.isna(swing_level):
        return None
    return int(np.round(swing_level / step) * step)

spot_chart['strike_price_call'] = spot_chart['high_swing'].apply(lambda x: select_strike_price(x))
spot_chart['strike_price_put'] = spot_chart['low_swing'].apply(lambda x: select_strike_price(x))

# Step 5: Merge with Premium Chart
def merge_with_premium(spot_df, premium_df):
    result = []
    for _, row in spot_df.iterrows():
        if pd.isna(row['strike_price_call']) and pd.isna(row['strike_price_put']):
            continue
        
        # Filter premium chart for the corresponding strike price and timeframe
        call_data = premium_df[(premium_df['strike_price'] == row['strike_price_call']) &
                               (premium_df['time'] <= row['time'])]
        put_data = premium_df[(premium_df['strike_price'] == row['strike_price_put']) &
                              (premium_df['time'] <= row['time'])]

        # Record the high of the premium chart for each swing
        call_high = call_data['high'].max() if not call_data.empty else None
        put_high = put_data['high'].max() if not put_data.empty else None

        result.append({
            'time': row['time'],
            'strike_price_call': row['strike_price_call'],
            'strike_price_put': row['strike_price_put'],
            'call_high': call_high,
            'put_high': put_high
        })

    return pd.DataFrame(result)

merged_data = merge_with_premium(spot_chart, premium_chart)

# Step 6: Save Processed Data
merged_data.to_csv("processed_data.csv", index=False)

print("Data preprocessing completed. Processed data saved as processed_data.csv")
