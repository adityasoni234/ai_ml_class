import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load Processed Data
processed_data = pd.read_csv("processed_data.csv")

# Feature Engineering for ML Model
processed_data['is_call'] = processed_data['strike_price_call'].notna()
processed_data['entry_price'] = processed_data.apply(
    lambda row: row['call_high'] + 2 if row['is_call'] else row['put_high'] + 2, axis=1
)
processed_data['target_price'] = processed_data['entry_price'] + 20
processed_data['stop_loss'] = processed_data['entry_price'] - 15

# Labeling Data (Target/Stop Loss hit classification)
processed_data['result'] = processed_data.apply(
    lambda row: 1 if (row['is_call'] and row['call_high'] >= row['target_price']) or 
                    (not row['is_call'] and row['put_high'] >= row['target_price']) else 0, axis=1
)

# Prepare Features and Labels
features = processed_data[['entry_price', 'strike_price_call', 'strike_price_put']].fillna(0)
labels = processed_data['result']

# Split Data into Train and Test
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Train Machine Learning Model
model = RandomForestClassifier(random_state=42, n_estimators=100)
model.fit(X_train, y_train)

# Evaluate Model
predictions = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, predictions))
print("Classification Report:\n", classification_report(y_test, predictions))

# Save the Model
import joblib
joblib.dump(model, "nifty_model.pkl")
print("Model training completed and saved as nifty_model.pkl")
