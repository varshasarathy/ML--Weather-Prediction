# Implementation of Random Forest Algorithm for Weather Prediction
## AIM:
To write a program to predict daily temperature , PM2.5 pollution level and Energy based on environmental sensor data using Random Forest Algorithm.

## Problem Statement and Dataset
Accurate prediction of environmental conditions such as temperature, air pollution (PM2.5), and solar energy is essential for effective urban planning, public health monitoring, and energy management. Environmental sensor data collected from weather stations often contain noise, missing values, and time-dependent patterns, making prediction challenging.

The objective of this project is to develop a machine learning model using the Random Forest Algorithm to predict daily temperature, PM2.5 pollution levels, and solar radiation (energy) based on historical environmental sensor data. The model should preprocess the dataset, handle missing values, extract relevant time-based and lag features, and provide accurate predictions along with performance evaluation.

The dataset used in this project is collected from an environmental weather station and stored in a CSV file named weather-station-eee-block_2024_07_13.csv. It contains time-series sensor readings recorded at regular intervals.

attributes in dataset:

*time

*tem(temperature)

*pm2_5 (PM2.5 Pollution Level) 

*hum(humidity)

*tsr (Total Solar Radiation / Energy)

*pressure (Atmospheric Pressure)

*wind_speed

*illumination

*co2 (Carbon Dioxide hLevel)

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load and preprocess weather dataset (clean, sort, fill missing values). 
2. Extract time-based features and create lag variables.
3. Split data into training and testing sets.
4. Evaluate performance and predict future values using latest data.

## Program:
```
/*
Program to implement the Random Forest Algorithm to predict daily temperature , PM2.5 pollution level and Energy based on environmental sensor data.
Developed by: VARSHA SARATHI
RegisterNumber:  21223040233
*/
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

df = pd.read_csv("weather-station-eee-block_2024_07_13.csv")
df.columns = df.columns.str.strip()

df['time'] = pd.to_datetime(df['time'])
df = df.sort_values('time').reset_index(drop=True)

cols_to_fill = ['tem', 'pm2_5', 'tsr', 'hum', 'pressure', 'wind_speed', 'illumination', 'co2']
for col in cols_to_fill:
    if col in df.columns:
        df[col] = df[col].interpolate(method='linear', limit=10)

df['hour'] = df['time'].dt.hour
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

targets = ['tem', 'pm2_5', 'tsr']
for t in targets:
    df[f'{t}_lag1'] = df[t].shift(1)
    df[f'{t}_lag2'] = df[t].shift(2)

processed_df = df.dropna(subset=['tem_lag2', 'pm2_5_lag2', 'tsr_lag2', 'hum', 'pressure']).reset_index(drop=True)
processed_df.to_csv("combined_processed_weather_data.csv", index=False)

features = [
    'hum', 'pressure', 'wind_speed', 'illumination', 'co2',
    'hour_sin', 'hour_cos', 'tem_lag1', 'pm2_5_lag1', 'tsr_lag1'
]
print("--- Feature Engineering Summary ---")
print(f"Original rows: {len(df)}")
print(f"Processed rows (after lags/cleaning): {len(processed_df)}")
print(f"Final high-performance feature set:",features)

split_idx = int(len(processed_df) * 0.8)
train, test = processed_df.iloc[:split_idx], processed_df.iloc[split_idx:]
X_train, X_test = train[features], test[features]

models = {}
results = {}

target_meta = {
    'tem': ('Temperature', '°C', 'red'),
    'pm2_5': ('Pollution (PM2.5)', 'µg/m³', 'green'),
    'tsr': ('Energy (Solar Radiation)', 'W/m²', 'orange')
}

for target in targets:
    y_train, y_test = train[target], test[target]
    
    model = RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42)
    model.fit(X_train, y_train);
    
    preds = model.predict(X_test)
    models[target] = model
    
    results[target] = {
        'r2': r2_score(y_test, preds),
        'mae': mean_absolute_error(y_test, preds),
        'preds': preds,
        'actual': y_test.values
    }

fig, axes = plt.subplots(3, 2, figsize=(16, 18))

for i, target in enumerate(targets):
    label, unit, color = target_meta[target]
    res = results[target]
    
    axes[i, 0].plot(res['actual'][-150:], label='Actual', color='black', alpha=0.4, linewidth=2)
    axes[i, 0].plot(res['preds'][-150:], label='Predicted', color=color, linestyle='--', linewidth=2)
    axes[i, 0].set_title(f"{label}: Actual vs Predicted\n$R^2$: {res['r2']:.3f} | MAE: {res['mae']:.2f}")
    axes[i, 0].set_ylabel(unit)
    axes[i, 0].legend()
    axes[i, 0].grid(True, alpha=0.3)
    
    importances = pd.Series(models[target].feature_importances_, index=features).sort_values()
    importances.plot(kind='barh', ax=axes[i, 1], color=color, alpha=0.7)
    axes[i, 1].set_title(f"Key Drivers: {label}")

plt.tight_layout()
plt.show()

last_row = processed_df.iloc[-1]
latest_data = pd.DataFrame([
    {
        'hum': last_row['hum'], 'pressure': last_row['pressure'], 'wind_speed': last_row['wind_speed'],
        'illumination': last_row['illumination'], 'co2': last_row['co2'],
        'hour_sin': last_row['hour_sin'], 'hour_cos': last_row['hour_cos'],
        'tem_lag1': last_row['tem'], 'pm2_5_lag1': last_row['pm2_5'], 'tsr_lag1': last_row['tsr']
    }
])

print("\n--- NEXT STEP PREDICTIONS (Using Latest Data) ---")
for target in targets:
    pred_val = models[target].predict(latest_data)[0]
    print(f"Predicted {target_meta[target][0]}: {pred_val:.2f} {target_meta[target][1]}")
```

## Output:

<img width="1267" height="553" alt="Screenshot 2026-03-18 084324" src="https://github.com/user-attachments/assets/043f187b-574a-4a34-aba1-265228347332" />

<img width="1274" height="467" alt="Screenshot 2026-03-18 084354" src="https://github.com/user-attachments/assets/b58cb545-321e-4689-9742-19b321c525aa" />

<img width="1289" height="561" alt="Screenshot 2026-03-18 084411" src="https://github.com/user-attachments/assets/3b74086a-8715-494d-843f-2946fd93f803" />

## Result:
Thus the program to implement Random Forest for weather prediction is written and verified using Python Programming.
