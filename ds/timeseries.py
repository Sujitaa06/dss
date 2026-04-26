pip install statsmodels

import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# Load dataset
df = pd.read_csv("sales.csv")

# Convert Date column to datetime
df["Date"] = pd.to_datetime(df["Date"])
df.set_index("Date", inplace=True)

# Plot data
df.plot(title="Sales Over Time")
plt.show()

# Apply ARIMA model
model = ARIMA(df["Sales"], order=(1,1,1))
model_fit = model.fit()

# Forecast next 5 days
forecast = model_fit.forecast(steps=5)

print("Forecasted Values:")
print(forecast)




import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

df = pd.read_csv("sales.csv")

df["Date"] = pd.to_datetime(df["Date"])
df.set_index("Date", inplace=True)

df = df.sort_index()

df.plot(title="Sales Over Time")
plt.show()

model = ARIMA(df["Sales"], order=(1,1,1))
model_fit = model.fit()

forecast = model_fit.forecast(steps=5)

print("Forecasted Values:")
print(forecast)