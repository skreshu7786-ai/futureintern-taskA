import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("SampleSuperstore.csv", encoding='latin1')

print(df.head())
print(df.columns)
# Convert Order Date to datetime
df['Order Date'] = pd.to_datetime(df['Order Date'])

# Remove duplicates
df = df.drop_duplicates()

# Check missing values
print(df.isnull().sum())

# Fill missing values if any
df = df.fillna(method='ffill')
daily_sales = df.groupby('Order Date')['Sales'].sum().reset_index()

daily_sales.columns = ['Date', 'Sales']

daily_sales = daily_sales.sort_values('Date')

print(daily_sales.head())
daily_sales['Year'] = daily_sales['Date'].dt.year
daily_sales['Month'] = daily_sales['Date'].dt.month
daily_sales['Day'] = daily_sales['Date'].dt.day
daily_sales['DayOfWeek'] = daily_sales['Date'].dt.dayofweek

# Lag features
daily_sales['Lag_1'] = daily_sales['Sales'].shift(1)
daily_sales['Lag_7'] = daily_sales['Sales'].shift(7)

# Rolling average
daily_sales['Rolling_7'] = daily_sales['Sales'].rolling(7).mean()

daily_sales = daily_sales.dropna()
from sklearn.model_selection import train_test_split

X = daily_sales.drop(['Date', 'Sales'], axis=1)
y = daily_sales['Sales']

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    shuffle=False
)
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)
predictions = model.predict(X_test)
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

mae = mean_absolute_error(y_test, predictions)

rmse = np.sqrt(
    mean_squared_error(y_test, predictions)
)

r2 = r2_score(y_test, predictions)

print("MAE :", round(mae,2))
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

mae = mean_absolute_error(y_test, predictions)

rmse = np.sqrt(
    mean_squared_error(y_test, predictions)
)

r2 = r2_score(y_test, predictions)

print("MAE :", round(mae,2))
print("RMSE:", round(rmse,2))
print("R2 Score:", round(r2,2))
plt.figure(figsize=(15,6))

plt.plot(y_test.values,
         label='Actual Sales')

plt.plot(predictions,
         label='Predicted Sales')

plt.title("Sales Forecasting")

plt.xlabel("Days")
plt.ylabel("Sales")

plt.legend()

plt.show()
future_data = daily_sales.copy()

future_predictions = []

for i in range(30):

    latest = future_data.iloc[-1]

    new_row = {
        'Year': latest['Year'],
        'Month': latest['Month'],
        'Day': latest['Day'],
        'DayOfWeek': latest['DayOfWeek'],
        'Lag_1': latest['Sales'],
        'Lag_7': future_data.iloc[-7]['Sales'],
        'Rolling_7': future_data['Sales'].tail(7).mean()
    }

    pred = model.predict(pd.DataFrame([new_row]))[0]

    future_predictions.append(pred)

    next_date = future_data.iloc[-1]['Date'] + pd.Timedelta(days=1)

    future_data.loc[len(future_data)] = [
        next_date,
        pred,
        next_date.year,
        next_date.month,
        next_date.day,
        next_date.dayofweek,
        latest['Sales'],
        future_data.iloc[-7]['Sales'],
        future_data['Sales'].tail(7).mean()
    ]

print(future_predictions)
# Sales by Category
category_sales = df.groupby('Category')['Sales'].sum()

category_sales.plot(kind='bar')
plt.title('Sales by Category')
plt.show()

# Sales by Region
region_sales = df.groupby('Region')['Sales'].sum()

region_sales.plot(kind='bar')
plt.title('Sales by Region')
plt.show()

# Monthly Sales Trend
monthly_sales = df.groupby(
    df['Order Date'].dt.to_period('M')
)['Sales'].sum()

monthly_sales.plot(figsize=(12,5))
plt.title('Monthly Sales Trend')
plt.show()

