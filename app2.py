import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib

data = {
    'horsepower': [100, 150, 120, 200, 250, 180],
    'weight': [2500, 3000, 2800, 3500, 4000, 3700],
    'price': [15000, 20000, 18000, 25000, 30000, 28000]
}

df = pd.DataFrame(data)

X = df[['horsepower', 'weight']]
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)

print(f"Training R2 Score: {train_score:.2f}")
print(f"Testing R2 Score: {test_score:.2f}")
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)

print(f"Training R2 Score: {train_score:.2f}")
print(f"Testing R2 Score: {test_score:.2f}")
joblib.dump(model, 'car_price_model.pkl')
