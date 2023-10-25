import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

p=pd.read_csv("C:\\Users\\IT\\Downloads\\archive (1)\\result.csv") #load the dataset
print(p.head())

# Handling missing values
p.dropna(inplace=True)

# Dealing with duplicates
p.drop_duplicates(inplace=True)

# Data transformation


p['UnEmployeeRate'] = pd.to_numeric(p['UnEmployeeRate'])
p['movieScore'] = pd.to_numeric(p['movieScore'])



# Data filtering (if needed)
p= p[['year','UnEmployeeRate','movieScore']]

print(p)
X = p[['UnEmployeeRate', 'movieScore']]
y = p['year']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
