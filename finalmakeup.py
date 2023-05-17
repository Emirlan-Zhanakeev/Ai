import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load the dataset
data = pd.DataFrame({
    'Sales': [9.50, 11.22, 10.06, 7.40, 4.15],
    'CompPrice': [138, 111, 113, 117, 141],
    'Income': [73, 48, 35, 100, 64],
    'Advertising': [11, 16, 10, 4, 3],
    'Population': [276, 260, 269, 466, 340],
    'Price': [120, 83, 80, 97, 128],
    'ShelveLoc': ['Bad', 'Good', 'Medium', 'Medium', 'Bad'],
    'Age': [42, 65, 59, 55, 38],
    'Education': [17, 10, 12, 14, 13],
    'Urban': ['Yes', 'Yes', 'Yes', 'Yes', 'Yes'],
    'US': ['Yes', 'Yes', 'Yes', 'Yes', 'No']
})

# Convert categorical variables to numeric
data = pd.get_dummies(data, columns=['ShelveLoc', 'Urban', 'US'])

# Split the data into training and testing sets
X = data.drop('Sales', axis=1)
y = data['Sales']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the Random Forest model
rf = RandomForestRegressor()
rf.fit(X_train, y_train)

# Predict on the test set
y_pred = rf.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)