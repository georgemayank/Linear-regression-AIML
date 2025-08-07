import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv("Housing.csv")          #This function reads the data from the uploaded File
df = pd.get_dummies(df, drop_first=True)

print("All rows of the dataset:")        # This prints the  All rows of the dataset
print(df.head())                         #This print head of the dataset
print("\nColumns in the dataset:")       #This prints the columns in the dataset to (\n) New line
print(df.columns) 

X = df[['area', 'bedrooms', 'bathrooms', 'stories', 'mainroad_yes',  
        'guestroom_yes', 'basement_yes', 'hotwaterheating_yes', 'airconditioning_yes',
        'parking', 'prefarea_yes', 'furnishingstatus_semi-furnished', 'furnishingstatus_unfurnished']]   #These are the features of the dataset Extracted from the top-written in coloured code
y = df['price']                                      # This is the target that user wants to predict

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) #Here starts the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)                                                               #.fit() is used to train the model on the data set

y_pred = model.predict(X_test)          #.predict() is used to make predictions on the trained model # What it Learns Predicts on that data !

#Evaluation metrics
mae = mean_absolute_error(y_test, y_pred) # Avg. of all the errors
mse = mean_squared_error(y_test, y_pred) # Square of the errors
r2 = r2_score(y_test, y_pred)            # How much the data is explained by the model

print("\nModel Evaluation:")
print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("R² Score:", r2)

# ✅ Graph for Actual vs Predicted prices
plt.scatter(y_test, y_pred, color='purple')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', lw=2)  # Diagonal reference line
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Actual vs Predicted House Prices')
plt.grid(True)
plt.show()
