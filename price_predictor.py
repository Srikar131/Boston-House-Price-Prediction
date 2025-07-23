# Step 2: Load and Explore the Dataset

# [cite_start]Import necessary libraries [cite: 7, 8]
import pandas as pd

# URL for the Boston Housing dataset
data_url = "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv"

# Load the dataset into a pandas DataFrame
# [cite_start]This is the modern way to load this data, as load_boston() is deprecated [cite: 19]
df = pd.read_csv(data_url)

# [cite_start]The original project used 'PRICE' as the target column name[cite: 22].
# This CSV uses 'medv' (median value), so we'll rename it for consistency.
df.rename(columns={'medv': 'PRICE'}, inplace=True)

# [cite_start]Display the first few rows of the dataset to verify it's loaded correctly [cite: 23]
print("Dataset loaded successfully. Here are the first 5 rows:")
print(df.head())

# --- Step 3: Prepare Data for Regression ---

from sklearn.model_selection import train_test_split

# Select the feature and target variables
X = df[['rm']]
y = df['PRICE']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Print the sizes of the training and testing sets to confirm the split
print("\nData prepared and split successfully.")
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)

# --- Step 4: Train the Linear Regression Model ---

from sklearn.linear_model import LinearRegression

# Initialize the model
model = LinearRegression() # [cite: 39, 64]

# Train the model using the training data
model.fit(X_train, y_train) # [cite: 39, 65]

# Print the learned parameters (the equation of the line)
print("\nModel training complete.")
print(f"Coefficient (slope): {model.coef_[0]}") # [cite: 41]
print(f"Intercept: {model.intercept_}") # [cite: 42]

# --- Step 5: Evaluate the Model ---

from sklearn.metrics import mean_squared_error, r2_score

# Make predictions on the test data
y_pred = model.predict(X_test)

# Calculate evaluation metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel evaluation results:")
print(f"Mean Squared Error (MSE): {mse}")
print(f"R-squared (R2) Score: {r2}")

# --- Step 6: Visualize the Predictions ---

import matplotlib.pyplot as plt

# Create a scatter plot of the test data (actual prices)
plt.scatter(X_test, y_test, color='blue', label='Actual Prices')

# Plot the regression line (our model's predictions)
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Predicted Prices')

# Add labels and a title for clarity
plt.title('Linear Regression: RM vs PRICE')
plt.xlabel('Average Number of Rooms (rm)')
plt.ylabel('Median House Price ($1000s)')
plt.legend()

plt.savefig('result_plot.png')
# Display the plot
plt.show()