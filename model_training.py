import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def generate_house_data():
    """Generate a synthetic dataset for house prices."""
    np.random.seed(42)

    # Generate random data for house features
    house_id = range(1, 101)  # IDs for 100 houses
    sqft = np.random.randint(500, 3500, size=100)  # Square footage
    bedrooms = np.random.randint(1, 6, size=100)  # Number of bedrooms
    distance_to_city = np.round(np.random.uniform(1, 50, size=100), 2)  # Distance to city center in km

    # Create house prices influenced by other features
    base_price = sqft * 100 + bedrooms * 50000 - distance_to_city * 1000  # Basic price formula
    price = np.round(base_price + np.random.normal(0, 50000, size=100), 2)  # Add some noise

    # Create DataFrame
    df = pd.DataFrame({
        'house_id': house_id,
        'sqft': sqft,
        'bedrooms': bedrooms,
        'distance_to_city': distance_to_city,
        'price': price
    })

    logging.info("House data generated successfully.")
    return df

def train_and_save_model(df):
    """Train a Linear Regression model and save it along with test data."""
    df = df.drop(columns=["house_id"])  # Drop unnecessary column
    X = df.drop(columns=["price"])  # Features
    y = df["price"]  # Target variable

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a Linear Regression model
    model = LinearRegression()
    model.fit(X_train, y_train)
    logging.info("Model trained successfully.")

    # Save the model to a pickle file
    with open("house_model.pkl", "wb") as file:
        pickle.dump(model, file)
        logging.info("Model saved to house_model.pkl")

    # Save the test data to a pickle file
    test_data = {'X_test': X_test, 'y_test': y_test}
    with open("house_data.pkl", "wb") as file:
        pickle.dump(test_data, file)
        logging.info("Test data saved to house_data.pkl")

if __name__ == "__main__":
    data = generate_house_data()
    print(data.head())  # Optional: Display the first few rows
    train_and_save_model(data)
