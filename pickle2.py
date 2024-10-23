import pickle
import logging
from sklearn.metrics import mean_squared_error, r2_score
import argparse

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_pickle_file(file_path):
    """Load a pickle file."""
    with open(file_path, "rb") as file:
        data = pickle.load(file)
        logging.info(f"Loaded {file_path} successfully.")
    return data

def evaluate_model(model, X_test, y_test):
    """Evaluate the model and print metrics."""
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Mean Squared Error: {mse:.2f}")
    print(f"R-squared: {r2:.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a saved house price prediction model.")
    parser.add_argument('--model', default='house_model.pkl', help='Path to the pickled model file.')
    parser.add_argument('--data', default='house_data.pkl', help='Path to the pickled test data file.')
    args = parser.parse_args()

    # Load the model and test data
    model = load_pickle_file(args.model)
    test_data = load_pickle_file(args.data)

    # Extract test data
    X_test = test_data['X_test']
    y_test = test_data['y_test']

    # Evaluate the model
    evaluate_model(model, X_test, y_test)
