## **House Price Prediction Project**  
This repository contains a complete workflow for training and evaluating a house price prediction model using synthetic data. It demonstrates the use of machine learning concepts such as data generation, Linear Regression model training, and performance evaluation. The model and test data are saved as .pkl files to enable easy reuse.

---

## **Repository Structure**  
- **`README.md`**: Project documentation.  
- **`house_data.pkl`**: Pickled file containing the test dataset for evaluation.  
- **`house_model.pkl`**: Pickled file containing the trained model.  
- **`model_training.py`**: Script to generate data, train the model, and save the model and test data.  
- **`model_prediction.py`**: Script to load the saved model and test data, make predictions, and evaluate performance.  
- **`requirements.txt`**: Dependencies required to run the project.

---

## **Setup Instructions**  

### 1. **Clone the Repository**  
```bash
git clone <https://github.com/ogshivam/House_Price_Prediction_pkl>
cd <Desktop/Gamy>
```

### 2. **Install Dependencies**  
Make sure you have Python installed (preferably using Miniconda). Install required packages:  
```bash
pip install -r requirements.txt
```

### 3. **Run the Model Training Script**  
This script generates a synthetic dataset, trains the model, and saves both the trained model and test data.  
```bash
python model_training.py
```

**Sample Output:**  
```
INFO - House data generated successfully.
INFO - Model trained successfully.
INFO - Model saved to house_model.pkl
INFO - Test data saved to house_data.pkl
```

### 4. **Run the Model Prediction and Evaluation Script**  
This script loads the trained model and test data, makes predictions, and evaluates performance using **Mean Squared Error (MSE)** and **R-squared (R²)** metrics.  
```bash
python model_prediction.py
```

**Sample Output:**  
```
INFO - Loaded house_model.pkl successfully.
INFO - Loaded house_data.pkl successfully.
Mean Squared Error: 2175973195.37
R-squared: 0.82
```

---

## **Files Overview**  

### 1. **`model_training.py`**  
- **Generates synthetic house data** including `sqft`, `bedrooms`, `distance_to_city`, and `price`.  
- **Trains a Linear Regression model** on this data.  
- **Saves the trained model** to `house_model.pkl`.  
- **Saves test data** to `house_data.pkl`.

**Example Generated Data:**  
```
   house_id  sqft  bedrooms  distance_to_city      price
0         1  1360         4             32.08  256444.05
1         2  1794         3             34.35  265205.13
2         3  1630         3             27.02  280419.80
```

### 2. **`model_prediction.py`**  
- **Loads the trained model** and test data from pickle files.  
- **Evaluates the model** using Mean Squared Error (MSE) and R² metrics to measure performance.

---

## **Dependencies**  
Ensure the following libraries are installed:
- `pandas`
- `numpy`
- `scikit-learn`
- `pickle`
- `argparse`

Install all dependencies with:  
```bash
pip install -r requirements.txt
```

---

## **Usage Example**  

1. **Training the model:**
   ```bash
   python model_training.py
   ```
2. **Evaluating the model:**
   ```bash
   python model_prediction.py --model house_model.pkl --data house_data.pkl
   ```

---

## **Project Logs**  
The project uses logging to track key events. Example logs include:
```
2024-10-23 17:27:35,532 - INFO - House data generated successfully.
2024-10-23 17:27:35,580 - INFO - Model trained successfully.
2024-10-23 17:30:54,046 - INFO - Loaded house_model.pkl successfully.
2024-10-23 17:30:54,048 - INFO - Loaded house_data.pkl successfully.
```

---

## **Contributing**  
Feel free to fork this repository and submit pull requests for improvements or bug fixes.

---

## **License**  
This project is licensed under the MIT License.

---

This README provides all the necessary instructions for setting up, training, and evaluating the house price prediction model using this codebase.
