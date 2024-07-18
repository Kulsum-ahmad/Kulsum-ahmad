# Import necessary libraries
import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
@st.cache_data  # Cache data for better performance
def load_data():
    data = pd.read_csv('data.csv')  # Replace with your dataset path
    return data

# Function to train model
def train_model(data):
    # Separate features and target variable
    X = data[['sqft_living', 'bedrooms', 'bathrooms', 'floors']]
    y = data['price']
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    st.write('Mean Squared Error:', mse)
    st.write('R-squared:', r2)
    
    return model

# Function to predict price
def predict_price(model, sqft_living, bedrooms, bathrooms, floors):
    input_data = pd.DataFrame({
        'sqft_living': [sqft_living],
        'bedrooms': [bedrooms],
        'bathrooms':[bathrooms],
        'floors': [floors]
    })
    prediction = model.predict(input_data)
    return prediction[0]

# Streamlit interface
def main():
    st.title('Real Estate Price Prediction')
    
    # Load data
    data = load_data()
    
    # Train model
    model = train_model(data)
    
    # Input fields
    sqft_living = st.number_input('Size (sqft)', min_value=1.0, format="%f")
    bedrooms = st.number_input('Bedrooms', min_value=1.0, format="%f")
    bathrooms = st.number_input('Bathrooms', min_value=1.0, format="%f")
    floors = st.number_input('Floors', min_value=1.0, format="%f")
    
    # Prediction and result
    if st.button('Predict Price'):
        price_prediction = predict_price(model, sqft_living, bedrooms, bathrooms, floors)
        st.success(f'Predicted Price: ${price_prediction:.2f}')

if __name__ == '__main__':
    main()