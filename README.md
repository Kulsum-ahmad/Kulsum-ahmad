-Real-estate-Project:
1. This project helps the user to get the predicted price of the desired house/flat/land so that the user can get an overall glance of the budget to purchase it.
2. A real estate price prediction application provides significant value across the entire real estate ecosystem, benefitting buyers, sellers, agents, investors and other stakeholders through
   accurate, transparent and efficient price estimation.
            
#File it contains:
a. realestate.py code.
b. dataset.csv
c. image.png
        
#Dataset details:
a. Kaggle datasets download -dshree1992/housedata

Explanation:-
Loading Data: The load_data() function reads the dataset (real_estate_data.csv) and caches it using Streamlit's @st.cache decorator for better performance.
Training the Model: The train_model() function splits the dataset into training and testing sets, trains a RandomForestRegressor model, and evaluates its performance using Mean Squared Error (MSE) and R-squared.
Prediction: The predict_price() function takes user inputs (location, size_in_sqft, bedrooms, amenities) and uses the trained model to predict the property price.Streamlit.
Interface: Displays a title for the app.Provides input fields (text_input, number_input, selectbox) for users to enter property details.Button triggers the price prediction and displays the result using st.success.

Running the Application:-
Save the above code in a Python file (e.g.,realestate.py).Place your data.csv file in the same directory.Open a terminal or command prompt and run the Streamlit app:
streamlit run realestate.py
