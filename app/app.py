"""
Streamlit app for house price prediction.
"""
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import sys

# Add src directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preprocessing import create_preprocessing_pipeline

# Set page config
st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
    }
    .prediction-box {
        background-color: #f0f2f6;
        padding: 2rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

def load_model():
    """Load the trained model."""
    try:
        model = joblib.load('../output/model.joblib')
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def create_input_data():
    """Create input data from user inputs."""
    # Create a dictionary to store all inputs
    data = {}
    
    # Location
    st.subheader("Location")
    col1, col2 = st.columns(2)
    with col1:
        data['MSZoning'] = st.selectbox(
            "Zoning Classification",
            ['RL', 'RM', 'C (all)', 'FV', 'RH']
        )
    with col2:
        data['Neighborhood'] = st.selectbox(
            "Neighborhood",
            ['NAmes', 'CollgCr', 'OldTown', 'Edwards', 'Somerst', 'Gilbert', 'NridgHt']
        )
    
    # House Characteristics
    st.subheader("House Characteristics")
    col1, col2, col3 = st.columns(3)
    with col1:
        data['YearBuilt'] = st.number_input("Year Built", 1800, 2024, 2000)
        data['TotalBsmtSF'] = st.number_input("Basement Area (sq ft)", 0, 5000, 1000)
    with col2:
        data['GrLivArea'] = st.number_input("Living Area (sq ft)", 500, 5000, 1500)
        data['FullBath'] = st.number_input("Full Bathrooms", 0, 5, 2)
    with col3:
        data['BedroomAbvGr'] = st.number_input("Bedrooms", 0, 10, 3)
        data['GarageCars'] = st.number_input("Garage Cars", 0, 5, 2)
    
    # Additional Features
    st.subheader("Additional Features")
    col1, col2 = st.columns(2)
    with col1:
        data['KitchenQual'] = st.selectbox(
            "Kitchen Quality",
            ['Ex', 'Gd', 'TA', 'Fa']
        )
        data['Fireplaces'] = st.number_input("Number of Fireplaces", 0, 5, 0)
    with col2:
        data['GarageType'] = st.selectbox(
            "Garage Type",
            ['Attchd', 'Detchd', 'BuiltIn', 'CarPort', 'Basment', '2Types']
        )
        data['PoolArea'] = st.number_input("Pool Area (sq ft)", 0, 1000, 0)
    
    # Convert to DataFrame
    df = pd.DataFrame([data])
    
    # Add required columns with default values
    required_columns = [
        'Street', 'LotShape', 'LandContour', 'Utilities', 'LotConfig',
        'LandSlope', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle',
        'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType',
        'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond',
        'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Heating', 'HeatingQC',
        'CentralAir', 'Electrical', 'Functional', 'FireplaceQu', 'GarageFinish',
        'GarageQual', 'GarageCond', 'PavedDrive', 'SaleType', 'SaleCondition'
    ]
    
    for col in required_columns:
        if col not in df.columns:
            df[col] = 'TA'  # Default value
    
    return df

def main():
    """Main function to run the Streamlit app."""
    st.title("🏠 House Price Prediction")
    st.write("""
    This app predicts house prices based on various features.
    Fill in the details below to get a price prediction.
    """)
    
    # Load model
    model = load_model()
    if model is None:
        st.error("Could not load the model. Please ensure the model file exists.")
        return
    
    # Create input form
    with st.form("prediction_form"):
        input_data = create_input_data()
        
        # Submit button
        submitted = st.form_submit_button("Predict Price")
        
        if submitted:
            try:
                # Create preprocessing pipeline
                pipeline = create_preprocessing_pipeline()
                
                # Transform input data
                processed_data = pipeline.transform(input_data)
                
                # Make prediction
                prediction = model.predict(processed_data)
                
                # Convert log prediction back to original scale
                price = np.expm1(prediction[0])
                
                # Display prediction
                st.markdown("""
                    <div class="prediction-box">
                        <h2>Predicted House Price</h2>
                        <h1>${:,.2f}</h1>
                    </div>
                """.format(price), unsafe_allow_html=True)
                
                # Display feature importance if available
                if hasattr(model, 'feature_importances_'):
                    st.subheader("Top 5 Important Features")
                    importances = model.feature_importances_
                    indices = np.argsort(importances)[::-1]
                    
                    # Get feature names
                    feature_names = (
                        pipeline.named_steps['preprocessor']
                        .transformers_[0][2] +  # numeric features
                        pipeline.named_steps['preprocessor']
                        .transformers_[1][2]    # categorical features
                    )
                    
                    # Display top 5 features
                    for i in range(5):
                        st.write(f"{i+1}. {feature_names[indices[i]]}: {importances[indices[i]]:.4f}")
                
            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")

if __name__ == "__main__":
    main() 