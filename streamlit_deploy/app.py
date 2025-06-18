"""
Streamlit app for house price prediction.
"""
import os
import sys
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.preprocessing import create_preprocessing_pipeline, FeatureEngineer
from src.utils import load_model

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
        color: #222 !important;
        padding: 2rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        font-size: 1.5rem;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# Get the absolute paths
model_path = os.path.join(project_root, 'output', 'model.joblib')
pipeline_path = os.path.join(project_root, 'output', 'preprocessing_pipeline.joblib')

# Load the model and pipeline
try:
    model = load_model(model_path)
    preprocessor = joblib.load(pipeline_path)
    st.success("Model and preprocessing pipeline loaded successfully!")
except Exception as e:
    st.error(f"Error loading model or pipeline: {str(e)}")
    st.error("Could not load the model or pipeline. Please ensure the files exist.")
    st.stop()

def create_input_data():
    """Create input data from user inputs."""
    # Create a dictionary with all required columns and their default values
    input_data = {
        'MSSubClass': 60,  # Default to 1-STORY 1946 & NEWER ALL STYLES
        'MSZoning': 'RL',  # Default to Residential Low Density
        'LotFrontage': 0,
        'LotArea': 0,
        'Street': 'Pave',  # Default to Paved
        'Alley': 'NA',  # Default to No alley access
        'LotShape': 'Reg',  # Default to Regular
        'LandContour': 'Lvl',  # Default to Level
        'Utilities': 'AllPub',  # Default to All public Utilities
        'LotConfig': 'Inside',  # Default to Inside lot
        'LandSlope': 'Gtl',  # Default to Gentle slope
        'Neighborhood': 'NAmes',  # Default to North Ames
        'Condition1': 'Norm',  # Default to Normal
        'Condition2': 'Norm',  # Default to Normal
        'BldgType': '1Fam',  # Default to Single-family Detached
        'HouseStyle': '1Story',  # Default to One story
        'OverallQual': 5,  # Default to Average
        'OverallCond': 5,  # Default to Average
        'YearBuilt': 2000,  # Default to recent year
        'YearRemodAdd': 2000,  # Default to recent year
        'RoofStyle': 'Gable',  # Default to Gable
        'RoofMatl': 'CompShg',  # Default to Standard (Composite) Shingle
        'Exterior1st': 'VinylSd',  # Default to Vinyl Siding
        'Exterior2nd': 'VinylSd',  # Default to Vinyl Siding
        'MasVnrType': 'None',  # Default to None
        'MasVnrArea': 0,
        'ExterQual': 'TA',  # Default to Typical/Average
        'ExterCond': 'TA',  # Default to Typical/Average
        'Foundation': 'PConc',  # Default to Poured Concrete
        'BsmtQual': 'TA',  # Default to Typical/Average
        'BsmtCond': 'TA',  # Default to Typical/Average
        'BsmtExposure': 'No',  # Default to No Exposure
        'BsmtFinType1': 'Unf',  # Default to Unfinished
        'BsmtFinSF1': 0,
        'BsmtFinType2': 'Unf',  # Default to Unfinished
        'BsmtFinSF2': 0,
        'BsmtUnfSF': 0,
        'TotalBsmtSF': 0,
        'Heating': 'GasA',  # Default to Gas forced warm air furnace
        'HeatingQC': 'TA',  # Default to Typical/Average
        'CentralAir': 'Y',  # Default to Yes
        'Electrical': 'SBrkr',  # Default to Standard Circuit Breakers & Romex
        '1stFlrSF': 0,
        '2ndFlrSF': 0,
        'LowQualFinSF': 0,  # Added missing column
        'GrLivArea': 0,
        'BsmtFullBath': 0,  # Added missing column
        'BsmtHalfBath': 0,  # Added missing column
        'FullBath': 0,
        'HalfBath': 0,
        'BedroomAbvGr': 0,
        'KitchenAbvGr': 0,
        'KitchenQual': 'TA',  # Default to Typical/Average
        'TotRmsAbvGrd': 0,
        'Functional': 'Typ',  # Default to Typical Functionality
        'Fireplaces': 0,
        'FireplaceQu': 'NA',  # Default to No Fireplace
        'GarageType': 'Attchd',  # Default to Attached to home
        'GarageYrBlt': 2000,  # Added missing column with default value
        'GarageFinish': 'Unf',  # Default to Unfinished
        'GarageCars': 0,
        'GarageArea': 0,
        'GarageQual': 'TA',  # Default to Typical/Average
        'GarageCond': 'TA',  # Default to Typical/Average
        'PavedDrive': 'Y',  # Default to Paved
        'WoodDeckSF': 0,
        'OpenPorchSF': 0,
        'EnclosedPorch': 0,
        '3SsnPorch': 0,
        'ScreenPorch': 0,
        'PoolArea': 0,
        'PoolQC': 'NA',  # Default to No Pool
        'Fence': 'NA',  # Default to No Fence
        'MiscFeature': 'NA',  # Default to None
        'MiscVal': 0,
        'MoSold': 1,  # Default to January
        'YrSold': 2024,  # Default to current year
        'SaleType': 'WD',  # Default to Warranty Deed - Conventional
        'SaleCondition': 'Normal'  # Default to Normal Sale
    }
    
    # Update with user inputs
    input_data.update({
        'MSSubClass': st.number_input('MSSubClass', min_value=20, max_value=190, value=60),
        'MSZoning': st.selectbox('MSZoning', ['RL', 'RM', 'C (all)', 'FV', 'RH']),
        'LotFrontage': st.number_input('LotFrontage', min_value=0, value=0),
        'LotArea': st.number_input('LotArea', min_value=0, value=0),
        'Street': st.selectbox('Street', ['Pave', 'Grvl']),
        'Alley': st.selectbox('Alley', ['NA', 'Pave', 'Grvl']),
        'LotShape': st.selectbox('LotShape', ['Reg', 'IR1', 'IR2', 'IR3']),
        'LandContour': st.selectbox('LandContour', ['Lvl', 'Bnk', 'HLS', 'Low']),
        'Utilities': st.selectbox('Utilities', ['AllPub', 'NoSewr', 'NoSeWa', 'ELO']),
        'LotConfig': st.selectbox('LotConfig', ['Inside', 'Corner', 'CulDSac', 'FR2', 'FR3']),
        'LandSlope': st.selectbox('LandSlope', ['Gtl', 'Mod', 'Sev']),
        'Neighborhood': st.selectbox('Neighborhood', [
            'NAmes', 'CollgCr', 'OldTown', 'Edwards', 'Somerst', 'Gilbert', 'NridgHt',
            'Sawyer', 'NWAmes', 'SawyerW', 'BrkSide', 'Crawfor', 'Mitchel', 'Timber',
            'NoRidge', 'StoneBr', 'SWISU', 'ClearCr', 'MeadowV', 'Blmngtn', 'BrDale',
            'Veenker', 'NPkVill', 'Blueste'
        ]),
        'Condition1': st.selectbox('Condition1', ['Norm', 'Feedr', 'PosN', 'Artery', 'RRAe', 'RRNn', 'RRAn', 'PosA', 'RRNe']),
        'Condition2': st.selectbox('Condition2', ['Norm', 'Feedr', 'PosN', 'Artery', 'RRAe', 'RRNn', 'RRAn', 'PosA', 'RRNe']),
        'BldgType': st.selectbox('BldgType', ['1Fam', '2FmCon', 'Duplex', 'TwnhsE', 'TwnhsI']),
        'HouseStyle': st.selectbox('HouseStyle', ['1Story', '1.5Fin', '1.5Unf', '2Story', '2.5Fin', '2.5Unf', 'SFoyer', 'SLvl']),
        'OverallQual': st.number_input('OverallQual', min_value=1, max_value=10, value=5),
        'OverallCond': st.number_input('OverallCond', min_value=1, max_value=10, value=5),
        'YearBuilt': st.number_input('YearBuilt', min_value=1872, max_value=2024, value=2000),
        'YearRemodAdd': st.number_input('YearRemodAdd', min_value=1950, max_value=2024, value=2000),
        'RoofStyle': st.selectbox('RoofStyle', ['Gable', 'Hip', 'Gambrel', 'Mansard', 'Flat', 'Shed']),
        'RoofMatl': st.selectbox('RoofMatl', ['CompShg', 'WdShngl', 'Metal', 'WdShake', 'Membran', 'Tar&Grv', 'Roll', 'ClyTile']),
        'Exterior1st': st.selectbox('Exterior1st', ['VinylSd', 'MetalSd', 'Wd Sdng', 'HdBoard', 'BrkFace', 'WdShing', 'CemntBd', 'Plywood', 'AsbShng', 'Stucco', 'BrkComm', 'AsphShn', 'Stone', 'ImStucc', 'CBlock']),
        'Exterior2nd': st.selectbox('Exterior2nd', ['VinylSd', 'MetalSd', 'Wd Sdng', 'HdBoard', 'BrkFace', 'WdShing', 'CemntBd', 'Plywood', 'AsbShng', 'Stucco', 'BrkComm', 'AsphShn', 'Stone', 'ImStucc', 'CBlock']),
        'MasVnrType': st.selectbox('MasVnrType', ['None', 'BrkFace', 'Stone', 'BrkCmn']),
        'MasVnrArea': st.number_input('MasVnrArea', min_value=0, value=0),
        'ExterQual': st.selectbox('ExterQual', ['Ex', 'Gd', 'TA', 'Fa', 'Po']),
        'ExterCond': st.selectbox('ExterCond', ['Ex', 'Gd', 'TA', 'Fa', 'Po']),
        'Foundation': st.selectbox('Foundation', ['PConc', 'CBlock', 'BrkTil', 'Wood', 'Slab', 'Stone']),
        'BsmtQual': st.selectbox('BsmtQual', ['Ex', 'Gd', 'TA', 'Fa', 'Po', 'NA']),
        'BsmtCond': st.selectbox('BsmtCond', ['Ex', 'Gd', 'TA', 'Fa', 'Po', 'NA']),
        'BsmtExposure': st.selectbox('BsmtExposure', ['Gd', 'Av', 'Mn', 'No', 'NA']),
        'BsmtFinType1': st.selectbox('BsmtFinType1', ['GLQ', 'ALQ', 'BLQ', 'Rec', 'LwQ', 'Unf', 'NA']),
        'BsmtFinSF1': st.number_input('BsmtFinSF1', min_value=0, value=0),
        'BsmtFinType2': st.selectbox('BsmtFinType2', ['GLQ', 'ALQ', 'BLQ', 'Rec', 'LwQ', 'Unf', 'NA']),
        'BsmtFinSF2': st.number_input('BsmtFinSF2', min_value=0, value=0),
        'BsmtUnfSF': st.number_input('BsmtUnfSF', min_value=0, value=0),
        'TotalBsmtSF': st.number_input('TotalBsmtSF', min_value=0, value=0),
        'Heating': st.selectbox('Heating', ['GasA', 'GasW', 'Grav', 'Wall', 'OthW', 'Floor']),
        'HeatingQC': st.selectbox('HeatingQC', ['Ex', 'Gd', 'TA', 'Fa', 'Po']),
        'CentralAir': st.selectbox('CentralAir', ['Y', 'N']),
        'Electrical': st.selectbox('Electrical', ['SBrkr', 'FuseA', 'FuseF', 'FuseP', 'Mix']),
        '1stFlrSF': st.number_input('1stFlrSF', min_value=0, value=0),
        '2ndFlrSF': st.number_input('2ndFlrSF', min_value=0, value=0),
        'LowQualFinSF': st.number_input('LowQualFinSF', min_value=0, value=0),
        'GrLivArea': st.number_input('GrLivArea', min_value=0, value=0),
        'BsmtFullBath': st.number_input('BsmtFullBath', min_value=0, value=0),
        'BsmtHalfBath': st.number_input('BsmtHalfBath', min_value=0, value=0),
        'FullBath': st.number_input('FullBath', min_value=0, value=0),
        'HalfBath': st.number_input('HalfBath', min_value=0, value=0),
        'BedroomAbvGr': st.number_input('BedroomAbvGr', min_value=0, value=0),
        'KitchenAbvGr': st.number_input('KitchenAbvGr', min_value=0, value=0),
        'KitchenQual': st.selectbox('KitchenQual', ['Ex', 'Gd', 'TA', 'Fa', 'Po']),
        'TotRmsAbvGrd': st.number_input('TotRmsAbvGrd', min_value=0, value=0),
        'Functional': st.selectbox('Functional', ['Typ', 'Min1', 'Min2', 'Mod', 'Maj1', 'Maj2', 'Sev', 'Sal']),
        'Fireplaces': st.number_input('Fireplaces', min_value=0, value=0),
        'FireplaceQu': st.selectbox('FireplaceQu', ['Ex', 'Gd', 'TA', 'Fa', 'Po', 'NA']),
        'GarageType': st.selectbox('GarageType', ['Attchd', 'Detchd', 'BuiltIn', 'CarPort', 'Basment', '2Types', 'NA']),
        'GarageYrBlt': st.number_input('GarageYrBlt', min_value=1900, max_value=2024, value=2000),
        'GarageFinish': st.selectbox('GarageFinish', ['Fin', 'RFn', 'Unf', 'NA']),
        'GarageCars': st.number_input('GarageCars', min_value=0, value=0),
        'GarageArea': st.number_input('GarageArea', min_value=0, value=0),
        'GarageQual': st.selectbox('GarageQual', ['Ex', 'Gd', 'TA', 'Fa', 'Po', 'NA']),
        'GarageCond': st.selectbox('GarageCond', ['Ex', 'Gd', 'TA', 'Fa', 'Po', 'NA']),
        'PavedDrive': st.selectbox('PavedDrive', ['Y', 'P', 'N']),
        'WoodDeckSF': st.number_input('WoodDeckSF', min_value=0, value=0),
        'OpenPorchSF': st.number_input('OpenPorchSF', min_value=0, value=0),
        'EnclosedPorch': st.number_input('EnclosedPorch', min_value=0, value=0),
        '3SsnPorch': st.number_input('3SsnPorch', min_value=0, value=0),
        'ScreenPorch': st.number_input('ScreenPorch', min_value=0, value=0),
        'PoolArea': st.number_input('PoolArea', min_value=0, value=0),
        'PoolQC': st.selectbox('PoolQC', ['Ex', 'Gd', 'TA', 'Fa', 'NA']),
        'Fence': st.selectbox('Fence', ['GdPrv', 'MnPrv', 'GdWo', 'MnWw', 'NA']),
        'MiscFeature': st.selectbox('MiscFeature', ['Elev', 'Gar2', 'Othr', 'Shed', 'TenC', 'NA']),
        'MiscVal': st.number_input('MiscVal', min_value=0, value=0),
        'MoSold': st.number_input('MoSold', min_value=1, max_value=12, value=1),
        'YrSold': st.number_input('YrSold', min_value=2006, max_value=2024, value=2024),
        'SaleType': st.selectbox('SaleType', ['WD', 'CWD', 'VWD', 'New', 'COD', 'Con', 'ConLw', 'ConLI', 'ConLD', 'Oth']),
        'SaleCondition': st.selectbox('SaleCondition', ['Normal', 'Abnorml', 'AdjLand', 'Alloca', 'Family', 'Partial'])
    })
    
    # Convert to DataFrame
    input_df = pd.DataFrame([input_data])
    
    # Apply feature engineering
    feature_engineer = FeatureEngineer()
    input_df = feature_engineer.transform(input_df)
    
    return input_df

def main():
    """Main function to run the Streamlit app."""
    st.title("🏠 House Price Prediction")
    st.write("""
    This app predicts house prices based on various features. Fill in the details below to get a price prediction.
    """)
    
    # Create input data
    input_df = create_input_data()
    
    # Make prediction
    if st.button("Predict Price"):
        try:
            # Transform input data using the preprocessing pipeline
            input_processed = preprocessor.transform(input_df)
            
            # Make prediction
            prediction = model.predict(input_processed)
            
            # Convert prediction back to original scale
            prediction = np.expm1(prediction)[0]
            
            # Display prediction
            st.markdown("### Prediction")
            st.markdown(f"<div class='prediction-box'>Estimated Price: ${prediction:,.2f}</div>", unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
            st.error("Please check your input values and try again.")

if __name__ == "__main__":
    main() 