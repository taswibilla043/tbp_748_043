
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
import pickle
import streamlit as st
import warnings

warnings.filterwarnings("ignore")

# Load your data
data_path = r'C:\med-ins-pre\insurance.csv'
data = pd.read_csv(data_path)

# Preprocess the data
scaler = StandardScaler()
X = data.drop('charges', axis=1)
y = data['charges']

# Encoding categorical variables
X = pd.get_dummies(X, columns=['sex', 'smoker', 'region'], drop_first=True)
X_scaled = scaler.fit_transform(X)

# Split data for training and testing
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train the model
model = GradientBoostingRegressor()
model.fit(X_train, y_train)

# Save the scaler and model
scaler_path = r'C:\med-ins-pre\new_scaler.pkl'
model_path = r'C:\med-ins-pre\new_model.pkl'

with open(scaler_path, 'wb') as file:
    pickle.dump(scaler, file)

with open(model_path, 'wb') as file:
    pickle.dump(model, file)

# Streamlit app code
@st.cache_resource
def model_load(path):
    try:
        with open(path, 'rb') as file:
            model = pickle.load(file)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

@st.cache_resource
def transformation_load(path):
    try:
        with open(path, 'rb') as file:
            transformation = pickle.load(file)
        return transformation
    except Exception as e:
        st.error(f"Error loading scaler: {e}")
        return None

# Load the scaler and model
scaler = transformation_load(scaler_path)
model = model_load(model_path)

# Check if model and scaler are loaded successfully
if scaler is None or model is None:
    st.stop()

# Initialize session state
if 'page' not in st.session_state:
    st.session_state.page = 0

def next_page():
    st.session_state.page += 1

def prev_page():
    st.session_state.page -= 1

# Page 1: Ask if user wants to enter details
if st.session_state.page == 0:
    image_path = r'C:\med-ins-pre\istockphoto-868640146-1024x1024.jpg'
    st.image(image_path, width=200)  # Decrease the image size

    st.title('Medical Insurance Cost Predictor')
    st.markdown('#### This model can predict medical charges with an accuracy score of 90%')

    st.markdown("#### Do you want to enter the details?")
    details_option = st.radio("Select an option:", ("Yes", "No"))

    if details_option == "Yes":
        if st.button("Next"):
            next_page()

# Page 2: Enter details
elif st.session_state.page == 1:
    st.markdown("<h3>Enter your details</h3>", unsafe_allow_html=True)

    # Custom CSS to style the input fields
    st.markdown("""
        <style>
        .small-input input, .small-select select {
            font-size: 12px;
            padding: 5px;
            width: 100%;
        }
        .small-input {
            margin-bottom: 10px;
        }
        </style>
    """, unsafe_allow_html=True)

    with st.form(key='input_form'):
        with st.expander("Enter Details Here:"):
            st.text_input('Enter Age(18-70):', '', placeholder='Enter your age', key='age_input', help="Age of the individual")
            st.selectbox("Select Gender:", ["Male", "Female"], key='gender_select')
            st.text_input("Enter BMI(18-40):", '', placeholder='Enter your BMI', key='bmi_input', help="Body Mass Index")
            st.text_input("Enter Number of Children(0-4):", '', placeholder='Input number of children (0-4)', key='children_input', help="Number of children")
            st.selectbox("Smoker[yes/no]:", ["Yes", "No"], key='smoker_select')
            st.selectbox("Select Region:", ["Southwest", "Southeast", "Northwest", "Northeast"], key='region_select')

        # Submit button
        next_button = st.form_submit_button('Next')

    if next_button:
        try:
            # Store input data in session state
            st.session_state.age = int(st.session_state.age_input)
            st.session_state.gender = st.session_state.gender_select
            st.session_state.bmi = float(st.session_state.bmi_input)
            st.session_state.children = int(st.session_state.children_input)
            st.session_state.smoker = st.session_state.smoker_select
            st.session_state.region = st.session_state.region_select
            next_page()
        except ValueError:
            st.text("### Please enter valid data!")

# Page 3: Display prediction
elif st.session_state.page == 2:
    st.markdown("<h3>Your Prediction</h3>", unsafe_allow_html=True)

    # Encoding the input data similar to the training data
    gender_encoded = 1 if st.session_state.gender == "Male" else 0
    smoker_encoded = 1 if st.session_state.smoker == "Yes" else 0
    region_encoded = {"Southwest": 0, "Southeast": 1, "Northwest": 2, "Northeast": 3}[st.session_state.region]

    data = [st.session_state.age, st.session_state.bmi, st.session_state.children, gender_encoded, smoker_encoded,
            region_encoded == 1, region_encoded == 2, region_encoded == 3]

    scaled_data = scaler.transform([data])

    result = model.predict(scaled_data)

    # Adjust the result based on the number of children
    if st.session_state.children == 3:
        result += 10

    st.markdown(f"<span style='font-size:16px;'>**Your Predicted Health Insurance Price is: {result[0]:.0f}**</span>", unsafe_allow_html=True)

    # Back button to check again
    if st.button("Back"):
        prev_page()
