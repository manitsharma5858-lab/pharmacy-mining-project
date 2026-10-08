import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# --- Page Configuration ---
st.set_page_config(page_title="Pharmacy Data Mining", page_icon="💊", layout="wide")

# --- 1. Generate 5000-Row Dataset ---
@st.cache_data
def load_data():
    np.random.seed(42)
    n_rows = 5000
    
    seasons = np.random.choice(['Summer', 'Winter', 'Monsoon'], n_rows)
    temperatures = []
    diseases = []
    medicines = []
    
    for s in seasons:
        if s == 'Winter':
            temp = np.random.randint(-5, 15)
            d = np.random.choice(['Flu', 'Common Cold'], p=[0.6, 0.4])
            m = 'Antipyretics (Fever)' if d == 'Flu' else 'Analgesics (Pain Relief)'
        elif s == 'Summer':
            temp = np.random.randint(25, 45)
            d = np.random.choice(['Allergies', 'Dehydration'], p=[0.7, 0.3])
            m = 'Antihistamines (Allergies)' if d == 'Allergies' else 'Electrolytes'
        else: # Monsoon
            temp = np.random.randint(20, 35)
            d = np.random.choice(['Malaria', 'Dengue', 'Viral Fever'], p=[0.3, 0.3, 0.4])
            m = 'Antibiotics' if d in ['Malaria', 'Dengue'] else 'Antipyretics (Fever)'
            
        temperatures.append(temp)
        diseases.append(d)
        medicines.append(m)

    df = pd.DataFrame({
        'Season': seasons,
        'Temperature_C': temperatures,
        'Prevailing_Disease': diseases,
        'Top_Selling_Medicine': medicines
    })
    return df

df = load_data()

# --- 2. Data Preprocessing & Model Training ---
# Encode categorical variables for the Machine Learning model
le_season = LabelEncoder()
le_disease = LabelEncoder()

df['Season_Encoded'] = le_season.fit_transform(df['Season'])
df['Disease_Encoded'] = le_disease.fit_transform(df['Prevailing_Disease'])

X = df[['Season_Encoded', 'Temperature_C', 'Disease_Encoded']]
y = df['Top_Selling_Medicine']

# Split data and train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=50, random_state=42)
model.fit(X_train, y_train)

# Calculate Accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# --- 3. Streamlit Frontend UI ---
st.title("💊 Pharmacy Sales & Inventory Predictor")
st.markdown("A mini data warehousing and mining project using a 5,000-record dataset to predict medicine demand.")

# Layout: Split into two columns
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📊 Dataset Overview")
    st.write(f"Total Records: **{len(df)} rows**")
    st.dataframe(df.sample(5, random_state=42), use_container_width=True) # Show a sample of 5 rows
    
    st.subheader("📈 Overall Sales Distribution")
    # Interactive Chart mapping medicine distribution
    medicine_counts = df['Top_Selling_Medicine'].value_counts()
    st.bar_chart(medicine_counts)

with col2:
    st.subheader("🤖 Machine Learning Prediction")
    st.info(f"**Model Accuracy:** {accuracy * 100:.2f}% using Random Forest Algorithm")
    
    st.markdown("### Predict Demand for Your Region")
    # User Inputs
    input_season = st.selectbox("Select Current Season", df['Season'].unique())
    
    # Filter diseases based on season just for logical UI flow, though model handles any combo
    filtered_diseases = df[df['Season'] == input_season]['Prevailing_Disease'].unique()
    input_disease = st.selectbox("Select Prevailing Local Disease", filtered_diseases)
    
    input_temp = st.slider("Average Temperature (°C)", min_value=-10, max_value=50, value=25)
    
    if st.button("Predict High-Demand Medicine", type="primary"):
        # Encode inputs
        season_encoded = le_season.transform([input_season])[0]
        disease_encoded = le_disease.transform([input_disease])[0]


        st.markdown("---")
st.header("💡 Advanced Mining: Association & Inventory Insights")

col3, col4 = st.columns(2)

with col3:
    st.subheader("🔗 Frequently Bought Together")
    st.write("Using association rule concepts, here are recommended cross-sells based on the primary medicine:")
    
    # Association Rule mapping (Market Basket Analysis)
    cross_sell_data = pd.DataFrame({
        'Primary Medicine': ['Antibiotics', 'Antipyretics (Fever)', 'Antihistamines (Allergies)', 'Analgesics (Pain Relief)', 'Electrolytes'],
        'Recommended Cross-Sell': ['Probiotics', 'Thermometers & Vitamin C', 'Nasal Spray', 'Hot Water Bags', 'Zinc Supplements']
    })
    st.table(cross_sell_data)

with col4:
    st.subheader("⚠️ Live Stock Alerts")
    st.write("Simulated inventory tracking based on current demand trends.")
    
    # Simulate current inventory levels
    inventory_df = pd.DataFrame({
        'Medicine Category': df['Top_Selling_Medicine'].unique(),
        'Current Stock (%)': np.random.randint(15, 95, size=len(df['Top_Selling_Medicine'].unique()))
    })
    
    # Function to highlight low stock in red
    def highlight_low_stock(val):
        color = '#ff4b4b' if val < 30 else '#00FF00'
        return f'color: {color}; font-weight: bold'
        
    # Apply the styling and display
    st.dataframe(inventory_df.style.map(highlight_low_stock, subset=['Current Stock (%)']), use_container_width=True)
        # Make Prediction
        prediction = model.predict([[season_encoded, input_temp, disease_encoded]])[0]
        
        st.success(f"**Predicted Top Seller:** {prediction}")
        st.write("Ensure your pharmacy has enough stock of this category for the upcoming weeks!")
