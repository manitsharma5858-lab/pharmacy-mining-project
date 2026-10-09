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
        # 1. Encode inputs into numbers
        season_encoded = le_season.transform([input_season])[0]
        disease_encoded = le_disease.transform([input_disease])[0]
        
        # 2. Package the input to match the training data exactly
        input_data = pd.DataFrame({
            'Season_Encoded': [season_encoded],
            'Temperature_C': [input_temp],
            'Disease_Encoded': [disease_encoded]
        })
        
        # 3. Make Prediction
        prediction = model.predict(input_data)[0]
        
        # 4. Display the results on screen
        st.success(f"**Predicted Top Seller:** {prediction}")
        st.write("Ensure your pharmacy has enough stock of this category for the upcoming weeks!")

st.markdown("---")
st.header("📂 Data Warehouse & Analytics Dashboard")
st.write("Deeper insights into the data mining model and current warehouse inventory.")

# Create a new two-column layout for the bottom section
col3, col4 = st.columns([1, 1])

with col3:
    st.subheader("🧠 Model Feature Importance")
    st.write("Displays which factor (Season, Temp, or Disease) the algorithm relies on most to make its prediction.")
    
    # Extract feature importance from the trained Random Forest model
    feature_importances = model.feature_importances_
    feature_names = ['Season', 'Temperature', 'Prevailing Disease']
    
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance Level': feature_importances
    }).set_index('Feature')
    
    # Render a bar chart for the importances
    st.bar_chart(importance_df)

with col4:
    st.subheader("📦 Warehouse Inventory Status")
    st.write("Simulated live tracking of current medicine stock levels.")
    
    # Create a simple, error-proof mock inventory dataframe
    inventory_data = pd.DataFrame({
        'Medicine Category': df['Top_Selling_Medicine'].unique(),
        'Current Stock (Units)': np.random.randint(100, 1500, size=len(df['Top_Selling_Medicine'].unique())),
        'Status': np.random.choice(['Optimal', 'Low Stock', 'Reorder Soon'], size=len(df['Top_Selling_Medicine'].unique()), p=[0.5, 0.3, 0.2])
    })
    
    # Display the dataframe cleanly
    st.dataframe(inventory_data, use_container_width=True)

st.markdown("---")
st.subheader("📋 Recent Warehouse Queries (Audit Log)")
st.write("A view into the latest historical records stored in the data warehouse.")
# Display the last 5 rows of the dataset to simulate a running log
st.dataframe(df.tail(5), use_container_width=True)

st.markdown("---")
st.header("⚖️ Algorithm Performance Comparison")
st.write("Comparing the accuracy of different Data Mining algorithms from your syllabus (Random Forest, J48/Decision Tree, and Naive Bayes) on our generated dataset.")

# Import the additional algorithms directly here to avoid touching the top of the file
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

# 1. Train J48 (Decision Tree equivalent in Scikit-Learn)
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
dt_accuracy = accuracy_score(y_test, dt_model.predict(X_test))

# 2. Train Naive Bayes
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)
nb_accuracy = accuracy_score(y_test, nb_model.predict(X_test))

# 3. Create a comparison dataframe
comparison_df = pd.DataFrame({
    'Algorithm': ['Random Forest', 'J48 (Decision Tree)', 'Naive Bayes'],
    'Accuracy (%)': [accuracy * 100, dt_accuracy * 100, nb_accuracy * 100]
})

# 4. Display as an interactive bar chart
st.subheader("📊 Accuracy Chart")
st.bar_chart(comparison_df.set_index('Algorithm'))

# 5. Show the exact percentage numbers in a clean table
st.dataframe(comparison_df.style.format({'Accuracy (%)': '{:.2f}%'}), use_container_width=True)

st.markdown("---")
st.header("🧩 Unsupervised Learning: K-Means Clustering")
st.write("Demonstrating the clustering rule process (Assignment 9) by grouping temperature and disease data without predefined labels.")

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# 1. Prepare data for clustering
X_cluster = df[['Temperature_C', 'Disease_Encoded']]

# 2. Apply K-Means Clustering (Aiming for 3 clusters representing typical environmental patterns)
kmeans = KMeans(n_clusters=3, random_state=42, n_init="auto")
df['Cluster'] = kmeans.fit_predict(X_cluster)

# 3. Calculate Clustering Validation Score
sil_score = silhouette_score(X_cluster, df['Cluster'])
st.info(f"**Silhouette Score:** {sil_score:.2f} (Score ranges from -1 to 1; closer to 1 means distinct, well-separated clusters)")

# 4. Visualize the Clusters
st.subheader("Scatter Plot: Temperature vs. Disease Groupings")
fig, ax = plt.subplots()
scatter = ax.scatter(df['Temperature_C'], df['Disease_Encoded'], c=df['Cluster'], cmap='viridis', alpha=0.6)
ax.set_xlabel("Temperature (°C)")
ax.set_ylabel("Disease (Encoded)")
ax.set_title("K-Means Groupings")
st.pyplot(fig)
