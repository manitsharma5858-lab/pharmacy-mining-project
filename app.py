import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, silhouette_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# --- Page Configuration ---
st.set_page_config(page_title="Pharmacy Data Mining", page_icon="💊", layout="wide")

# --- 1. Generate 5000-Row Dataset (With Realistic Noise) ---
@st.cache_data
def load_data():
    np.random.seed(42)
    n_rows = 5000
    
    # Added Spring and Autumn for complexity
    seasons = np.random.choice(['Summer', 'Winter', 'Monsoon', 'Spring', 'Autumn'], n_rows)
    temperatures = []
    diseases = []
    medicines = []
    
    all_meds = ['Antipyretics (Fever)', 'Analgesics (Pain Relief)', 'Antihistamines (Allergies)', 
                'Electrolytes', 'Antibiotics', 'Cough Syrup', 'Vitamin C']
    
    for s in seasons:
        if s == 'Winter':
            temp = np.random.randint(-5, 15)
            d = np.random.choice(['Flu', 'Common Cold', 'Pneumonia'], p=[0.5, 0.4, 0.1])
            m = np.random.choice(['Antipyretics (Fever)', 'Cough Syrup', 'Antibiotics'], p=[0.5, 0.4, 0.1])
        elif s == 'Summer':
            temp = np.random.randint(25, 45)
            d = np.random.choice(['Allergies', 'Dehydration', 'Heat Stroke'], p=[0.4, 0.5, 0.1])
            m = np.random.choice(['Antihistamines (Allergies)', 'Electrolytes', 'Analgesics (Pain Relief)'], p=[0.4, 0.5, 0.1])
        elif s == 'Monsoon':
            temp = np.random.randint(20, 35)
            d = np.random.choice(['Malaria', 'Dengue', 'Viral Fever', 'Typhoid'], p=[0.2, 0.2, 0.5, 0.1])
            m = np.random.choice(['Antibiotics', 'Antipyretics (Fever)', 'Analgesics (Pain Relief)'], p=[0.3, 0.5, 0.2])
        elif s == 'Spring':
            temp = np.random.randint(15, 25)
            d = np.random.choice(['Allergies', 'Asthma', 'Common Cold'], p=[0.6, 0.2, 0.2])
            m = np.random.choice(['Antihistamines (Allergies)', 'Cough Syrup', 'Vitamin C'], p=[0.6, 0.2, 0.2])
        else: # Autumn
            temp = np.random.randint(10, 20)
            d = np.random.choice(['Flu', 'Viral Fever', 'Allergies'], p=[0.4, 0.4, 0.2])
            m = np.random.choice(['Antipyretics (Fever)', 'Vitamin C', 'Antihistamines (Allergies)'], p=[0.5, 0.3, 0.2])
            
        # Introduce 15% random noise (simulating unpredictable real-world customers) to drop accuracy
        if np.random.rand() < 0.15:
            m = np.random.choice(all_meds)

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
le_season = LabelEncoder()
le_disease = LabelEncoder()

df['Season_Encoded'] = le_season.fit_transform(df['Season'])
df['Disease_Encoded'] = le_disease.fit_transform(df['Prevailing_Disease'])

X = df[['Season_Encoded', 'Temperature_C', 'Disease_Encoded']]
y = df['Top_Selling_Medicine']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Baseline Random Forest Model
rf_model = RandomForestClassifier(n_estimators=50, random_state=42)
rf_model.fit(X_train, y_train)
rf_accuracy = accuracy_score(y_test, rf_model.predict(X_test))

# --- 3. Streamlit Frontend UI ---
st.title("💊 Pharmacy Sales & Inventory Predictor")
st.markdown("A complete data warehousing and mining project integrating classification and clustering.")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📊 Dataset Overview")
    st.write(f"Total Records: **{len(df)} rows** (Including random noise for realistic accuracy)")
    st.dataframe(df.sample(5, random_state=42), use_container_width=True)
    
    st.subheader("📈 Overall Sales Distribution")
    medicine_counts = df['Top_Selling_Medicine'].value_counts()
    st.bar_chart(medicine_counts)

with col2:
    st.subheader("🤖 Predict Local Demand")
    
    input_season = st.selectbox("Select Current Season", df['Season'].unique())
    filtered_diseases = df[df['Season'] == input_season]['Prevailing_Disease'].unique()
    input_disease = st.selectbox("Select Prevailing Local Disease", filtered_diseases)
    input_temp = st.slider("Average Temperature (°C)", min_value=-10, max_value=50, value=25)
    
    if st.button("Predict High-Demand Medicine", type="primary"):
        season_encoded = le_season.transform([input_season])[0]
        disease_encoded = le_disease.transform([input_disease])[0]
        
        input_data = pd.DataFrame({
            'Season_Encoded': [season_encoded],
            'Temperature_C': [input_temp],
            'Disease_Encoded': [disease_encoded]
        })
        
        prediction = rf_model.predict(input_data)[0]
        st.success(f"**Predicted Top Seller:** {prediction}")

st.markdown("---")
st.header("📂 Data Warehouse & Analytics Dashboard")

col3, col4 = st.columns([1, 1])

with col3:
    st.subheader("🧠 Model Feature Importance")
    feature_importances = rf_model.feature_importances_
    importance_df = pd.DataFrame({
        'Feature': ['Season', 'Temperature', 'Disease'],
        'Importance Level': feature_importances
    }).set_index('Feature')
    st.bar_chart(importance_df)

with col4:
    st.subheader("📦 Warehouse Inventory Status")
    inventory_data = pd.DataFrame({
        'Medicine Category': df['Top_Selling_Medicine'].unique(),
        'Stock (Units)': np.random.randint(100, 1500, size=len(df['Top_Selling_Medicine'].unique())),
        'Status': np.random.choice(['Optimal', 'Low Stock', 'Reorder Soon'], size=len(df['Top_Selling_Medicine'].unique()), p=[0.5, 0.3, 0.2])
    })
    st.dataframe(inventory_data, use_container_width=True)

st.markdown("---")
st.header("⚖️ Algorithm Performance Comparison")

# Train J48 (Decision Tree)
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
dt_accuracy = accuracy_score(y_test, dt_model.predict(X_test))

# Train Naive Bayes
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)
nb_accuracy = accuracy_score(y_test, nb_model.predict(X_test))

comparison_df = pd.DataFrame({
    'Algorithm': ['Random Forest', 'J48 (Decision Tree)', 'Naive Bayes'],
    'Accuracy (%)': [rf_accuracy * 100, dt_accuracy * 100, nb_accuracy * 100]
})
st.bar_chart(comparison_df.set_index('Algorithm'))

st.markdown("---")
st.header("🧩 Unsupervised Learning: K-Means Clustering")
st.write("Grouping data points purely based on Temperature and Disease factors (without looking at the target labels).")

X_cluster = df[['Temperature_C', 'Disease_Encoded']]
kmeans = KMeans(n_clusters=3, random_state=42, n_init="auto")
df['Cluster'] = kmeans.fit_predict(X_cluster)
sil_score = silhouette_score(X_cluster, df['Cluster'])

st.info(f"**Silhouette Score:** {sil_score:.2f} (Score ranges from -1 to 1)")

fig, ax = plt.subplots()
scatter = ax.scatter(df['Temperature_C'], df['Disease_Encoded'], c=df['Cluster'], cmap='viridis', alpha=0.6)
ax.set_xlabel("Temperature (°C)")
ax.set_ylabel("Disease (Encoded)")
st.pyplot(fig)

st.markdown("---")
st.header("🏆 Master DWM Performance Summary")
st.write("A unified chart comparing the evaluation metrics of all Data Warehousing and Mining (DWM) algorithms applied in this project.")

# Combine all model scores into a single dataframe using a 0 to 1 scale
master_performance_df = pd.DataFrame({
    'DWM Algorithm': [
        'Random Forest (Accuracy)', 
        'J48 / Decision Tree (Accuracy)', 
        'Naive Bayes (Accuracy)', 
        'K-Means (Silhouette Score)'
    ],
    'Evaluation Score (0 to 1)': [rf_accuracy, dt_accuracy, nb_accuracy, sil_score]
})

# Render the unified chart
st.bar_chart(master_performance_df.set_index('DWM Algorithm'))

st.markdown("---")

# Convert the Random Forest accuracy to a percentage
final_accuracy = rf_accuracy * 100


    unsafe_allow_html=True
)
