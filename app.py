import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler  # Needed even if loading from pickle
import numpy as np

# --- 1. Load Model and Scaler ---
@st.cache_resource
def load_assets():
    try:
        # Load K-Means model
        with open('kmeans_model.pkl', 'rb') as file:
            loaded_model = pickle.load(file)
        
        # Load StandardScaler
        with open('scaler.pkl', 'rb') as file:
            loaded_scaler = pickle.load(file)
        
        return loaded_model, loaded_scaler
    except FileNotFoundError:
        st.error("Error: 'kmeans_model.pkl' or 'scaler.pkl' not found. Make sure they are in the same directory as this app.")
        return None, None

model, scaler = load_assets()

if model is None or scaler is None:
    st.stop()  # Stop app if loading fails

# --- 2. Streamlit UI ---
st.title("🛍️ Customer Segmentation App (K-Means)")
st.sidebar.header("Choose Input Type")

# --- 3. Prediction Function ---
def predict_cluster(data, scaler, model):
    # Convert data to DataFrame with proper column names
    data_df = pd.DataFrame(data, columns=['Annual Income (k$)', 'Spending Score (1-100)'])
    
    # Scale data (same as training)
    data_scaled = scaler.transform(data_df)
    
    # Predict clusters
    predictions = model.predict(data_scaled)
    return predictions

# --- 4. Select Input Type (Single Customer or CSV Upload) ---
input_type = st.sidebar.radio("Select Input Method", ["Single Customer Input", "Upload CSV File"])

if input_type == "Single Customer Input":
    st.header("👤 Predict Single Customer Cluster")
    
    # Input fields for a single customer
    income = st.number_input("Annual Income (k$)", min_value=1.0, max_value=200.0, value=50.0)
    spending = st.number_input("Spending Score (1-100)", min_value=1, max_value=100, value=50)
    
    if st.button("Predict Cluster"):
        single_customer_data = np.array([[income, spending]])
        cluster_prediction = predict_cluster(single_customer_data, scaler, model)
        
        st.success(f"Predicted Cluster for this customer is: **Cluster {cluster_prediction[0]}**")
        
        # Provide business insight based on cluster
        cluster_profiles = {
            0: "Average Customers: Moderate income and spending. (Target for upselling/loyalty programs)",
            1: "High-Spending, High-Income Customers: The 'Moneymakers'. (Highly valuable, focus on premium offers)",
            2: "High-Spending, Low-Income Customers: The 'Careful Spenders'. (Potential for impulse buys, focus on value)",
            3: "Low-Spending, High-Income Customers: The 'Miser'. (Hard to convert, focus on high-value/necessity items)",
            4: "Low-Spending, Low-Income Customers: The 'General' or 'Strugglers'. (Focus on basic necessities, cost-sensitive)",
        }
        
        st.info(f"**Insight:** {cluster_profiles.get(cluster_prediction[0], 'No specific insight available for this cluster.')}")

elif input_type == "Upload CSV File":
    st.header("📂 Predict Clusters for Uploaded CSV")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            new_data_df = pd.read_csv(uploaded_file)
            st.write("Original Data Preview:")
            st.dataframe(new_data_df.head())
            
            # Check required columns
            required_cols = ['Annual Income (k$)', 'Spending Score (1-100)']
            if not all(col in new_data_df.columns for col in required_cols):
                st.error(f"Error: The uploaded CSV must contain the columns: {required_cols}")
            else:
                X_new = new_data_df[required_cols]
                cluster_predictions = predict_cluster(X_new, scaler, model)
                new_data_df['Predicted Cluster'] = cluster_predictions
                
                st.subheader("Results with Predicted Clusters")
                st.dataframe(new_data_df)
                
                # Allow CSV download
                @st.cache_data
                def convert_df(df):
                    return df.to_csv(index=False).encode('utf-8')

                csv = convert_df(new_data_df)

                st.download_button(
                    label="Download CSV with Predicted Clusters",
                    data=csv,
                    file_name='clustered_customers.csv',
                    mime='text/csv',
                )
        except Exception as e:
            st.error(f"An error occurred while processing the file: {e}")

# --- 5. Optional Model Info ---
st.sidebar.markdown("---")
st.sidebar.caption(f"Model: K-Means with k={model.n_clusters} clusters.")
st.sidebar.caption("Features used: Annual Income and Spending Score.")

