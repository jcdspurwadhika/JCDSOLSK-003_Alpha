import streamlit as st
import pickle
import pandas as pd

# -------------------------------------------------
# Page Configuration
# -------------------------------------------------
st.set_page_config(
    page_title="Customer Churn Prediction",
    layout="wide"
)

# -------------------------------------------------
# Load Model
# -------------------------------------------------
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# -------------------------------------------------
# Header
# -------------------------------------------------
st.markdown(
    """
    <h1 style='text-align:center;'>Customer Churn Prediction System</h1>
    <p style='text-align:center; color:gray;'>
    Predict whether a customer is likely to churn based on behavioral and demographic data
    </p>
    """,
    unsafe_allow_html=True
)

st.divider()

# -------------------------------------------------
# Sidebar - Customer Profile
# -------------------------------------------------
st.sidebar.header("Customer Profile")

PreferredLoginDevice = st.sidebar.selectbox(
    "Preferred Login Device",
    ["Mobile Phone", "Computer"]
)

PreferredPaymentMode = st.sidebar.selectbox(
    "Preferred Payment Method",
    ["Debit Card", "Credit Card", "UPI", "E-wallet", "Cash on Delivery"]
)

Gender = st.sidebar.selectbox("Gender", ["Male", "Female"])

PreferedOrderCat = st.sidebar.selectbox(
    "Preferred Product Category",
    ["Laptop & Accessory", "Mobile Phone", "Fashion", "Grocery"]
)

MaritalStatus = st.sidebar.selectbox(
    "Marital Status",
    ["Single", "Married"]
)

st.sidebar.subheader("Usage & Transaction Information")

Tenure = st.sidebar.number_input(
    "Customer Tenure (months)",
    min_value=0
)

CityTier = st.sidebar.selectbox(
    "City Tier",
    [1, 2, 3]
)

WarehouseToHome = st.sidebar.number_input(
    "Distance from Warehouse to Home",
    min_value=0
)

HourSpendOnApp = st.sidebar.number_input(
    "Average Hours Spent on App per Day",
    min_value=0
)

NumberOfDeviceRegistered = st.sidebar.number_input(
    "Number of Registered Devices",
    min_value=1
)

SatisfactionScore = st.sidebar.slider(
    "Customer Satisfaction Score (1 = Very Low, 5 = Very High)",
    1, 5
)

NumberOfAddress = st.sidebar.number_input(
    "Number of Registered Addresses",
    min_value=1
)

Complain_text = st.sidebar.selectbox(
    "Has the customer ever submitted a complaint?",
    ["No", "Yes"]
)

OrderAmountHikeFromlastYear = st.sidebar.number_input(
    "Order Amount Increase Compared to Last Year (%)",
    min_value=0
)

CouponUsed = st.sidebar.number_input(
    "Number of Coupons Used",
    min_value=0
)

OrderCount = st.sidebar.number_input(
    "Total Number of Orders",
    min_value=0
)

DaySinceLastOrder = st.sidebar.number_input(
    "Days Since Last Order",
    min_value=0
)

CashbackAmount = st.sidebar.number_input(
    "Total Cashback Amount",
    min_value=0
)

# -------------------------------------------------
# Convert Friendly Inputs
# -------------------------------------------------
Complain = 1 if Complain_text == "Yes" else 0
SatisfactionScore_model = 6 - SatisfactionScore

# -------------------------------------------------
# Main Panel
# -------------------------------------------------
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Prediction Result")
    st.write("Click the button to generate churn prediction")

with col2:
    predict_btn = st.button("Run Prediction", use_container_width=True)

# -------------------------------------------------
# Prediction
# -------------------------------------------------
if predict_btn:

    input_df = pd.DataFrame({
        "PreferredLoginDevice": [PreferredLoginDevice],
        "PreferredPaymentMode": [PreferredPaymentMode],
        "Gender": [Gender],
        "PreferedOrderCat": [PreferedOrderCat],
        "MaritalStatus": [MaritalStatus],
        "Tenure": [Tenure],
        "CityTier": [CityTier],
        "WarehouseToHome": [WarehouseToHome],
        "HourSpendOnApp": [HourSpendOnApp],
        "NumberOfDeviceRegistered": [NumberOfDeviceRegistered],
        "SatisfactionScore": [SatisfactionScore_model],
        "NumberOfAddress": [NumberOfAddress],
        "Complain": [Complain],
        "OrderAmountHikeFromlastYear": [OrderAmountHikeFromlastYear],
        "CouponUsed": [CouponUsed],
        "OrderCount": [OrderCount],
        "DaySinceLastOrder": [DaySinceLastOrder],
        "CashbackAmount": [CashbackAmount]
    })

    # Probability-based prediction
    proba = model.predict_proba(input_df)[0][1]
    prediction = 1 if proba >= 0.35 else 0

    st.divider()
    st.write(f"Churn Probability: {proba:.2%}")

    if prediction == 1:
        st.error("High risk: Customer is likely to churn.")
        st.write("Recommended action: Provide retention offer or personalized promotion.")
    else:
        st.success("Low risk: Customer is likely to stay.")
        st.write("Recommended action: Maintain engagement and loyalty program.")

# -------------------------------------------------
# Batch Prediction via CSV Upload
# -------------------------------------------------
st.divider()
st.subheader("Batch Prediction (Upload CSV)")


uploaded_file = st.file_uploader(
    "Upload CSV file for batch churn prediction",
    type=["csv"]
)

if uploaded_file is not None:

    try:
        df_batch = pd.read_csv(uploaded_file, sep=None, engine="python")

        st.write("Preview of uploaded data:")
        st.dataframe(df_batch, height=300)

        probas = model.predict_proba(df_batch)[:, 1]
        preds = model.predict(df_batch)

        df_batch["churn_probability"] = probas
        df_batch["churn_flag"] = ["Churn" if p == 1 else "Non-Churn" for p in preds]


        st.success("Batch prediction completed.")

        st.write("Preview result:")
        st.dataframe(df_batch, height=300)

        csv_result = df_batch.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="Download Result CSV",
            data=csv_result,
            file_name="churn_prediction_result.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error("Error processing file.")
        st.text(str(e))

# -------------------------------------------------
# Footer
# -------------------------------------------------
st.markdown(
    """
    <hr>
    <p style='text-align:center; color:gray; font-size:12px;'>
    Machine Learning Deployment Project using Streamlit
    </p>
    """,
    unsafe_allow_html=True
)
