import streamlit as st
import pickle
import pandas as pd
import shap

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

xgb_model = model.named_steps["model"]
explainer = shap.TreeExplainer(xgb_model)

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
# Sidebar - Grouped Inputs
# -------------------------------------------------
st.sidebar.header("Customer Inputs")

with st.sidebar.expander("Demographics", expanded=True):
    Gender = st.selectbox("Gender", ["Male", "Female"])
    MaritalStatus = st.selectbox("Marital Status", ["Single", "Married"])
    CityTier = st.selectbox("City Tier", [1, 2, 3])

with st.sidebar.expander("Preferences"):
    PreferredLoginDevice = st.selectbox(
        "Preferred Login Device",
        ["Mobile Phone", "Computer"]
    )

    PreferredPaymentMode = st.selectbox(
        "Preferred Payment Method",
        ["Debit Card", "Credit Card", "UPI", "E-wallet", "Cash on Delivery"]
    )

    PreferedOrderCat = st.selectbox(
        "Preferred Product Category",
        ["Laptop & Accessory", "Mobile Phone", "Fashion", "Grocery"]
    )

with st.sidebar.expander("Usage & Transactions"):
    Tenure = st.number_input("Customer Tenure (months)", min_value=0)
    WarehouseToHome = st.number_input("Distance from Warehouse to Home", min_value=0)
    HourSpendOnApp = st.number_input("Avg Hours Spent on App per Day", min_value=0)
    NumberOfDeviceRegistered = st.number_input("Registered Devices", min_value=1)
    NumberOfAddress = st.number_input("Registered Addresses", min_value=1)
    OrderAmountHikeFromlastYear = st.number_input("Order Amount Increase (%)", min_value=0)
    CouponUsed = st.number_input("Coupons Used", min_value=0)
    OrderCount = st.number_input("Total Orders", min_value=0)
    DaySinceLastOrder = st.number_input("Days Since Last Order", min_value=0)
    CashbackAmount = st.number_input("Total Cashback Amount", min_value=0)

with st.sidebar.expander("Satisfaction"):
    SatisfactionScore = st.slider(
        "Customer Satisfaction (1 = Very Low, 5 = Very High)",
        1, 5
    )

    Complain_text = st.selectbox(
        "Ever submitted complaint?",
        ["No", "Yes"]
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
    st.write("Click the button below to generate churn prediction.")

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

    proba = model.predict_proba(input_df)[0][1]
    prediction = 1 if proba >= 0.35 else 0

    st.divider()
    st.write(f"Churn Probability: {proba:.2%}")

    if prediction == 1:
        st.error("**HIGH RISK** — Customer is likely to churn.")
    else:
        st.success("**LOW RISK** — Customer is likely to stay.")

    # -------------------------------------------------
    # Top Contributing Factors (SHAP - Text)
    # -------------------------------------------------
    st.subheader("Top Contributing Factors")

    feature_mapping = {
        "SatisfactionScore": "Customer satisfaction",
        "Tenure": "Customer tenure",
        "DaySinceLastOrder": "Time since last order",
        "Complain": "History of complaints",
        "WarehouseToHome": "Distance from warehouse to home",
        "HourSpendOnApp": "Average hours spent on app",
        "NumberOfDeviceRegistered": "Number of registered devices",
        "NumberOfAddress": "Number of registered addresses",
        "OrderAmountHikeFromlastYear": "Order amount increase from last year",
        "CouponUsed": "Number of coupons used",
        "OrderCount": "Total number of orders",
        "CashbackAmount": "Total cashback amount",
        "CityTier": "City tier",
        "PreferredLoginDevice": "Preferred login device",
        "PreferredPaymentMode": "Preferred payment method",
        "PreferedOrderCat": "Preferred product category",
        "Gender": "Gender",
        "MaritalStatus": "Marital status"
    }

    X_transformed = model.named_steps["preprocessor"].transform(input_df)
    shap_values = explainer.shap_values(X_transformed)
    feature_names = model.named_steps["preprocessor"].get_feature_names_out()

    shap_df = pd.DataFrame({
        "feature": feature_names,
        "impact": shap_values[0]
    })

    shap_df["abs_impact"] = shap_df["impact"].abs()
    top_features = shap_df.sort_values("abs_impact", ascending=False).head(5)

    for _, row in top_features.iterrows():

        raw_feature = row["feature"]

        # Remove transformer prefix
        if "__" in raw_feature:
            cleaned = raw_feature.split("__")[1]
        else:
            cleaned = raw_feature

        # Handle one-hot encoded categorical
        base_feature = cleaned.split("_")[0]

        friendly_name = feature_mapping.get(
            base_feature,
            base_feature
        )

        direction = (
            "increased churn risk"
            if row["impact"] > 0
            else "reduced churn risk"
        )

        st.write(f"- {friendly_name} {direction}")

# -------------------------------------------------
# Batch Prediction
# -------------------------------------------------
st.divider()
st.subheader("Batch Prediction (Upload CSV)")

uploaded_file = st.file_uploader(
    "Upload CSV file",
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
