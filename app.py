
import streamlit as st
import pandas as pd
import joblib


st.set_page_config(
    page_title="Customer Churn Prediction",
    layout="wide"
)


@st.cache_resource
def load_model_package():
    package = joblib.load("best_churn_model.joblib")
    return package


package = load_model_package()

model = package["model"]
best_threshold = package.get("best_threshold", 0.5)
feature_columns = package["feature_columns"]
categorical_options = package["categorical_options"]
reference_date = pd.to_datetime(package["reference_date"])

st.title("Customer Churn Prediction")

st.write(
    "Aplikasi ini digunakan untuk memprediksi apakah pelanggan berpotensi churn "
    "atau tidak churn berdasarkan data pelanggan, aktivitas penggunaan, transaksi, "
    "interaksi pemasaran, dan kepuasan pelanggan."
)

st.info(
    "Model terbaik: "
    + str(package["best_model_name"])
    + " | Skenario: "
    + str(package["best_scenario"])
    + " | Threshold: "
    + str(round(float(best_threshold), 2))
    + " | F1-Score: "
    + str(round(float(package["metrics"]["f1_score"]), 4))
)

st.subheader("Input Data Pelanggan")

col1, col2, col3 = st.columns(3)

with col1:
    gender = st.selectbox("Gender", categorical_options.get("gender", ["Unknown"]))
    age = st.number_input("Age", min_value=0.0, max_value=100.0, value=30.0)
    country = st.selectbox("Country", categorical_options.get("country", ["Unknown"]))
    city = st.selectbox("City", categorical_options.get("city", ["Unknown"]))

    acquisition_channel = st.selectbox(
        "Acquisition Channel",
        categorical_options.get("acquisition_channel", ["Unknown"])
    )

    device_type = st.selectbox(
        "Device Type",
        categorical_options.get("device_type", ["Unknown"])
    )

    subscription_type = st.selectbox(
        "Subscription Type",
        categorical_options.get("subscription_type", ["Unknown"])
    )

    payment_method = st.selectbox(
        "Payment Method",
        categorical_options.get("payment_method", ["Unknown"])
    )

with col2:
    is_premium_user = st.selectbox("Is Premium User", [0, 1])
    total_visits = st.number_input("Total Visits", min_value=0, value=20)
    avg_session_time = st.number_input("Average Session Time", min_value=0.0, value=10.0)
    pages_per_session = st.number_input("Pages per Session", min_value=0.0, value=3.0)
    email_open_rate = st.number_input("Email Open Rate", min_value=0.0, max_value=1.0, value=0.5)
    email_click_rate = st.number_input("Email Click Rate", min_value=0.0, max_value=1.0, value=0.2)
    total_spent = st.number_input("Total Spent", min_value=0.0, value=500.0)
    avg_order_value = st.number_input("Average Order Value", min_value=0.0, value=100.0)

with col3:
    discount_used = st.selectbox("Discount Used", [0, 1])
    support_tickets = st.number_input("Support Tickets", min_value=0, value=1)
    refund_requested = st.selectbox("Refund Requested", [0, 1])
    delivery_delay_days = st.number_input("Delivery Delay Days", min_value=0, value=0)
    satisfaction_score = st.number_input("Satisfaction Score", min_value=0.0, max_value=10.0, value=7.0)
    nps_score = st.number_input("NPS Score", min_value=-100, max_value=100, value=30)
    marketing_spend_per_user = st.number_input("Marketing Spend per User", min_value=0.0, value=50.0)
    lifetime_value = st.number_input("Lifetime Value", min_value=0.0, value=1000.0)
    last_3_month_purchase_freq = st.number_input("Last 3 Month Purchase Frequency", min_value=0, value=2)

st.subheader("Input Tanggal dan Kupon")

date_col1, date_col2, date_col3 = st.columns(3)

with date_col1:
    signup_date = st.date_input("Signup Date")

with date_col2:
    last_purchase_date = st.date_input("Last Purchase Date")

with date_col3:
    coupon_code = st.text_input("Coupon Code", value="")


def create_input_features(raw_input):
    data = pd.DataFrame([raw_input])

    data["signup_date"] = pd.to_datetime(data["signup_date"], errors="coerce")
    data["last_purchase_date"] = pd.to_datetime(data["last_purchase_date"], errors="coerce")

    data["customer_tenure_days"] = (
        data["last_purchase_date"] - data["signup_date"]
    ).dt.days

    data["recency_days"] = (
        reference_date - data["last_purchase_date"]
    ).dt.days

    data["has_coupon_code"] = 1

    if str(raw_input.get("coupon_code", "")).strip() == "":
        data["has_coupon_code"] = 0

    data["signup_month"] = data["signup_date"].dt.month
    data["signup_year"] = data["signup_date"].dt.year

    drop_cols = [
        "customer_id",
        "signup_date",
        "last_purchase_date",
        "coupon_code"
    ]

    data = data.drop(columns=drop_cols, errors="ignore")
    data = data.reindex(columns=feature_columns)

    return data


raw_input = {
    "gender": gender,
    "age": age,
    "country": country,
    "city": city,
    "signup_date": signup_date,
    "last_purchase_date": last_purchase_date,
    "acquisition_channel": acquisition_channel,
    "device_type": device_type,
    "subscription_type": subscription_type,
    "is_premium_user": is_premium_user,
    "total_visits": total_visits,
    "avg_session_time": avg_session_time,
    "pages_per_session": pages_per_session,
    "email_open_rate": email_open_rate,
    "email_click_rate": email_click_rate,
    "total_spent": total_spent,
    "avg_order_value": avg_order_value,
    "discount_used": discount_used,
    "coupon_code": coupon_code,
    "support_tickets": support_tickets,
    "refund_requested": refund_requested,
    "delivery_delay_days": delivery_delay_days,
    "payment_method": payment_method,
    "satisfaction_score": satisfaction_score,
    "nps_score": nps_score,
    "marketing_spend_per_user": marketing_spend_per_user,
    "lifetime_value": lifetime_value,
    "last_3_month_purchase_freq": last_3_month_purchase_freq
}

if st.button("Prediksi Churn"):
    input_df = create_input_features(raw_input)

    churn_probability = model.predict_proba(input_df)[0][1]
    prediction = int(churn_probability >= best_threshold)

    st.subheader("Hasil Prediksi")

    if prediction == 1:
        st.error("Pelanggan diprediksi CHURN")
    else:
        st.success("Pelanggan diprediksi TIDAK CHURN")

    st.metric("Probabilitas Churn", "{:.2%}".format(churn_probability))
    st.write("Threshold yang digunakan:", round(float(best_threshold), 2))

    with st.expander("Lihat data input setelah feature engineering"):
        st.dataframe(input_df)
