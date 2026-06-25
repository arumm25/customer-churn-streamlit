
import streamlit as st
import pandas as pd
import joblib
from datetime import date, timedelta


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Customer Churn Prediction",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>
    .main-title {
        font-size: 42px;
        font-weight: 800;
        color: #1F2937;
        margin-bottom: 8px;
    }

    .subtitle {
        font-size: 17px;
        color: #4B5563;
        margin-bottom: 24px;
        line-height: 1.6;
    }

    .info-card {
        background-color: #F9FAFB;
        border: 1px solid #E5E7EB;
        border-radius: 14px;
        padding: 18px 22px;
        margin-bottom: 18px;
    }

    .result-card-high {
        background-color: #FEE2E2;
        border: 1px solid #FCA5A5;
        color: #991B1B;
        padding: 24px;
        border-radius: 16px;
        font-size: 22px;
        font-weight: 700;
        text-align: center;
        margin-top: 18px;
    }

    .result-card-low {
        background-color: #DCFCE7;
        border: 1px solid #86EFAC;
        color: #166534;
        padding: 24px;
        border-radius: 16px;
        font-size: 22px;
        font-weight: 700;
        text-align: center;
        margin-top: 18px;
    }

    .small-note {
        font-size: 14px;
        color: #6B7280;
        line-height: 1.5;
    }

    div[data-testid="stMetric"] {
        background-color: #FFFFFF;
        padding: 16px;
        border-radius: 14px;
        border: 1px solid #E5E7EB;
        box-shadow: 0 3px 10px rgba(0,0,0,0.035);
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD MODEL PACKAGE
# ============================================================

@st.cache_resource
def load_model_package():
    package = joblib.load("best_churn_model.joblib")
    return package


package = load_model_package()

model = package["model"]
best_threshold = package.get("best_threshold", 0.5)
feature_columns = package["feature_columns"]
categorical_options = package.get("categorical_options", {})
reference_date = pd.to_datetime(package["reference_date"])

metrics = package.get("metrics", {})
model_name = package.get("best_model_name", "Model")
scenario_name = package.get("best_scenario", "Scenario")


# ============================================================
# HELPER FUNCTION
# ============================================================

def get_options(column_name, default_options):
    options = categorical_options.get(column_name, default_options)

    if options is None or len(options) == 0:
        return default_options

    return options


def create_input_features(raw_input):
    data = pd.DataFrame([raw_input])

    data["signup_date"] = pd.to_datetime(data["signup_date"], errors="coerce")
    data["last_purchase_date"] = pd.to_datetime(data["last_purchase_date"], errors="coerce")

    # Feature engineering sesuai preprocessing notebook
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

    # Kolom mentah yang tidak dipakai model akhir
    drop_cols = [
        "customer_id",
        "signup_date",
        "last_purchase_date",
        "coupon_code"
    ]

    data = data.drop(columns=drop_cols, errors="ignore")

    # Pastikan urutan dan jumlah kolom sama dengan model
    data = data.reindex(columns=feature_columns)

    return data


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.header("Informasi Model")

    st.write(
        "Aplikasi ini menggunakan model machine learning terbaik "
        "untuk memprediksi potensi churn pelanggan."
    )

    st.metric("Model", str(model_name))
    st.metric("Skenario", str(scenario_name))
    st.metric("Threshold", round(float(best_threshold), 2))

    if "f1_score" in metrics:
        st.metric("F1-Score", round(float(metrics["f1_score"]), 4))

    if "recall" in metrics:
        st.metric("Recall", round(float(metrics["recall"]), 4))

    st.divider()

    st.caption(
        "Catatan: pelanggan diklasifikasikan churn apabila probabilitas churn "
        "lebih besar atau sama dengan threshold model."
    )


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">Customer Churn Prediction</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="subtitle">
    Aplikasi ini memprediksi apakah pelanggan berpotensi churn berdasarkan
    profil pelanggan, aktivitas penggunaan, transaksi, serta indikator risiko
    dan kepuasan pelanggan.
    </div>
    """,
    unsafe_allow_html=True
)

top_col1, top_col2, top_col3, top_col4 = st.columns(4)

with top_col1:
    st.metric("Model Terbaik", str(model_name))

with top_col2:
    st.metric("Threshold", round(float(best_threshold), 2))

with top_col3:
    st.metric("F1-Score", round(float(metrics.get("f1_score", 0)), 4))

with top_col4:
    st.metric("Recall", round(float(metrics.get("recall", 0)), 4))


st.markdown(
    """
    <div class="info-card">
    <b>Fitur yang ditampilkan:</b> 19 fitur utama. 
    Beberapa fitur tambahan seperti tanggal pendaftaran, tanggal pembelian terakhir,
    email rate, marketing spend, dan fitur hasil feature engineering tetap diproses
    otomatis di belakang layar agar sesuai dengan model.
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# INPUT FORM — 19 FITUR UTAMA
# ============================================================

st.markdown("## Input Data Pelanggan")

with st.form("customer_churn_form"):

    tab1, tab2, tab3, tab4 = st.tabs([
        "Profil Pelanggan",
        "Aktivitas Penggunaan",
        "Transaksi",
        "Risiko & Kepuasan"
    ])

    # --------------------------------------------------------
    # TAB 1: PROFIL PELANGGAN — 6 FITUR
    # --------------------------------------------------------
    with tab1:
        st.markdown("### Profil Pelanggan")

        col1, col2 = st.columns(2)

        with col1:
            gender = st.selectbox(
                "Gender",
                get_options("gender", ["Unknown"])
            )

            age = st.number_input(
                "Age",
                min_value=0.0,
                max_value=100.0,
                value=30.0,
                step=1.0
            )

            country = st.selectbox(
                "Country",
                get_options("country", ["Unknown"])
            )

        with col2:
            city = st.selectbox(
                "City",
                get_options("city", ["Unknown"])
            )

            device_type = st.selectbox(
                "Device Type",
                get_options("device_type", ["Unknown"])
            )

            subscription_type = st.selectbox(
                "Subscription Type",
                get_options("subscription_type", ["Unknown"])
            )

    # --------------------------------------------------------
    # TAB 2: AKTIVITAS PENGGUNAAN — 5 FITUR
    # --------------------------------------------------------
    with tab2:
        st.markdown("### Aktivitas Penggunaan")

        col1, col2, col3 = st.columns(3)

        with col1:
            is_premium_user = st.selectbox(
                "Is Premium User",
                [0, 1],
                help="0 = bukan premium, 1 = premium"
            )

            total_visits = st.number_input(
                "Total Visits",
                min_value=0,
                value=20,
                step=1
            )

        with col2:
            avg_session_time = st.number_input(
                "Average Session Time",
                min_value=0.0,
                value=10.0,
                step=0.5
            )

            pages_per_session = st.number_input(
                "Pages per Session",
                min_value=0.0,
                value=3.0,
                step=0.5
            )

        with col3:
            last_3_month_purchase_freq = st.number_input(
                "Last 3 Month Purchase Frequency",
                min_value=0,
                value=2,
                step=1
            )

    # --------------------------------------------------------
    # TAB 3: TRANSAKSI — 4 FITUR
    # --------------------------------------------------------
    with tab3:
        st.markdown("### Data Transaksi")

        col1, col2 = st.columns(2)

        with col1:
            total_spent = st.number_input(
                "Total Spent",
                min_value=0.0,
                value=500.0,
                step=50.0
            )

            avg_order_value = st.number_input(
                "Average Order Value",
                min_value=0.0,
                value=100.0,
                step=10.0
            )

        with col2:
            discount_used = st.selectbox(
                "Discount Used",
                [0, 1],
                help="0 = tidak menggunakan diskon, 1 = menggunakan diskon"
            )

            payment_method = st.selectbox(
                "Payment Method",
                get_options("payment_method", ["Unknown"])
            )

    # --------------------------------------------------------
    # TAB 4: RISIKO & KEPUASAN — 4 FITUR
    # --------------------------------------------------------
    with tab4:
        st.markdown("### Risiko & Kepuasan Pelanggan")

        col1, col2 = st.columns(2)

        with col1:
            support_tickets = st.number_input(
                "Support Tickets",
                min_value=0,
                value=1,
                step=1
            )

            refund_requested = st.selectbox(
                "Refund Requested",
                [0, 1],
                help="0 = tidak refund, 1 = mengajukan refund"
            )

        with col2:
            satisfaction_score = st.number_input(
                "Satisfaction Score",
                min_value=0.0,
                max_value=10.0,
                value=7.0,
                step=0.5
            )

            nps_score = st.number_input(
                "NPS Score",
                min_value=-100,
                max_value=100,
                value=30,
                step=1
            )

    submitted = st.form_submit_button(
        "Prediksi Churn",
        use_container_width=True
    )


# ============================================================
# PREDICTION
# ============================================================

if submitted:

    # 19 fitur utama berasal dari input user
    # fitur lain diisi default agar tetap sesuai dengan feature_columns model
    raw_input = {
        # ====================================================
        # FITUR YANG DITAMPILKAN DI STREAMLIT — 19 FITUR
        # ====================================================
        "gender": gender,
        "age": age,
        "country": country,
        "city": city,
        "device_type": device_type,
        "subscription_type": subscription_type,

        "is_premium_user": is_premium_user,
        "total_visits": total_visits,
        "avg_session_time": avg_session_time,
        "pages_per_session": pages_per_session,
        "last_3_month_purchase_freq": last_3_month_purchase_freq,

        "total_spent": total_spent,
        "avg_order_value": avg_order_value,
        "discount_used": discount_used,
        "payment_method": payment_method,

        "support_tickets": support_tickets,
        "refund_requested": refund_requested,
        "satisfaction_score": satisfaction_score,
        "nps_score": nps_score,

        # ====================================================
        # FITUR DEFAULT — TIDAK DITAMPILKAN DI UI
        # ====================================================
        "customer_id": 0,
        "signup_date": date.today() - timedelta(days=365),
        "last_purchase_date": date.today() - timedelta(days=30),
        "acquisition_channel": "Unknown",
        "email_open_rate": 0.5,
        "email_click_rate": 0.2,
        "coupon_code": "",
        "delivery_delay_days": 0,
        "marketing_spend_per_user": 50.0,
        "lifetime_value": 1000.0
    }

    input_df = create_input_features(raw_input)

    churn_probability = model.predict_proba(input_df)[0][1]
    prediction = int(churn_probability >= best_threshold)

    st.markdown("## Hasil Prediksi")

    result_col1, result_col2 = st.columns([1.3, 1])

    with result_col1:
        if prediction == 1:
            st.markdown(
                '<div class="result-card-high">Pelanggan diprediksi BERPOTENSI CHURN</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                '<div class="result-card-low">Pelanggan diprediksi TIDAK CHURN</div>',
                unsafe_allow_html=True
            )

        st.write("")
        st.progress(float(churn_probability))

        st.caption(
            "Progress bar menunjukkan probabilitas churn pelanggan. "
            "Semakin tinggi nilainya, semakin besar risiko pelanggan churn."
        )

    with result_col2:
        metric_col1, metric_col2 = st.columns(2)

        with metric_col1:
            st.metric(
                "Probabilitas Churn",
                "{:.2%}".format(churn_probability)
            )

        with metric_col2:
            st.metric(
                "Threshold",
                round(float(best_threshold), 2)
            )

        if churn_probability >= 0.75:
            risk_level = "Tinggi"
        elif churn_probability >= best_threshold:
            risk_level = "Sedang"
        else:
            risk_level = "Rendah"

        st.metric("Level Risiko", risk_level)

    st.markdown("### Perbandingan Probabilitas dengan Threshold")

    chart_df = pd.DataFrame(
        {
            "Nilai": [
                float(churn_probability),
                float(best_threshold)
            ]
        },
        index=[
            "Probabilitas Churn",
            "Threshold"
        ]
    )

    st.bar_chart(chart_df)

    with st.expander("Lihat data yang dikirim ke model"):
        st.dataframe(input_df, use_container_width=True)

    with st.expander("Interpretasi Singkat"):
        if prediction == 1:
            st.write(
                "Pelanggan memiliki probabilitas churn yang melewati threshold model. "
                "Pelanggan seperti ini sebaiknya diprioritaskan untuk strategi retensi, "
                "misalnya follow-up layanan, penawaran khusus, atau peningkatan pengalaman pelanggan."
            )
        else:
            st.write(
                "Pelanggan memiliki probabilitas churn di bawah threshold model. "
                "Pelanggan ini relatif lebih stabil, tetapi tetap perlu dipantau melalui aktivitas penggunaan, "
                "kepuasan, dan riwayat transaksi."
            )

else:
    st.markdown(
        """
        <div class="small-note">
        Isi data pelanggan pada form di atas, lalu klik tombol <b>Prediksi Churn</b>
        untuk melihat hasil klasifikasi pelanggan.
        </div>
        """,
        unsafe_allow_html=True
    )
