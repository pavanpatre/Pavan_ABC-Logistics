import streamlit as st
import joblib
import pandas as pd

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Delivery Delay Predictor",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# CUSTOM CSS
# =========================================================
st.markdown("""
<style>

/* ---------- GLOBAL ---------- */
.stApp {
    background: linear-gradient(135deg, #f8fafc 0%, #eef2ff 100%);
}

.block-container {
    max-width: 1200px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

/* ---------- HERO ---------- */
.hero {
    background: linear-gradient(135deg, #111827, #1e3a8a);
    padding: 35px 40px;
    border-radius: 22px;
    margin-bottom: 25px;
    color: white;
    box-shadow: 0 15px 35px rgba(30, 58, 138, 0.18);
}

.hero h1 {
    font-size: 42px;
    font-weight: 800;
    margin-bottom: 8px;
    letter-spacing: -1px;
}

.hero p {
    font-size: 17px;
    color: #dbeafe;
    margin-bottom: 0;
}

/* ---------- SECTION CARDS ---------- */
.section-card {
    background: rgba(255,255,255,0.95);
    padding: 25px;
    border-radius: 18px;
    border: 1px solid #e5e7eb;
    box-shadow: 0 8px 25px rgba(15, 23, 42, 0.06);
    margin-bottom: 20px;
}

.section-title {
    font-size: 21px;
    font-weight: 750;
    color: #111827;
    margin-bottom: 3px;
}

.section-subtitle {
    color: #6b7280;
    font-size: 14px;
    margin-bottom: 18px;
}

/* ---------- RESULT CARD ---------- */
.result-card {
    padding: 30px;
    border-radius: 20px;
    text-align: center;
    margin-top: 25px;
    box-shadow: 0 10px 30px rgba(15, 23, 42, 0.10);
}

.result-on-time {
    background: linear-gradient(135deg, #ecfdf5, #d1fae5);
    border: 1px solid #86efac;
}

.result-delayed {
    background: linear-gradient(135deg, #fff7ed, #fed7aa);
    border: 1px solid #fdba74;
}

.result-icon {
    font-size: 55px;
    margin-bottom: 8px;
}

.result-title {
    font-size: 32px;
    font-weight: 800;
    margin-bottom: 5px;
}

.result-subtitle {
    font-size: 15px;
    color: #4b5563;
}

/* ---------- METRIC CARDS ---------- */
.metric-card {
    background: white;
    padding: 18px;
    border-radius: 15px;
    border: 1px solid #e5e7eb;
    text-align: center;
    box-shadow: 0 5px 18px rgba(15, 23, 42, 0.05);
}

.metric-value {
    font-size: 28px;
    font-weight: 800;
    color: #111827;
}

.metric-label {
    font-size: 13px;
    color: #6b7280;
}

/* ---------- BUTTON ---------- */
.stButton > button,
.stFormSubmitButton > button {
    width: 100%;
    border-radius: 12px;
    height: 3.2em;
    font-weight: 700;
    font-size: 16px;
    border: none;
    background: linear-gradient(90deg, #2563eb, #4f46e5);
    color: white;
    transition: all 0.2s ease;
}

.stButton > button:hover,
.stFormSubmitButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 20px rgba(37, 99, 235, 0.25);
}

/* ---------- SIDEBAR ---------- */
[data-testid="stSidebar"] {
    background: #111827;
}

[data-testid="stSidebar"] * {
    color: #f9fafb !important;
}

/* ---------- INPUT LABELS ---------- */
label {
    font-weight: 600 !important;
}

/* ---------- FOOTER ---------- */
.footer {
    text-align: center;
    color: #6b7280;
    font-size: 13px;
    margin-top: 35px;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# LOAD MODEL
# =========================================================
@st.cache_resource
def load_model():
    return joblib.load("delivery_delay.sav")


try:
    model = load_model()
except Exception as e:
    st.error("⚠️ Unable to load the trained model.")
    st.code(str(e))
    st.stop()

# =========================================================
# FEATURE COLUMNS
# =========================================================
feature_columns = [
    'Delivery_Distance',
    'Traffic_Congestion',
    'Weather_Condition',
    'Delivery_Slot',
    'Driver_Experience',
    'Num_Stops',
    'Vehicle_Age',
    'Road_Condition_Score',
    'Package_Weight',
    'Fuel_Efficiency',
    'Warehouse_Processing_Time'
]

# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:

    st.markdown("## 🚚 Delivery AI")

    st.markdown("""
    **Delivery Delay Predictor**

    This machine learning application predicts whether a delivery is likely to arrive **on time or with a delay** based on operational and environmental conditions.
    """)

    st.divider()

    st.markdown("### 📊 Model Inputs")

    st.markdown("""
    - 📍 Delivery distance
    - 🚦 Traffic congestion
    - 🌦️ Weather conditions
    - 🕐 Delivery slot
    - 👨‍✈️ Driver experience
    - 📦 Number of stops
    - 🚗 Vehicle age
    - 🛣️ Road condition
    - ⚖️ Package weight
    - ⛽ Fuel efficiency
    - 🏭 Warehouse processing time
    """)

    st.divider()

    st.caption("ML Delivery Analytics Dashboard")
    st.caption("Built with Python + Streamlit")

# =========================================================
# HERO HEADER
# =========================================================
st.markdown("""
<div class="hero">
    <h1>🚚 Delivery Delay Predictor</h1>
    <p>
        Predict delivery performance using operational,
        vehicle, traffic and environmental conditions.
    </p>
</div>
""", unsafe_allow_html=True)

# =========================================================
# INTRO METRICS
# =========================================================
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-value">11</div>
        <div class="metric-label">Prediction Features</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-value">2</div>
        <div class="metric-label">Prediction Outcomes</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-value">AI</div>
        <div class="metric-label">Machine Learning Model</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# =========================================================
# INPUT FORM
# =========================================================
with st.form("prediction_form"):

    # -----------------------------------------------------
    # DELIVERY CONDITIONS
    # -----------------------------------------------------
    st.markdown("""
    <div class="section-card">
        <div class="section-title">📍 Delivery Conditions</div>
        <div class="section-subtitle">
            Enter information about the delivery route and schedule.
        </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        delivery_distance = st.slider(
            "Delivery Distance",
            min_value=1.0,
            max_value=100.0,
            value=20.0,
            step=0.5,
            help="Total distance between the warehouse and delivery destination."
        )

    with col2:
        traffic_congestion = st.select_slider(
            "Traffic Congestion",
            options=[1, 2, 3, 4, 5],
            value=2,
            format_func=lambda x: {
                1: "1 — Very Low",
                2: "2 — Low",
                3: "3 — Moderate",
                4: "4 — High",
                5: "5 — Severe"
            }[x]
        )

    with col3:
        weather_condition = st.select_slider(
            "Weather Condition",
            options=[1, 2, 3, 4, 5],
            value=2,
            format_func=lambda x: {
                1: "1 — Excellent",
                2: "2 — Good",
                3: "3 — Moderate",
                4: "4 — Bad",
                5: "5 — Severe"
            }[x]
        )

    col1, col2 = st.columns(2)

    with col1:
        delivery_slot = st.selectbox(
            "Delivery Slot",
            options=[1, 2, 3],
            format_func=lambda x: {
                1: "🌅 Morning",
                2: "☀️ Afternoon",
                3: "🌆 Evening"
            }[x]
        )

    with col2:
        road_condition_score = st.select_slider(
            "Road Condition",
            options=[1, 2, 3, 4, 5],
            value=4,
            format_func=lambda x: {
                1: "1 — Poor",
                2: "2 — Below Average",
                3: "3 — Average",
                4: "4 — Good",
                5: "5 — Excellent"
            }[x]
        )

    st.markdown("</div>", unsafe_allow_html=True)

    # -----------------------------------------------------
    # VEHICLE & DRIVER
    # -----------------------------------------------------
    st.markdown("""
    <div class="section-card">
        <div class="section-title">🚗 Vehicle & Driver</div>
        <div class="section-subtitle">
            Provide vehicle and driver-related information.
        </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        driver_experience = st.slider(
            "Driver Experience",
            min_value=1,
            max_value=20,
            value=5,
            help="Driver experience in years."
        )

    with col2:
        vehicle_age = st.slider(
            "Vehicle Age",
            min_value=1,
            max_value=10,
            value=3,
            help="Age of the delivery vehicle."
        )

    with col3:
        fuel_efficiency = st.slider(
            "Fuel Efficiency",
            min_value=5.0,
            max_value=25.0,
            value=15.0,
            step=0.5,
            help="Vehicle fuel efficiency in km/l."
        )

    st.markdown("</div>", unsafe_allow_html=True)

    # -----------------------------------------------------
    # PACKAGE & OPERATIONS
    # -----------------------------------------------------
    st.markdown("""
    <div class="section-card">
        <div class="section-title">📦 Package & Operations</div>
        <div class="section-subtitle">
            Enter package and warehouse processing information.
        </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        num_stops = st.slider(
            "Number of Stops",
            min_value=1,
            max_value=15,
            value=3
        )

    with col2:
        package_weight = st.slider(
            "Package Weight",
            min_value=0.1,
            max_value=50.0,
            value=5.0,
            step=0.1
        )

    with col3:
        warehouse_processing_time = st.slider(
            "Warehouse Processing",
            min_value=10,
            max_value=120,
            value=60,
            step=5,
            help="Processing time at warehouse in minutes."
        )

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # -----------------------------------------------------
    # SUBMIT
    # -----------------------------------------------------
    submitted = st.form_submit_button(
        "🚀 Predict Delivery Outcome"
    )

# =========================================================
# PREDICTION
# =========================================================
if submitted:

    input_data = {
        'Delivery_Distance': delivery_distance,
        'Traffic_Congestion': traffic_congestion,
        'Weather_Condition': weather_condition,
        'Delivery_Slot': delivery_slot,
        'Driver_Experience': driver_experience,
        'Num_Stops': num_stops,
        'Vehicle_Age': vehicle_age,
        'Road_Condition_Score': road_condition_score,
        'Package_Weight': package_weight,
        'Fuel_Efficiency': fuel_efficiency,
        'Warehouse_Processing_Time': warehouse_processing_time
    }

    input_df = pd.DataFrame([input_data], columns=feature_columns)

    try:

        # Prediction
        prediction = model.predict(input_df)

        # Probability
        probability = model.predict_proba(input_df)[0]

        # Safely identify probability for class 1
        if hasattr(model, "classes_"):
            classes = list(model.classes_)

            if 1 in classes:
                delayed_index = classes.index(1)
                delayed_probability = probability[delayed_index]
            else:
                delayed_probability = probability[-1]
        else:
            delayed_probability = probability[-1]

        delayed_percentage = delayed_probability * 100
        on_time_percentage = 100 - delayed_percentage

        st.markdown("---")

        # =================================================
        # RESULT
        # =================================================
        if prediction[0] == 1:

            st.markdown(f"""
            <div class="result-card result-delayed">
                <div class="result-icon">⚠️</div>
                <div class="result-title">DELIVERY LIKELY DELAYED</div>
                <div class="result-subtitle">
                    The model indicates a higher probability of delivery delay.
                </div>
            </div>
            """, unsafe_allow_html=True)

        else:

            st.markdown(f"""
            <div class="result-card result-on-time">
                <div class="result-icon">✅</div>
                <div class="result-title">DELIVERY LIKELY ON TIME</div>
                <div class="result-subtitle">
                    The model indicates a higher probability of on-time delivery.
                </div>
            </div>
            """, unsafe_allow_html=True)

        # =================================================
        # PROBABILITY
        # =================================================
        st.markdown("<br>", unsafe_allow_html=True)

        st.subheader("📊 Prediction Confidence")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "On-Time Probability",
                f"{on_time_percentage:.1f}%"
            )
            st.progress(
                max(0.0, min(1.0, on_time_percentage / 100))
            )

        with col2:
            st.metric(
                "Delay Probability",
                f"{delayed_percentage:.1f}%"
            )
            st.progress(
                max(0.0, min(1.0, delayed_percentage / 100))
            )

        # =================================================
        # INPUT SUMMARY
        # =================================================
        st.markdown("<br>", unsafe_allow_html=True)

        with st.expander("🔎 View Prediction Inputs"):

            display_df = pd.DataFrame({
                "Feature": [
                    "Delivery Distance",
                    "Traffic Congestion",
                    "Weather Condition",
                    "Delivery Slot",
                    "Driver Experience",
                    "Number of Stops",
                    "Vehicle Age",
                    "Road Condition",
                    "Package Weight",
                    "Fuel Efficiency",
                    "Warehouse Processing"
                ],
                "Value": [
                    f"{delivery_distance:.1f} km",
                    traffic_congestion,
                    weather_condition,
                    {
                        1: "Morning",
                        2: "Afternoon",
                        3: "Evening"
                    }[delivery_slot],
                    f"{driver_experience} years",
                    num_stops,
                    f"{vehicle_age} years",
                    road_condition_score,
                    f"{package_weight:.1f} kg",
                    f"{fuel_efficiency:.1f} km/l",
                    f"{warehouse_processing_time} min"
                ]
            })

            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True
            )

        # =================================================
        # SIMPLE INTERPRETATION
        # =================================================
        st.markdown("<br>", unsafe_allow_html=True)

        if delayed_percentage >= 70:

            st.warning(
                "🚨 **High delay probability:** "
                "Consider reviewing route conditions, traffic, "
                "warehouse processing time and delivery scheduling."
            )

        elif delayed_percentage >= 50:

            st.info(
                "⚠️ **Moderate delay probability:** "
                "Operational conditions may require additional attention."
            )

        else:

            st.success(
                "✅ **Lower delay probability:** "
                "The current combination of delivery conditions is "
                "more favourable for an on-time delivery."
            )

    except Exception as e:

        st.error("Something went wrong while generating the prediction.")
        st.exception(e)

# =========================================================
# FOOTER
# =========================================================
st.markdown("""
<div class="footer">
    🚚 Delivery Delay Prediction System &nbsp;•&nbsp;
    Powered by Machine Learning & Streamlit
</div>
""", unsafe_allow_html=True)
