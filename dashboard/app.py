import streamlit as st
import pandas as pd
import joblib
import json
import numpy as np
import math
import os
import altair as alt

# --- 1. Page Configuration ---
st.set_page_config(
    page_title="eCall Injury Prediction",
    page_icon="🚑",
    layout="wide"
)


# --- 2. Load Data & Model ---
@st.cache_resource
def load_model_assets():
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_dir, 'catboost_model.pkl')
        meta_path = os.path.join(base_dir, 'model_metadata.json')

        model = joblib.load(model_path)
        with open(meta_path, 'r') as f:
            metadata = json.load(f)
        return model, metadata
    except Exception as e:
        print(f"Error loading assets: {e}")
        return None, None


model, metadata = load_model_assets()


# --- 3. Helper Functions ---
def get_cyclical_features(value, max_val):
    sin_val = np.sin(2 * np.pi * value / max_val)
    cos_val = np.cos(2 * np.pi * value / max_val)
    return sin_val, cos_val


def get_impact_score(vehicle_type):
    mapping = {
        "Heavy Truck / Bus": 6,
        "Light Truck / Van": 5,
        "Passenger Car": 4,
        "Motorcycle": 3,
        "Bicycle / Scooter": 2,
        "Pedestrian": 0,
        "Unknown": 1
    }
    return mapping.get(vehicle_type, 1)


# --- 4. Session State & Scenario Management ---
# Initialize session state for all widgets if not present
def init_session_state():
    defaults = {
        'lighting_ordinal': 0, 'weather_ordinal': 0, 'surface_quality_indicator': 0,
        'time_of_day': 1, 'hour_val': 14, 'day_sel': 2,
        'role': 0, 'sex': 1, 'age': 35, 'used_belt': True, 'used_airbag': True,
        'speed_limit': 50, 'road_complexity_index': 2.0, 'reserved_lane_present': False,
        'my_vehicle': "Passenger Car", 'other_vehicle': "Passenger Car"
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


init_session_state()


# Helper to load scenarios
def load_scenario(type="normal"):
    if type == "high_risk":
        # Critical Situation: Elderly cyclist vs Truck at night/ice
        st.session_state['lighting_ordinal'] = 3  # Night Unlit
        st.session_state['weather_ordinal'] = 4  # Fog/Snow
        st.session_state['surface_quality_indicator'] = 2  # Ice
        st.session_state['time_of_day'] = 3  # Night
        st.session_state['hour_val'] = 3  # 3 AM
        st.session_state['day_sel'] = 5  # Saturday

        st.session_state['role'] = 0  # Driver
        st.session_state['sex'] = 1  # Male
        st.session_state['age'] = 82  # Elderly (High Risk)
        st.session_state['used_belt'] = False  # No Belt
        st.session_state['used_airbag'] = False  # No Airbag

        st.session_state['speed_limit'] = 110
        st.session_state['road_complexity_index'] = 8.0
        st.session_state['reserved_lane_present'] = False

        st.session_state['my_vehicle'] = "Bicycle / Scooter"  # Vulnerable
        st.session_state['other_vehicle'] = "Heavy Truck / Bus"  # Dangerous obstacle

    elif type == "medium_risk":
        # Medium Risk: Middle-aged motorcyclist vs Truck in rain
        st.session_state['lighting_ordinal'] = 2  # Night (Lit)
        st.session_state['weather_ordinal'] = 2  # Heavy Rain/Wind
        st.session_state['surface_quality_indicator'] = 1  # Wet
        st.session_state['time_of_day'] = 3  # Night
        st.session_state['hour_val'] = 22  # 10 PM
        st.session_state['day_sel'] = 4  # Friday

        st.session_state['role'] = 0  # Driver
        st.session_state['sex'] = 1  # Male
        st.session_state['age'] = 55  # Middle Aged (Higher vulnerability than 21)
        st.session_state['used_belt'] = True  # Helmet/Belt assumed
        st.session_state['used_airbag'] = False  # Moto often has no airbag

        st.session_state['speed_limit'] = 90
        st.session_state['road_complexity_index'] = 6.0
        st.session_state['reserved_lane_present'] = False

        st.session_state['my_vehicle'] = "Motorcycle"
        st.session_state['other_vehicle'] = "Heavy Truck / Bus"  # Increased delta vs Car

    else:  # Reset to Normal
        # Safe Situation: Standard car commute
        st.session_state['lighting_ordinal'] = 0
        st.session_state['weather_ordinal'] = 0
        st.session_state['surface_quality_indicator'] = 0
        st.session_state['time_of_day'] = 1
        st.session_state['hour_val'] = 14
        st.session_state['day_sel'] = 2
        st.session_state['role'] = 0
        st.session_state['sex'] = 1
        st.session_state['age'] = 30  # Slightly younger
        st.session_state['used_belt'] = True
        st.session_state['used_airbag'] = True
        st.session_state['speed_limit'] = 30  # Reduced to 30 km/h to ensure Low Risk
        st.session_state['road_complexity_index'] = 0.0  # Reduced complexity to minimum
        st.session_state['my_vehicle'] = "Passenger Car"
        st.session_state['other_vehicle'] = "Bicycle / Scooter"  # Colliding with lighter vehicle = safer for driver


# --- 5. Header & Controls ---
st.title("🚑 eCall AI - Injury Severity Prediction")
st.markdown("""
This dashboard predicts the severity of injuries in traffic accidents using a machine learning model.

* **Data Source:** **ONISR** (Observatoire national interministériel de la sécurité routière) - French accident data (2019-2022).
* **Model:** **CatBoost Classifier**, optimized for categorical features and imbalanced datasets.
* **Usage:** Configure the accident parameters below or use the quick-load buttons to test specific scenarios.
""")

if model is None or metadata is None:
    st.error("⚠️ Model files missing or corrupted.")
    st.stop()

features = metadata['feature_names']
cat_features = metadata['cat_features']

# Scenario Buttons
# Adjusted columns to [1, 1, 1, 5] to fit 3 buttons tightly
sc_col1, sc_col2, sc_col3, _ = st.columns([1, 1, 1, 5])
with sc_col1:
    if st.button("🛡️ Normal Scenario", use_container_width=True):
        load_scenario("normal")
        st.rerun()
with sc_col2:
    if st.button("🌩️ Medium Risk", use_container_width=True):
        load_scenario("medium_risk")
        st.rerun()
with sc_col3:
    if st.button("🔥 High Risk", use_container_width=True):
        load_scenario("high_risk")
        st.rerun()

# --- 6. Input Form ---
with st.form("prediction_form"):
    st.subheader("📝 Accident Report Simulation")
    col1, col2, col3 = st.columns(3)
    user_input = {}

    # --- COLUMN 1: Environment ---
    with col1:
        st.markdown("### 🌤 Environment")
        light_opts = {0: "Daylight", 1: "Twilight", 2: "Night (Lit)", 3: "Night (Unlit)"}
        user_input['lighting_ordinal'] = st.selectbox("Lighting", options=list(light_opts.keys()),
                                                      format_func=lambda x: light_opts[x], key='lighting_ordinal')

        weather_opts = {0: "Normal", 1: "Light Rain", 2: "Heavy Rain/Wind", 4: "Fog/Snow"}
        user_input['weather_ordinal'] = st.selectbox("Weather", options=list(weather_opts.keys()),
                                                     format_func=lambda x: weather_opts[x], key='weather_ordinal')

        surface_opts = {0: "Normal", 1: "Wet/Leaves", 2: "Ice/Oil"}
        user_input['surface_quality_indicator'] = st.selectbox("Road Surface", options=list(surface_opts.keys()),
                                                               format_func=lambda x: surface_opts[x],
                                                               key='surface_quality_indicator')

        st.markdown("### 🕒 Time")
        time_opts = {0: "Morning Rush", 1: "Midday", 2: "Evening Rush", 3: "Night"}
        user_input['time_of_day'] = st.selectbox("Time Window", options=list(time_opts.keys()),
                                                 format_func=lambda x: time_opts[x], key='time_of_day')

        # Calculate cyclical features
        hour_val = st.slider("Hour (0-23)", 0, 23, key='hour_val')
        h_sin, h_cos = get_cyclical_features(hour_val, 24)
        user_input['hour_sin'] = h_sin
        user_input['hour_cos'] = h_cos

        day_opts = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"}
        day_sel = st.selectbox("Day", options=list(day_opts.keys()), format_func=lambda x: day_opts[x], key='day_sel')
        user_input['day_of_week'] = day_sel
        d_sin, d_cos = get_cyclical_features(day_sel, 7)
        user_input['day_of_week_sin'] = d_sin
        user_input['day_of_week_cos'] = d_cos

    # --- COLUMN 2: Person ---
    with col2:
        st.markdown("### 👤 Person")
        role_opts = {0: "Driver", 1: "Passenger", 2: "Pedestrian"}
        user_input['role'] = st.selectbox("Role", options=list(role_opts.keys()), format_func=lambda x: role_opts[x],
                                          key='role')

        sex_opts = {1: "Male", 0: "Female", -1: "Unknown"}
        user_input['sex'] = st.selectbox("Sex", options=list(sex_opts.keys()), format_func=lambda x: sex_opts[x],
                                         key='sex')

        age_val = st.number_input("Age", 18, 99, key='age')
        user_input['age'] = age_val
        if age_val < 18:
            age_grp = 0
        elif age_val < 26:
            age_grp = 1
        elif age_val < 45:
            age_grp = 2
        elif age_val < 65:
            age_grp = 3
        else:
            age_grp = 4
        user_input['age_group'] = age_grp

        st.markdown("###### Safety")
        # Checkbox returns bool, model needs int (1/0)
        belt_bool = st.checkbox("Seatbelt?", key='used_belt')
        airbag_bool = st.checkbox("Airbag?", key='used_airbag')
        user_input['used_belt'] = 1 if belt_bool else 0
        user_input['used_airbag'] = 1 if airbag_bool else 0

    # --- COLUMN 3: Crash ---
    with col3:
        st.markdown("### 💥 Crash")
        user_input['speed_limit'] = st.slider("Speed Limit", 30, 130, step=10, key='speed_limit')
        user_input['road_complexity_index'] = st.slider("Road Complexity (0-10)", 0.0, 10.0,
                                                        key='road_complexity_index')

        st.markdown("###### Vehicles")
        # We use session state keys directly
        my_vehicle = st.selectbox("My Vehicle",
                                  ["Passenger Car", "Heavy Truck / Bus", "Motorcycle", "Bicycle / Scooter"],
                                  key='my_vehicle')
        other_vehicle = st.selectbox("Other Vehicle",
                                     ["Passenger Car", "Heavy Truck / Bus", "Motorcycle", "Pedestrian", "Unknown"],
                                     key='other_vehicle')

        score_me = get_impact_score(my_vehicle)
        score_other = get_impact_score(other_vehicle)

        user_input['impact_score'] = score_me
        user_input['impact_score_other'] = score_other
        user_input['impact_delta'] = score_me - score_other

        reserved_bool = st.checkbox("Reserved Lane?", key='reserved_lane_present')
        user_input['reserved_lane_present'] = 1 if reserved_bool else 0

    # Fill missing features
    for f in features:
        if f not in user_input:
            if f in cat_features:
                user_input[f] = -1
            else:
                user_input[f] = 0.0

    submit_btn = st.form_submit_button("🚨 Predict Severity")

# --- 7. Prediction Logic ---
if submit_btn:
    input_df = pd.DataFrame([user_input])
    input_df = input_df[features]
    for col in input_df.columns:
        if col in cat_features:
            input_df[col] = input_df[col].astype('category')

    try:
        # Predict
        probs = model.predict_proba(input_df)[0]
        p_safe, p_hosp, p_fatal = probs[0], probs[1], probs[2]

        # --- RESULTS SECTION ---
        st.divider()
        st.subheader("📊 Analysis Results")

        # Create 3 columns for Layout
        res_col1, res_col2 = st.columns([1, 2])

        with res_col1:
            # --- 1. SEVERITY GAUGE (Donut Chart) ---
            st.markdown("##### Severe Injury Risk")

            # Data for Gauge
            gauge_data = pd.DataFrame({
                'category': ['Severe Risk', 'Safe'],
                'value': [p_fatal, 1 - p_fatal]
            })

            # --- ADJUSTED THRESHOLDS for Imbalanced Data ---
            # Fatalities are rare events (<1% base rate).
            # Thresholds are tuned to detect relative risk increases.

            if p_fatal > 0.08:
                risk_color = '#F44336'  # RED
                risk_label = "HIGH RISK"
            elif p_fatal > 0.02 or p_hosp > 0.30:
                risk_color = '#FFC107'  # YELLOW
                risk_label = "MEDIUM RISK"
            else:
                risk_color = '#4CAF50'  # GREEN
                risk_label = "LOW RISK"

            base = alt.Chart(gauge_data).encode(
                theta=alt.Theta("value", stack=True)
            )

            # We fix the width/height to ensure it remains a circle (not stretched)
            pie = base.mark_arc(outerRadius=100, innerRadius=70).encode(
                color=alt.Color("category",
                                scale=alt.Scale(domain=['Severe Risk', 'Safe'], range=[risk_color, '#e0e0e0']),
                                legend=None),
                tooltip=["category", alt.Tooltip("value", format=".1%")]
            ).properties(width=250, height=250)

            text = base.mark_text(radius=0, size=24, color=risk_color, fontStyle="bold").encode(
                text=alt.value(f"{p_fatal:.1%}")
            ).properties(width=250, height=250)

            # Display WITHOUT use_container_width to keep aspect ratio
            # Use nested columns to center the chart horizontally
            sub_col_l, sub_col_m, sub_col_r = st.columns([1, 2, 1])
            with sub_col_m:
                st.altair_chart(pie + text, use_container_width=False)

            # Text Summary with Calibrated Thresholds
            if risk_label == "HIGH RISK":
                st.error("🔴 **HIGH RISK**\n\nPotential Fatality or Severe Injury.")
            elif risk_label == "MEDIUM RISK":
                st.warning("🟡 **MEDIUM RISK**\n\nHospitalization Likely.")
            else:
                st.success("🟢 **LOW RISK**\n\nLikely Minor Injuries.")

        with res_col2:
            # --- 2. HORIZONTAL PROBABILITY BAR ---
            st.markdown("##### Prediction Probabilities")

            chart_data = pd.DataFrame({
                'Outcome': ['Uninjured/Light', 'Hospitalized', 'Fatal'],
                'Probability': [p_safe, p_hosp, p_fatal]
            })

            # Define sorting order explicitly
            sort_order = ['Uninjured/Light', 'Hospitalized', 'Fatal']

            # Base chart
            bar_chart = alt.Chart(chart_data).mark_bar(cornerRadius=5, height=40).encode(
                x=alt.X('Probability', axis=alt.Axis(format='%', title='')),
                y=alt.Y('Outcome', sort=sort_order, title=''),
                # Explicitly map colors to categories
                color=alt.Color('Outcome', scale=alt.Scale(
                    domain=['Uninjured/Light', 'Hospitalized', 'Fatal'],
                    range=['#4CAF50', '#FFC107', '#F44336']
                ), legend=None),
                tooltip=['Outcome', alt.Tooltip('Probability', format='.1%')]
            ).properties(height=220)  # Added explicit height to fix layout

            # Text labels ON the bars
            text_chart = bar_chart.mark_text(
                align='left',
                baseline='middle',
                dx=5,
                color='black'
            ).encode(
                text=alt.Text('Probability', format='.1%')
            )

            st.altair_chart(bar_chart + text_chart, use_container_width=True)

            # --- 3. RISK FACTORS BREAKDOWN ---
            st.markdown("##### 🔍 Contextual Risk Factors")
            # Normalize inputs to 0-1 scale for visualization
            factors = pd.DataFrame({
                'Factor': ['Speed Limit', 'Road Complexity', 'Weather Severity', 'Lighting Conditions'],
                'Intensity': [
                    user_input['speed_limit'] / 130,
                    user_input['road_complexity_index'] / 10.0,
                    user_input['weather_ordinal'] / 4.0,
                    user_input['lighting_ordinal'] / 3.0
                ]
            })

            factor_chart = alt.Chart(factors).mark_bar(color='#5DADE2', height=20).encode(
                x=alt.X('Intensity', scale=alt.Scale(domain=[0, 1]), title="Relative Intensity (0=Low, 1=Max)"),
                y=alt.Y('Factor', sort=None),
                tooltip=['Factor', alt.Tooltip('Intensity', format='.2f')]
            ).properties(height=200)  # Added explicit height to fix layout

            st.altair_chart(factor_chart, use_container_width=True)

    except Exception as e:
        st.error(f"Prediction Error: {str(e)}")