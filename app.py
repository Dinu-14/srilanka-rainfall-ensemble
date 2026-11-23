from urllib.request import Request, urlopen
from pathlib import Path
from datetime import datetime, date, timedelta
from zoneinfo import ZoneInfo
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError
import json

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import truststore

# Use the operating system's trusted certificates
truststore.inject_into_ssl()


# ==================================================
# 1. PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="RainLens | Ensemble Intelligence",
    page_icon="🌧️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"

DATA_FILE = DATA_DIR / "SriLanka_Weather_25000.csv"
DATE_FILE = DATA_DIR / "SriLanka_Weather_Recent_25000.csv"

MODEL_FILES = {
    "Decision Tree": "decision_tree.joblib",
    "Bagging Classifier": "bagging.joblib",
    "Random Forest": "random_forest.joblib",
    "AdaBoost": "adaboost.joblib",
    "XGBoost": "xgboost.joblib",
    "Stacking Classifier": "stacking.joblib"
}

METRICS = [
    "F1-score",
    "Accuracy",
    "Precision",
    "Recall",
    "ROC-AUC"
]

COLORS = {
    "Baseline": "#94A3B8",
    "Bagging": "#38BDF8",
    "Boosting": "#A78BFA",
    "Stacking": "#2DD4BF"
}


# ==================================================
# 2. DASHBOARD THEME
# ==================================================

st.markdown("""
<style>

.stApp {
    background:
        radial-gradient(
            ellipse at 85% 5%,
            #103B59 0%,
            transparent 42%
        ),
        linear-gradient(
            135deg,
            #061321,
            #0A1B30
        );
    color: #E6F1FF;
}

[data-testid="stHeader"] {
    background: transparent;
}

#MainMenu,
[data-testid="stMainMenu"],
[data-testid="stToolbar"],
[data-testid="stDecoration"] {
    display: none !important;
}

.block-container {
    max-width: 1500px;
    padding: 0.65rem 1.25rem 0.8rem;
}

h1, h2, h3 {
    color: #EAF5FF;
}

[data-testid="stMetric"] {
    background: #102940;
    border: 1px solid #294A63;
    border-radius: 14px;
    padding: 12px 16px;
    min-height: 92px;
}

[data-testid="stMetricLabel"] p {
    color: #AEC5D8 !important;
}

[data-testid="stMetricValue"] {
    color: #67E8F9;
    font-weight: 750;
    font-size: 1.7rem;
}

[data-testid="stVerticalBlockBorderWrapper"] > div {
    border-color: #294A63 !important;
    border-radius: 15px;
}

.stButton button[kind="primary"] {
    background: linear-gradient(
        90deg,
        #087CAA,
        #0CA6A9
    );
    color: white;
    border: none;
    border-radius: 10px;
    font-weight: 700;
}

.stDownloadButton button {
    background: #14374E;
    color: #EAF5FF;
    border: 1px solid #37728C;
    border-radius: 10px;
}

[data-testid="stExpander"] {
    background: #102940;
    border: 1px solid #294A63;
    border-radius: 12px;
}

[data-testid="stWidgetLabel"] p,
[data-testid="stCaptionContainer"] p {
    color: #B7CEDF !important;
}

.header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 16px;
    margin-bottom: 15px;
}

.brand {
    color: white;
    font-size: 1.85rem;
    font-weight: 800;
    letter-spacing: -0.035em;
}

.subtitle {
    color: #A9C4D8;
    font-size: .85rem;
    margin-top: 3px;
}

.panel-title {
    color: #F2F8FF;
    font-size: 1.08rem;
    font-weight: 750;
    margin-bottom: 8px;
}

.result-card {
    background: #102B42;
    border: 1px solid #27506A;
    border-radius: 12px;
    padding: 17px;
    text-align: center;
}

.result-title {
    color: #EAF6FF;
    font-size: 1rem;
    font-weight: 700;
    margin-top: 7px;
}

.result-description {
    color: #B6D5E6;
    font-size: .84rem;
    margin-top: 5px;
}

@media (max-width: 750px) {
    .block-container {
        padding: .7rem;
    }

    .header {
        flex-wrap: wrap;
    }

    .brand {
        font-size: 1.45rem;
    }
}

</style>
""", unsafe_allow_html=True)


# ==================================================
# 3. LOAD DATA AND MODELS
# ==================================================

@st.cache_data(show_spinner=False)
def load_data():
    return pd.read_csv(DATA_FILE)


@st.cache_data(show_spinner=False)
def load_results():
    return (
        pd.read_csv(MODEL_DIR / "model_results.csv")
        .drop_duplicates("Model", keep="last")
    )


@st.cache_resource(show_spinner=False)
def load_model(name):
    # Load only trusted models created by this project.
    return joblib.load(
        MODEL_DIR / MODEL_FILES[name]
    )


@st.cache_data(show_spinner=False)
def load_reference():

    if not DATE_FILE.exists():
        return None

    reference = pd.read_csv(DATE_FILE)
    main = load_data()

    required = {
        "time",
        "city",
        "latitude",
        "longitude",
        "elevation",
        "rain_tomorrow"
    }

    if (
        len(reference) != len(main)
        or not required.issubset(reference.columns)
    ):
        return None

    if not np.array_equal(
        reference["rain_tomorrow"].to_numpy(),
        main["rain_tomorrow"].to_numpy()
    ):
        return None

    if not np.allclose(
        reference[["latitude", "longitude"]].to_numpy(),
        main[["latitude", "longitude"]].to_numpy()
    ):
        return None

    reference["time"] = pd.to_datetime(
        reference["time"],
        errors="coerce"
    )

    if reference["time"].isna().any():
        return None

    reference["_row_id"] = np.arange(len(reference))

    return reference


@st.cache_data(show_spinner=False)
def load_test_data():

    reference = load_reference()

    if reference is None:
        return None

    unique_dates = sorted(reference["time"].unique())

    split_date = pd.Timestamp(
        unique_dates[int(len(unique_dates) * 0.8)]
    )

    return reference.loc[
        reference["time"] >= split_date
    ].copy()


# ==================================================
# 4. OPEN-METEO FUTURE WEATHER API
# ==================================================

@st.cache_data(ttl=900, show_spinner=False)
def fetch_forecast(latitude, longitude):

    daily_variables = [
        "weather_code",
        "temperature_2m_max",
        "temperature_2m_min",
        "apparent_temperature_max",
        "apparent_temperature_min",
        "shortwave_radiation_sum",
        "rain_sum",
        "precipitation_hours",
        "wind_speed_10m_max",
        "wind_gusts_10m_max",
        "wind_direction_10m_dominant",
        "et0_fao_evapotranspiration"
    ]

    parameters = {
        "latitude": float(latitude),
        "longitude": float(longitude),
        "daily": ",".join(daily_variables),
        "hourly": "temperature_2m,apparent_temperature",
        "forecast_days": 16,
        "timezone": "Asia/Colombo",
        "temperature_unit": "celsius",
        "wind_speed_unit": "kmh",
        "precipitation_unit": "mm"
    }

    url = (
        "https://api.open-meteo.com/v1/forecast?"
        + urlencode(parameters)
    )

    request = Request(
        url,
        headers={
            "User-Agent": "RainLens-Assignment/1.0"
        }
    )

    with urlopen(request, timeout=20) as response:
        forecast = json.load(response)

    if (
        forecast.get("error")
        or "daily" not in forecast
        or "hourly" not in forecast
    ):
        raise ValueError(
            forecast.get(
                "reason",
                "Weather forecast unavailable"
            )
        )

    return forecast


# ==================================================
# 5. CONSTRUCT FUTURE MODEL INPUT FEATURES
# ==================================================

def build_future_inputs(
    forecast,
    weather_day,
    city,
    location,
    feature_names
):

    daily = forecast["daily"]
    hourly = forecast["hourly"]

    day_string = weather_day.isoformat()

    if day_string not in daily["time"]:
        raise ValueError(
            "Date is outside the forecast period."
        )

    day_index = daily["time"].index(day_string)

    def daily_value(variable):

        value = daily[variable][day_index]

        if (
            value is None
            or not np.isfinite(float(value))
        ):
            raise ValueError(
                f"Missing forecast variable: {variable}"
            )

        return float(value)

    hourly_dates = pd.to_datetime(
        hourly["time"]
    ).date

    day_mask = np.asarray(
        hourly_dates == weather_day
    )

    if day_mask.sum() != 24:
        raise ValueError(
            "Insufficient hourly forecast readings."
        )

    def hourly_mean(variable):

        values = np.asarray(
            hourly[variable],
            dtype=float
        )[day_mask]

        if not np.isfinite(values).all():
            raise ValueError(
                f"Missing hourly variable: {variable}"
            )

        return float(values.mean())

    month = weather_day.month

    values = {
        "latitude": float(location["latitude"]),
        "longitude": float(location["longitude"]),
        "elevation": float(location["elevation"]),

        "weathercode": daily_value("weather_code"),

        "temperature_2m_max": daily_value(
            "temperature_2m_max"
        ),
        "temperature_2m_min": daily_value(
            "temperature_2m_min"
        ),
        "temperature_2m_mean": hourly_mean(
            "temperature_2m"
        ),

        "apparent_temperature_max": daily_value(
            "apparent_temperature_max"
        ),
        "apparent_temperature_min": daily_value(
            "apparent_temperature_min"
        ),
        "apparent_temperature_mean": hourly_mean(
            "apparent_temperature"
        ),

        "shortwave_radiation_sum": daily_value(
            "shortwave_radiation_sum"
        ),
        "rain_sum": daily_value("rain_sum"),
        "precipitation_hours": daily_value(
            "precipitation_hours"
        ),

        "windspeed_10m_max": daily_value(
            "wind_speed_10m_max"
        ),
        "windgusts_10m_max": daily_value(
            "wind_gusts_10m_max"
        ),
        "winddirection_10m_dominant": daily_value(
            "wind_direction_10m_dominant"
        ),

        "et0_fao_evapotranspiration": daily_value(
            "et0_fao_evapotranspiration"
        ),

        "month_sin": float(
            np.sin(2 * np.pi * month / 12)
        ),
        "month_cos": float(
            np.cos(2 * np.pi * month / 12)
        )
    }

    city_feature = f"city_{city}"

    if city_feature not in feature_names:
        raise ValueError(
            "Selected city is not in model training features."
        )

    for feature in feature_names:

        if feature.startswith("city_"):
            values[feature] = int(
                feature == city_feature
            )

    missing = [
        feature
        for feature in feature_names
        if feature not in values
    ]

    if missing:
        raise ValueError(
            "Unmapped features: " + ", ".join(missing)
        )

    inputs = pd.DataFrame(
        [{
            feature: values[feature]
            for feature in feature_names
        }],
        columns=feature_names
    )

    if not np.isfinite(
        inputs.to_numpy(dtype=float)
    ).all():
        raise ValueError(
            "Invalid model input values."
        )

    return inputs


# ==================================================
# 6. INITIALIZE DATA
# ==================================================

try:

    df = load_data()
    results = load_results()

    feature_names = joblib.load(
        MODEL_DIR / "feature_names.joblib"
    )

    if (
        len(feature_names) != 49
        or any(
            feature not in df.columns
            for feature in feature_names
        )
    ):
        raise ValueError(
            "Model feature mismatch."
        )

except (OSError, ValueError, KeyError) as error:

    st.error(
        f"Unable to load project files: {error}"
    )

    st.stop()


# ==================================================
# 7. DASHBOARD HEADER
# ==================================================

st.markdown(
    '<div class="header">'
    '<div>'
    '<div class="brand">🌧️ RainLens</div>'
    '<div class="subtitle">'
    'Sri Lankan Rainfall Prediction · Ensemble Learning'
    '</div>'
    '</div>'
    '</div>',
    unsafe_allow_html=True
)


# ==================================================
# 8. SUMMARY CARDS
# ==================================================

best = results.loc[
    results["F1-score"].idxmax()
]

a, b, c, d = st.columns(
    4,
    gap="small"
)

a.metric(
    "Weather Records",
    f"{len(df):,}"
)

b.metric(
    "Input Features",
    len(feature_names)
)

c.metric(
    "Trained Models",
    len(MODEL_FILES)
)

d.metric(
    "Best F1-score",
    f"{best['F1-score']:.2%}"
)


# ==================================================
# 9. MODEL PERFORMANCE CHART
# ==================================================

def create_performance_chart(data, metric):

    ordered = data.sort_values(
        metric,
        ascending=True
    )

    fig = go.Figure()

    for technique in ordered["Technique"].unique():

        subset = ordered.loc[
            ordered["Technique"] == technique
        ]

        fig.add_trace(
            go.Bar(
                x=subset[metric] * 100,
                y=subset["Model"],
                orientation="h",
                name=technique,
                marker_color=COLORS.get(
                    technique,
                    "#94A3B8"
                ),
                text=[
                    f"{value:.2%}"
                    for value in subset[metric]
                ],
                textposition="outside",
                cliponaxis=False,
                hovertemplate=(
                    "%{y}: %{x:.2f}%"
                    "<extra></extra>"
                )
            )
        )

    fig.update_layout(
        height=285,
        barmode="overlay",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(
            color="#D8EAF5",
            size=12
        ),
        margin=dict(
            l=5,
            r=65,
            t=14,
            b=12
        ),
        xaxis=dict(
            range=[0, 105],
            gridcolor="#23445B",
            showgrid=True
        ),
        yaxis=dict(
            title="",
            tickfont=dict(
                color="#D8EAF5"
            )
        ),
        legend=dict(
            orientation="h",
            y=1.16,
            font=dict(
                color="#D8EAF5",
                size=10
            )
        )
    )

    return fig


# ==================================================
# 10. MAIN DASHBOARD LAYOUT
# ==================================================

left, right = st.columns(
    [1.3, 1],
    gap="medium"
)


# ==================================================
# LEFT COLUMN — MODEL PERFORMANCE
# ==================================================

with left:

    with st.container(border=True):

        st.markdown(
            '<div class="panel-title">'
            '📊 Model Performance'
            '</div>',
            unsafe_allow_html=True
        )

        metric = st.selectbox(
            "Compare by",
            METRICS,
            label_visibility="collapsed",
            key="metric_selector"
        )

        st.plotly_chart(
            create_performance_chart(
                results,
                metric
            ),
            use_container_width=True,
            key="performance_chart"
        )

        winner = results.loc[
            results[metric].idxmax()
        ]

        st.markdown(
            f"🏆 **Top {metric}:** "
            f"{winner['Model']} "
            f"({winner[metric]:.2%})"
        )

    # Dataset Explorer below model performance
    with st.container(border=True):

        st.markdown(
            "**📁 Dataset Explorer**"
        )

        preview_col, download_col = st.columns(
            2,
            gap="small"
        )

        with preview_col:

            with st.popover(
                "🔍 Preview Dataset",
                use_container_width=True
            ):

                preview_rows = st.selectbox(
                    "Rows",
                    [5, 10, 20, 50],
                    key="preview_rows"
                )

                st.dataframe(
                    df.head(preview_rows),
                    hide_index=True,
                    use_container_width=True
                )

        with download_col:

            st.download_button(
                "⬇ Download CSV",
                data=DATA_FILE.read_bytes(),
                file_name="SriLanka_Weather_25000.csv",
                mime="text/csv",
                use_container_width=True,
                key="download_dataset"
            )


# ==================================================
# RIGHT COLUMN — RAINFALL PREDICTION
# ==================================================

with right:

    with st.container(border=True):

        st.markdown(
            '<div class="panel-title">'
            '🌦️ Rainfall Prediction'
            '</div>',
            unsafe_allow_html=True
        )

        mode = st.radio(
            "Prediction Mode",
            [
                "Historical Test",
                "Future Forecast"
            ],
            horizontal=True,
            label_visibility="collapsed",
            key="prediction_mode"
        )

        model_name = st.selectbox(
            "Model",
            list(MODEL_FILES),
            index=5,
            key="model_selector"
        )

        input_features = None
        actual = None
        signature = None
        info_text = ""

        # ------------------------------------------
        # HISTORICAL TEST MODE
        # ------------------------------------------

        if mode == "Historical Test":

            test_data = load_test_data()

            if test_data is None:

                st.warning(
                    "A matching date-preserving CSV "
                    "is required for historical testing."
                )

            else:

                cities = sorted(
                    test_data["city"]
                    .dropna()
                    .unique()
                    .tolist()
                )

                city = st.selectbox(
                    "City",
                    cities,
                    key="historical_city"
                )

                records = (
                    test_data.loc[
                        test_data["city"] == city
                    ]
                    .reset_index(drop=True)
                )

                selected_index = st.selectbox(
                    "Test Date",
                    range(len(records)),
                    format_func=lambda i: (
                        records.iloc[i]["time"]
                        .strftime("%d %b %Y")
                    ),
                    key="historical_date"
                )

                record = records.iloc[
                    selected_index
                ]

                row_id = int(
                    record["_row_id"]
                )

                input_features = df.iloc[
                    [row_id]
                ][feature_names]

                actual = int(
                    df.iloc[row_id]["rain_tomorrow"]
                )

                signature = (
                    "historical",
                    model_name,
                    row_id
                )

                info_text = (
                    "Recorded historical test observation."
                )

        # ------------------------------------------
        # FUTURE FORECAST MODE
        # ------------------------------------------

        else:

            reference = load_reference()

            if reference is None:

                st.warning(
                    "A matching date-preserving CSV "
                    "is required for city coordinates."
                )

            else:

                cities = sorted(
                    city
                    for city in reference[
                        "city"
                    ].dropna().unique()
                    if f"city_{city}" in feature_names
                )

                city = st.selectbox(
                    "City",
                    cities,
                    key="future_city"
                )

                location = reference.loc[
                    reference["city"] == city
                ].iloc[0]

                today = datetime.now(
                    ZoneInfo("Asia/Colombo")
                ).date()

                st.caption(
                    "Internet required · Open-Meteo weather data"
                )

                try:

                    forecast = fetch_forecast(
                        location["latitude"],
                        location["longitude"]
                    )

                    available_days = [
                        date.fromisoformat(value)
                        for value in forecast["daily"]["time"]
                    ]

                    target_dates = sorted({
                        weather_day + timedelta(days=1)
                        for weather_day in available_days
                        if (
                            weather_day + timedelta(days=1)
                        ) > today
                    })

                    if not target_dates:
                        raise ValueError(
                            "No future dates are available."
                        )

                    target_date = st.selectbox(
                        "Future Rainfall Date",
                        target_dates,
                        format_func=lambda day: (
                            day.strftime("%d %b %Y")
                        ),
                        key="future_date"
                    )

                    weather_day = (
                        target_date - timedelta(days=1)
                    )

                    signature = (
                        "future",
                        model_name,
                        city,
                        target_date.isoformat()
                    )

                    info_text = (
                        f"Weather inputs: "
                        f"{weather_day:%d %b %Y} · "
                        f"Rain target: "
                        f"{target_date:%d %b %Y}"
                    )

                except (
                    HTTPError,
                    URLError,
                    TimeoutError,
                    ValueError,
                    KeyError,
                    TypeError
                ) as error:

                    st.warning(
                        f"Future forecast unavailable: {error}"
                    )

        # ------------------------------------------
        # SHARED PREDICTION CONTROLS
        # ------------------------------------------

        if signature is not None:

            threshold = st.slider(
                "Decision Threshold",
                min_value=0.10,
                max_value=0.90,
                value=0.50,
                step=0.05,
                key="threshold_slider"
            )

            signature = (
                *signature,
                threshold
            )

            st.caption(info_text)

            # Clear old result if inputs change
            if (
                st.session_state.get(
                    "prediction_signature"
                ) != signature
            ):
                st.session_state.pop(
                    "prediction_result",
                    None
                )

            predict_button = st.button(
                "🌧️ Predict Rainfall",
                type="primary",
                use_container_width=True,
                key="predict_button"
            )

            if predict_button:

                try:

                    if mode == "Future Forecast":

                        input_features = build_future_inputs(
                            forecast,
                            weather_day,
                            city,
                            location,
                            feature_names
                        )

                    model = load_model(
                        model_name
                    )

                    probability = float(
                        model.predict_proba(
                            input_features
                        )[0, 1]
                    )

                    st.session_state[
                        "prediction_result"
                    ] = {
                        "probability": probability,
                        "actual": actual
                    }

                    st.session_state[
                        "prediction_signature"
                    ] = signature

                except (
                    KeyError,
                    ValueError,
                    TypeError,
                    IndexError,
                    OSError
                ) as error:

                    st.error(
                        f"Prediction failed: {error}"
                    )

            # --------------------------------------
            # PREDICTION RESULTS
            # --------------------------------------

            result = st.session_state.get(
                "prediction_result"
            )

            if result is None:

                st.markdown(
                    '<div class="result-card">'
                    '<div style="font-size:1.8rem">🌦️</div>'
                    '<div class="result-title">'
                    'Ready to Predict'
                    '</div>'
                    '<div class="result-description">'
                    'Choose your inputs and press '
                    'Predict Rainfall.'
                    '</div>'
                    '</div>',
                    unsafe_allow_html=True
                )

            else:

                probability = result[
                    "probability"
                ]

                predicted = int(
                    probability >= threshold
                )

                st.metric(
                    "Estimated Rain Probability",
                    f"{probability:.1%}"
                )

                st.progress(
                    float(
                        np.clip(
                            probability,
                            0,
                            1
                        )
                    )
                )

                if predicted == 1:

                    st.info(
                        "🌧️ Rain Predicted"
                    )

                else:

                    st.success(
                        "☀️ No Rain Predicted"
                    )

                if result["actual"] is not None:

                    actual_value = result["actual"]

                    actual_text = (
                        "Rain"
                        if actual_value == 1
                        else "No Rain"
                    )

                    st.caption(
                        f"Recorded Outcome: {actual_text}"
                    )

                    if predicted == actual_value:

                        st.success(
                            "✓ Correct Classification"
                        )

                    else:

                        st.warning(
                            "Prediction differs from "
                            "the recorded outcome."
                        )

                else:

                    st.caption(
                        "Experimental forecast using "
                        "predicted weather measurements."
                    )

                    st.info(
    "🌦️ AI-Powered Rainfall Forecast: "
    "This prediction is generated using forecasted "
    "weather conditions from Open-Meteo and the "
    "selected ensemble learning model."
)


# ==================================================
# 11. FOOTER
# ==================================================

st.caption(
    "RainLens · Sri Lankan Weather Data · "
    "Bagging | Boosting | Stacking"
)

st.caption(
    "Historical evaluation and experimental "
    "future predictions. Prediction-time feature "
    "availability and time-aware stacking validation "
    "require review. AI-assisted tools supported "
    "coding and dashboard development."
)
