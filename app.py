from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


# ---------------------------------------------------------
# APPLICATION CONFIGURATION
# ---------------------------------------------------------

APP_FOLDER = Path(__file__).resolve().parent
MODEL_PATH = APP_FOLDER / "hotel_cancellation_app_model.joblib"

# Selected during threshold tuning:
# Accuracy: 79.39%
# Precision: 62.18%
# Recall: 64.10%
# F1-score: 63.13%
PREDICTION_THRESHOLD = 0.35


# ---------------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------------

@st.cache_resource
def load_model():
    """Load and cache the complete preprocessing + XGBoost pipeline."""

    if not MODEL_PATH.exists():
        st.error(
            "Model file not found. Make sure "
            "'hotel_cancellation_app_model.joblib' is in the "
            "same folder as app.py."
        )
        st.stop()

    try:
        return joblib.load(MODEL_PATH)
    except Exception as error:
        st.error(f"Unable to load the model: {error}")
        st.stop()


model = load_model()


# ---------------------------------------------------------
# PAGE SETUP
# ---------------------------------------------------------

st.set_page_config(
    page_title="Hotel Cancellation Risk Predictor",
    page_icon="🏨",
    layout="wide"
)

st.title("🏨 Hotel Booking Cancellation Risk Predictor")

st.write(
    "Enter the booking information below to estimate the probability "
    "that the reservation will be cancelled."
)

st.info(
    "The application uses an XGBoost classification pipeline trained "
    "on historical hotel booking data. An operational threshold of "
    "35% is used to flag bookings for possible follow-up."
)


# ---------------------------------------------------------
# BOOKING FORM
# ---------------------------------------------------------

with st.form("booking_form"):

    left_column, right_column = st.columns(2)

    with left_column:

        st.subheader("Booking and stay details")

        hotel = st.selectbox(
            "Hotel type",
            ["City Hotel", "Resort Hotel"]
        )

        lead_time = st.number_input(
            "Lead time in days",
            min_value=0,
            max_value=800,
            value=30,
            help="Number of days between booking and arrival."
        )

        arrival_date_month = st.selectbox(
            "Arrival month",
            [
                "January",
                "February",
                "March",
                "April",
                "May",
                "June",
                "July",
                "August",
                "September",
                "October",
                "November",
                "December"
            ]
        )

        stays_in_weekend_nights = st.number_input(
            "Weekend nights",
            min_value=0,
            max_value=20,
            value=1
        )

        stays_in_week_nights = st.number_input(
            "Weekday nights",
            min_value=0,
            max_value=50,
            value=2
        )

        adults = st.number_input(
            "Adults",
            min_value=1,
            max_value=20,
            value=2
        )

        children = st.number_input(
            "Children",
            min_value=0,
            max_value=10,
            value=0
        )

        babies = st.number_input(
            "Babies",
            min_value=0,
            max_value=10,
            value=0
        )

        meal = st.selectbox(
            "Meal package",
            ["BB", "HB", "FB", "SC", "Undefined"],
            help=(
                "BB: Bed & Breakfast, HB: Half Board, "
                "FB: Full Board, SC: Self Catering."
            )
        )

        market_segment = st.selectbox(
            "Market segment",
            [
                "Online TA",
                "Offline TA/TO",
                "Direct",
                "Corporate",
                "Groups",
                "Complementary",
                "Aviation",
                "Undefined"
            ]
        )

        distribution_channel = st.selectbox(
            "Distribution channel",
            [
                "TA/TO",
                "Direct",
                "Corporate",
                "GDS",
                "Undefined"
            ]
        )

    with right_column:

        st.subheader("Guest, payment and engagement details")

        is_repeated_guest = st.selectbox(
            "Is this a repeated guest?",
            options=[0, 1],
            format_func=lambda value: (
                "Yes" if value == 1 else "No"
            )
        )

        previous_cancellations = st.number_input(
            "Previous cancellations",
            min_value=0,
            max_value=30,
            value=0
        )

        previous_bookings_not_canceled = st.number_input(
            "Previous completed bookings",
            min_value=0,
            max_value=100,
            value=0
        )

        reserved_room_type = st.selectbox(
            "Reserved room type",
            ["A", "B", "C", "D", "E", "F", "G", "H", "L", "P"]
        )

        booking_changes = st.number_input(
            "Number of booking changes",
            min_value=0,
            max_value=30,
            value=0
        )

        deposit_type = st.selectbox(
            "Deposit type",
            [
                "No Deposit",
                "Non Refund",
                "Refundable"
            ]
        )

        customer_type = st.selectbox(
            "Customer type",
            [
                "Transient",
                "Transient-Party",
                "Contract",
                "Group"
            ]
        )

        adr = st.number_input(
            "Average daily rate",
            min_value=0.0,
            max_value=1000.0,
            value=100.0,
            step=5.0,
            help="Average price paid per occupied room per day."
        )

        required_car_parking_spaces = st.number_input(
            "Required parking spaces",
            min_value=0,
            max_value=5,
            value=0
        )

        total_of_special_requests = st.number_input(
            "Number of special requests",
            min_value=0,
            max_value=10,
            value=0
        )

    submitted = st.form_submit_button(
        "Predict cancellation risk",
        use_container_width=True
    )


# ---------------------------------------------------------
# PREDICTION
# ---------------------------------------------------------

if submitted:

    input_data = pd.DataFrame(
        [{
            "hotel": hotel,
            "lead_time": lead_time,
            "arrival_date_month": arrival_date_month,
            "stays_in_weekend_nights": stays_in_weekend_nights,
            "stays_in_week_nights": stays_in_week_nights,
            "adults": adults,
            "children": children,
            "babies": babies,
            "meal": meal,
            "market_segment": market_segment,
            "distribution_channel": distribution_channel,
            "is_repeated_guest": is_repeated_guest,
            "previous_cancellations": previous_cancellations,
            "previous_bookings_not_canceled":
                previous_bookings_not_canceled,
            "reserved_room_type": reserved_room_type,
            "booking_changes": booking_changes,
            "deposit_type": deposit_type,
            "customer_type": customer_type,
            "adr": adr,
            "required_car_parking_spaces":
                required_car_parking_spaces,
            "total_of_special_requests":
                total_of_special_requests
        }]
    )

    try:
        cancellation_probability = float(
            model.predict_proba(input_data)[0][1]
        )

        predicted_cancellation = int(
            cancellation_probability >= PREDICTION_THRESHOLD
        )

        if cancellation_probability < 0.20:
            risk_level = "Low"
        elif cancellation_probability < 0.35:
            risk_level = "Moderate"
        elif cancellation_probability < 0.60:
            risk_level = "High"
        else:
            risk_level = "Very High"

        st.divider()
        st.subheader("Prediction result")

        probability_column, risk_column, flag_column = st.columns(3)

        with probability_column:
            st.metric(
                "Cancellation probability",
                f"{cancellation_probability:.1%}"
            )

        with risk_column:
            st.metric(
                "Risk level",
                risk_level
            )

        with flag_column:
            st.metric(
                "Follow-up flag",
                "Yes" if predicted_cancellation == 1 else "No"
            )

        st.progress(cancellation_probability)

        if predicted_cancellation == 1:
            st.warning(
                "This booking is above the selected 35% operational "
                "threshold and has been flagged for possible follow-up."
            )
        else:
            st.success(
                "This booking is below the selected 35% operational "
                "threshold."
            )

        if risk_level == "Very High":
            st.error(
                "Suggested action: confirm the reservation and review "
                "the booking details, payment conditions and guest history."
            )
        elif risk_level == "High":
            st.warning(
                "Suggested action: consider sending a booking confirmation "
                "or reminder before arrival."
            )
        elif risk_level == "Moderate":
            st.info(
                "Suggested action: normal monitoring may be sufficient."
            )
        else:
            st.success(
                "Suggested action: no additional intervention is currently "
                "suggested."
            )

        with st.expander("View submitted booking data"):
            st.dataframe(
                input_data,
                use_container_width=True,
                hide_index=True
            )

        st.caption(
            "This prediction is a statistical risk estimate based on "
            "historical data. It should support, not replace, staff "
            "judgement or hotel policy."
        )

    except Exception as error:
        st.error(f"Prediction failed: {error}")