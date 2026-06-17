# Hotel Booking Cancellation Risk Predictor

An end-to-end machine learning application that estimates the probability of a hotel booking being cancelled.

The application was developed using an XGBoost classification pipeline and deployed through a Streamlit user interface. It allows hotel or reservations staff to enter booking information and receive a cancellation-risk probability, risk category and follow-up recommendation.

## Live Application

The deployed Streamlit application link will be added here after deployment.

## Application Preview

### Booking Input Form

![Hotel cancellation application form](screenshots/app_form.png)

### Prediction Result

![Hotel cancellation prediction result](screenshots/prediction_result.png)

## Business Problem

Booking cancellations can affect occupancy planning, revenue forecasting and hotel operations. This application demonstrates how historical booking data can be used to identify reservations that may require proactive confirmation or follow-up.

The tool is intended to support staff decision-making rather than automatically take action against a booking.

## Application Features

* Interactive Streamlit booking form
* Real-time cancellation probability
* Low, Moderate, High and Very High risk categories
* Operational follow-up flag
* Suggested business action based on predicted risk
* Expandable view of the submitted booking details
* Complete preprocessing and prediction pipeline

## Machine Learning Model

The application uses an XGBoost classifier trained on booking information that would normally be available when a reservation is created.

The application-friendly model uses features including:

* hotel type
* lead time
* arrival month
* stay duration
* number of guests
* meal package
* market segment
* distribution channel
* previous booking history
* reserved room type
* booking changes
* deposit type
* customer type
* average daily rate
* parking requirements
* special requests

## Model Performance

At the default classification threshold of 0.50, the model achieved:

* Accuracy: 80.95%
* Precision: 72.34%
* Recall: 49.83%
* F1-score: 59.01%
* ROC-AUC: 84.25%

Because the application is intended to screen bookings for possible follow-up, threshold tuning was performed.

An operational threshold of **0.35** was selected, producing:

* Accuracy: 79.39%
* Precision: 62.18%
* Recall: 64.10%
* F1-score: 63.13%
* ROC-AUC: 84.25%

Lowering the threshold improved the model's ability to identify actual cancellations while maintaining reasonable overall accuracy and precision.

## Technology Stack

* Python
* Streamlit
* pandas
* scikit-learn
* XGBoost
* joblib

## Project Structure

```text
Hotel-Cancellation-ML-App/
├── app.py
├── hotel_cancellation_app_model.joblib
├── requirements.txt
├── README.md
├── .gitignore
└── screenshots/
    ├── app_form.png
    └── prediction_result.png
```

## Run the Application Locally

Clone the repository:

```bash
git clone https://github.com/akshay040/Hotel-Cancellation-ML-App.git
```

Move into the project folder:

```bash
cd Hotel-Cancellation-ML-App
```

Install the required packages:

```bash
python -m pip install -r requirements.txt
```

Start the application:

```bash
python -m streamlit run app.py
```

Open the local Streamlit address displayed in the terminal.

## Model Limitations

* Predictions are based on historical booking patterns and do not guarantee that a booking will or will not be cancelled.
* The dataset may not represent every hotel, location or customer population.
* Some cancellation behaviour may be affected by external factors that are not included in the dataset.
* The operational threshold was chosen to improve recall and create a useful risk-screening tool.
* The application should support, not replace, staff judgement and hotel policy.

## Related Analysis Project

The original exploratory analysis, feature engineering and machine learning comparison project can be found in my Hotel Booking Cancellation Analysis repository.

## Author

**Akshay Kumar**

