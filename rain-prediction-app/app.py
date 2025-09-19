import streamlit as st
import pandas as pd
import tensorflow as tf
import joblib
# ================================Load Model , Scaler, preprocess input========================
@st.cache_resource
def load_model_and_scaler():
    model = tf.keras.models.load_model("rain_model_tf.keras")
    scaler = joblib.load("scaler.pkl")
    feature_names = joblib.load("features.pkl")
    return model, scaler, feature_names

model, scaler, feature_names = load_model_and_scaler()

def preprocess_input(df, feature_names):
    
    if "Date" in df.columns:
        df = df.drop("Date", axis=1)

    cat_cols =["Location", "WindGustDir", "WindDir9am", "WindDir3pm", "RainToday"]
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True, dtype=int)

    for col in feature_names:
        if col not in df.columns:
            df[col] = 0

    df = df[feature_names]
    return df


# ======================Interface=====================
try:
    st.title("🌦️ Rain Prediction App")
    with st.expander("🌡️ Temperature & Sun"):
        MinTemp = st.number_input("Minimum Temperature (°C)", -10.0, 40.0, 15.0)
        MaxTemp = st.number_input("Maximum Temperature (°C)", -10.0, 50.0, 25.0)
        Sunshine = st.number_input("Sunshine (hours)", 0.0, 15.0, 7.0)
        Temp9am = st.number_input("Temperature 9am (°C)", -10.0, 45.0, 18.0)
        Temp3pm = st.number_input("Temperature 3pm (°C)", -10.0, 45.0, 24.0)

    with st.expander("💧 Rain & Clouds"):
        Rainfall = st.number_input("Rainfall (mm)", 0.0, 200.0, 0.0)
        Evaporation = st.number_input("Evaporation (mm)", 0.0, 100.0, 5.0)
        Cloud9am = st.slider("Cloud 9am (oktas)", 0, 9, 4)
        Cloud3pm = st.slider("Cloud 3pm (oktas)", 0, 9, 4)

    with st.expander("💨 Wind"):
        WindGustSpeed = st.number_input("Wind Gust Speed (km/h)", 0.0, 150.0, 35.0)
        WindSpeed9am = st.number_input("Wind Speed 9am (km/h)", 0.0, 100.0, 10.0)
        WindSpeed3pm = st.number_input("Wind Speed 3pm (km/h)", 0.0, 100.0, 15.0)

        wind_dirs = ["N","S","E","W","NE","NW","SE","SW",
                    "NNE","NNW","ENE","ESE","SSE","SSW","WNW","WSW"]
        WindGustDir = st.selectbox("Wind Gust Direction", wind_dirs, index=wind_dirs.index("W"))
        WindDir9am = st.selectbox("Wind Direction 9am", wind_dirs, index=wind_dirs.index("N"))
        WindDir3pm = st.selectbox("Wind Direction 3pm", wind_dirs, index=wind_dirs.index("SE"))

    with st.expander("🌫️ Humidity & Pressure"):
        Humidity9am = st.slider("Humidity 9am (%)", 0, 100, 60)
        Humidity3pm = st.slider("Humidity 3pm (%)", 0, 100, 50)
        Pressure9am = st.number_input("Pressure 9am (hPa)", 900.0, 1100.0, 1012.0)
        Pressure3pm = st.number_input("Pressure 3pm (hPa)", 900.0, 1100.0, 1010.0)

    with st.expander("📍 Location"):
        locations = [
            "Canberra","Sydney","Perth","Darwin","Hobart","Brisbane","Adelaide","Bendigo","Townsville",
            "AliceSprings","MountGambier","Ballarat","Launceston","Albany","Albury","MelbourneAirport",
            "PerthAirport","Mildura","SydneyAirport","Nuriootpa","Sale","Watsonia","Tuggeranong","Portland",
            "Woomera","Cairns","Cobar","Wollongong","GoldCoast","WaggaWagga","Penrith","NorfolkIsland",
            "Newcastle","SalmonGums","CoffsHarbour","Witchcliffe","Richmond","Dartmoor","NorahHead",
            "BadgerysCreek","MountGinini","Moree","Walpole","PearceRAAF","Williamtown","Melbourne",
            "Nhil","Katherine","Uluru"
        ]
        Location = st.selectbox("Location", locations, index=locations.index("Canberra"))
        RainToday = st.selectbox("Rain Today", ["No","Yes"], index=0)
    # Convert variables to Data Frame
    input_data = pd.DataFrame({
    
        "Location": [Location],
        "MinTemp": [MinTemp],
        "MaxTemp": [MaxTemp],
        "Rainfall": [Rainfall],
        "Evaporation": [Evaporation],
        "Sunshine": [Sunshine],
        "WindGustDir": [WindGustDir],
        "WindGustSpeed": [WindGustSpeed],
        "WindDir9am": [WindDir9am],
        "WindDir3pm": [WindDir3pm],
        "WindSpeed9am": [WindSpeed9am],
        "WindSpeed3pm": [WindSpeed3pm],
        "Humidity9am": [Humidity9am],
        "Humidity3pm": [Humidity3pm],
        "Pressure9am": [Pressure9am],
        "Pressure3pm": [Pressure3pm],
        "Cloud9am": [Cloud9am],
        "Cloud3pm": [Cloud3pm],
        "Temp9am": [Temp9am],
        "Temp3pm": [Temp3pm],
        "RainToday": [RainToday],
    })

    # input_data = pd.get_dummies(input_data, drop_first=True, dtype=int)
    if st.button("Predict"):
        #preprocessing
        input_processed =preprocess_input(input_data.copy(), feature_names)

        #Scale
        data_scaled = scaler.transform(input_processed)

        #Prediction
        prediction = model.predict(data_scaled)[0][0]
        probability = prediction * 100

        st.write(f"🔮 Probability of Rain Tomorrow: {probability:.2f}%")

        if prediction > 0.5:
            st.success("🌧️ Yes rain tomorrow")
        else:
            st.info("☀️ No rain expected tomorrow")

except Exception as e:
    st.error(f" Something went wrong: {e}")