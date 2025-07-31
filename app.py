import streamlit as st
import numpy as np
import pandas as pd
import joblib

model = joblib.load('model.pkl')
model_features = ['Temperature', 'Humidity', 'Wind Speed', 'Precipitation (%)', 'Cloud Cover', 'Atmospheric Pressure', 'UV Index', 'Visibility (km)', 'Wind_x_Temp', 'Vis_x_Humid', 'visibility_band_High', 'visibility_band_Low', 'visibility_band_Medium', 'uv_group_High', 'uv_group_Low', 'uv_group_Medium', 'Season_Autumn', 'Season_Spring', 'Season_Summer', 'Season_Winter', 'Location_coastal', 'Location_inland', 'Location_mountain']

st.set_page_config(
    page_title="Weather Type Classifier",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(
    """
    <style>
    .stApp {
        background-color: #e6f3f5;
        background: url('https://t3.ftcdn.net/jpg/05/12/49/76/360_F_512497688_LvVSsqt4bTuWtdkdNzH7MPnfADWPCt56.jpg') no-repeat center center fixed;
        background-size: cover;
    }   
    .stExpander {
        background-color: white;
        border: 2px solid #d1e7f0;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0px 4px 10px rgba(0,0,0,0.05);
    }
    .stForm {
        background-color: white;
        border: 1px solid #e1e1e1;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0px 4px 10px rgba(0,0,0,0.05);
        margin-bottom: 2rem;
    }
    .stSuccess {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        padding: 10px;
        border-radius: 5px;
        color: #155724;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🌤️ Weather Type Classifier")
st.markdown(
    "Predict the weather type based on meteorological inputs. "
    "Use the legend below to understand typical feature ranges."
)

with st.expander("See typical feature ranges for Weather Types from the Kaggle Dataset"):
    cols = st.columns(2)

    with cols[0]:
        st.markdown("""
        ### ☁️ Cloudy  
        - Temperature (°C): -20 to 84  
        - Humidity (%): 20 to 109  
        - Wind Speed (km/h): 0 to 36  
        - Precipitation (%): 10 to 109  
        - Atmospheric Pressure (hPa): 800 to 1200
        - UV Index: 0 to 14  
        - Visibility (km): 0 to 20  
        - Cloud Cover: partly cloudy, cloudy, overcast  
        - Season: All 
        - Location: coastal, inland, mountain
        """)

        st.markdown("""
        ### 🌧️ Rainy  
        - Temperature (°C): -20 to 84  
        - Humidity (%): 20 to 109  
        - Wind Speed (km/h): 0 to 48  
        - Precipitation (%): 10 to 109  
        - Atmospheric Pressure (hPa): 800 to 1200  
        - UV Index: 0 to 14 
        - Visibility (km): 0 to 20  
        - Cloud Cover: partly cloudy, cloudy, overcast  
        - Season: All 
        - Location: coastal, inland, mountain
        """)

    with cols[1]:
        st.markdown("""
        ### ☀️ Sunny  
        - Temperature (°C): -20 to 109  
        - Humidity (%): 20 to 109  
        - Wind Speed (km/h): 0 to 25  
        - Precipitation (%): 0 to 109   
        - Atmospheric Pressure (hPa): 800 to 1200  
        - UV Index: 0 to 14  
        - Visibility (km): 0 to 20  
        - Cloud Cover: Clear
        - Season: All 
        - Location: coastal, inland, mountain
        """)

        st.markdown("""
        ### ❄️ Snowy  
        - Temperature (°C): -25 to -1  
        - Humidity (%): 41 to 109  
        - Wind Speed (km/h): 3 to 50  
        - Precipitation (%): 10 to 109  
        - Atmospheric Pressure (hPa): 800 to 1200  
        - UV Index: 0 to 14 
        - Visibility (km): 0 to 20  
        - Cloud Cover: partly cloudy, cloudy, overcast 
        - Season: Winter 
        - Location: coastal, inland, mountain
        """)

col1, col2 = st.columns(2)

with col1:
    with st.form(key='weather_form'):
        st.subheader("Enter Weather Parameters")
        col3, col4 = st.columns(2)

        with col3:
            temperature = st.number_input("Temperature (°C)", -25.0, 110.0, 20.0, help="Typical range: -25 to 110°C")
            humidity = st.number_input("Humidity (%)", 0.0, 109.0, 20.0, help="Typical range: 0% to 109%")
            wind_speed = st.number_input("Wind Speed (km/h)", 0.0, 50.0, 10.0, help="Typical range: 0 to 50 km/h")
            precipitation = st.number_input("Precipitation (%)", 0.0, 109.0, 10.0, help="Typical range: 0% to 109%")
            atmosphere = st.number_input("Atmospheric Pressure (hPa)", 800.0, 1200.0, 1000.0, help="Typical range: 800 to 1200 hPa")

        with col4:
            uv = st.number_input("UV Index", 0.0, 14.0, 5.0, help="Typical range: 0 to 14")
            visibility = st.number_input("Visibility (km)", 0.0, 20.0, 10.0, help="Typical range: 0 to 20 km")
            cloud_cover_label = st.selectbox("Cloud Cover", ['clear', 'partly cloudy', 'cloudy', 'overcast'])
            season = st.selectbox("Season", ['spring', 'summer', 'autumn', 'winter'])
            location = st.selectbox("Location", ['coastal', 'inland', 'mountain'])

        submit_button = st.form_submit_button("Predict Weather")

    def generate_features():
        cloud_cover_map = {
            'clear': 0,
            'partly cloudy': 1,
            'cloudy': 2,
            'overcast': 3
        }

        # Calculate uv_group and visibility_band first
        if uv <= 2:
            uv_group = "Low"
        elif uv <= 7:
            uv_group = "Medium"
        else:
            uv_group = "High"

        if visibility < 5:
            visibility_band = "Low"
        elif visibility < 10:
            visibility_band = "Medium"
        else:
            visibility_band = "High"

        # Calculate interaction features
        wind_x_temp = wind_speed * temperature
        vis_x_humid = visibility * humidity

        # Core input dictionary including interaction features
        input_dict = {
            'Temperature': temperature,
            'Humidity': humidity,
            'Wind Speed': wind_speed,
            'Precipitation (%)': precipitation,
            'Atmospheric Pressure': atmosphere,
            'UV Index': uv,
            'Visibility (km)': visibility,
            'Cloud Cover': cloud_cover_map[cloud_cover_label],
            'Wind_x_Temp': wind_x_temp,
            'Vis_x_Humid': vis_x_humid,
            # One-hot encoded categorical variables
            f'visibility_band_{visibility_band}': 1,
            f'uv_group_{uv_group}': 1,
            'Season_Autumn': 1 if season == 'autumn' else 0,
            'Season_Spring': 1 if season == 'spring' else 0,
            'Season_Summer': 1 if season == 'summer' else 0,
            'Season_Winter': 1 if season == 'winter' else 0,
            'Location_coastal': 1 if location == 'coastal' else 0,
            'Location_inland': 1 if location == 'inland' else 0,
            'Location_mountain': 1 if location == 'mountain' else 0,
        }

        # Add missing columns with 0 to match model features
        for col in model_features:
            if col not in input_dict:
                input_dict[col] = 0

        df_input = pd.DataFrame([input_dict])
        df_input = df_input[model_features]
        return df_input

with col2:
    if submit_button:
        X_input = generate_features()
        prediction = model.predict(X_input)[0]

        weather_images = {
            "Sunny": "sunny.jpg",
            "Cloudy": "cloudy.jpg",
            "Snowy": "snowy.jpg",
            "Rainy": "rainy.jpg"
        }

        
        predicted_label = prediction
        image_path = weather_images.get(predicted_label, "")

        st.success(f"### 🌤️ The predicted weather type is: **{prediction}**")

        st.image(image_path, caption=f"Predicted Weather: {predicted_label}", use_container_width=True)
