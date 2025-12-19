import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="HomeValue AI", layout="centered")

@st.cache_resource
def load_model():
    bundle = joblib.load("final_model_bundle.pkl")
    return bundle["model"], bundle["features"]

model, features = load_model()

def predict_price(input_data):
    df = pd.DataFrame([input_data])
    df = pd.get_dummies(df)
    df = df.reindex(columns=features, fill_value=0)
    return model.predict(df)[0]

st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #1d2671, #c33764);
}
.big-title {
    font-size:48px;
    font-weight:800;
    text-align:center;
    color:#ffffff;
}
.subtitle {
    text-align:center;
    color:#f1f1f1;
    font-size:18px;
}
.card {
    background-color:#ffffff;
    padding:25px;
    border-radius:15px;
    box-shadow:0px 10px 30px rgba(0,0,0,0.2);
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="big-title">🏠 HomeValue AI</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Urban & Rural House Price Prediction System</div><br>', unsafe_allow_html=True)

with st.container():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    with st.form("house_form"):
        area = st.number_input("Area (sqft)", min_value=100)
        bedrooms = st.slider("Bedrooms", 1, 10, 3)
        bathrooms = st.slider("Bathrooms", 1, 10, 2)
        stories = st.slider("Stories", 1, 5, 2)
        parking = st.slider("Parking", 0, 5, 1)

        mainroad = st.selectbox("Main Road", ["yes", "no"])
        guestroom = st.selectbox("Guest Room", ["yes", "no"])
        basement = st.selectbox("Basement", ["yes", "no"])
        hotwaterheating = st.selectbox("Hot Water Heating", ["yes", "no"])
        airconditioning = st.selectbox("Air Conditioning", ["yes", "no"])
        prefarea = st.selectbox("Preferred Area", ["yes", "no"])
        furnishingstatus = st.selectbox(
            "Furnishing Status",
            ["furnished", "semi-furnished", "unfurnished"]
        )

        submit = st.form_submit_button("✨ Predict Home Value")
    st.markdown('</div>', unsafe_allow_html=True)

if submit:
    input_data = {
        "area": area,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "stories": stories,
        "mainroad": mainroad,
        "guestroom": guestroom,
        "basement": basement,
        "hotwaterheating": hotwaterheating,
        "airconditioning": airconditioning,
        "parking": parking,
        "prefarea": prefarea,
        "furnishingstatus": furnishingstatus
    }

    price = predict_price(input_data)

    st.markdown(f"""
    <div style="
        background:linear-gradient(135deg,#11998e,#38ef7d);
        padding:25px;
        border-radius:15px;
        text-align:center;
        color:white;
        font-size:26px;
        font-weight:700;
        margin-top:20px;
    ">
    Estimated House Price<br>₹ {int(price):,}
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br><center>🚀 Powered by Machine Learning | HomeValue AI</center>", unsafe_allow_html=True)
