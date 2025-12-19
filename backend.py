import pandas as pd
import joblib

bundle = joblib.load("final_model_bundle.pkl")
model = bundle["model"]
features = bundle["features"]

def predict_price(input_data):
    df = pd.DataFrame([input_data])
    df = pd.get_dummies(df)
    df = df.reindex(columns=features, fill_value=0)
    return model.predict(df)[0]

if __name__ == "__main__":
    test_house = {
        "area": 7420,
        "bedrooms": 4,
        "bathrooms": 2,
        "stories": 3,
        "mainroad": "yes",
        "guestroom": "no",
        "basement": "no",
        "hotwaterheating": "no",
        "airconditioning": "yes",
        "parking": 2,
        "prefarea": "yes",
        "furnishingstatus": "furnished"
    }

    print("Predicted Price:", predict_price(test_house))
