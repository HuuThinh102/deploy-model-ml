import joblib


def predict(data):
    scaleInputFromUser = joblib.load("scale_func.sav")
    data = scaleInputFromUser.transform(data)
    model = joblib.load("rf_model.sav")
    return model.predict(data)