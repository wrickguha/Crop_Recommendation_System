from flask import Flask, request, render_template
import numpy as np
import pickle

app = Flask(__name__)

model = pickle.load(open('model.pkl', 'rb'))
###sc = pickle.load(open('standscaler.pkl', 'rb'))
ms = pickle.load(open('minmaxscaler.pkl', 'rb'))

@app.route('/')
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    N = request.form['N']
    P = request.form['P']
    K = request.form['K']
    temp = request.form['temperature']
    rainfall = request.form['rainfall']
    ph = request.form['ph']
    humidity = request.form['humidity']

    feature_list = [N, P, K, temp, rainfall, ph, humidity]
    single_pred = np.array(feature_list).reshape(1, -1)  # Fix: Use np.array instead of np.ndarrayarray

    scaled_feature = ms.transform(single_pred)
    ###final_feature = sc.transform(scaled_feature)
    prediction = model.predict(scaled_feature)  

    crop_dict = {1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya", 7: "Orange",
                 8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana",
                 14: "Pomegranate", 15: "Lentil", 16: "Blackgram", 17: "Mungbean", 18: "Mothbeans",
                 19: "Pigeonpeas", 20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"}

    if prediction[0] in crop_dict:
        crop = crop_dict[prediction[0]]
        result = "{} is the best crop you can cultivate".format(crop)
    else:
        result = "Sorry, Could not find suitable crop to be cultivated with the provided data"

    return render_template('index.html', result=result)  # Fix: Add return statement

if __name__ == "__main__":
    app.run(debug=True)