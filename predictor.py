import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

model = load_model('./Models')
labels = ['Arsenal', 'Aston-villa', 'Brentford', 'Brighton', 'Burnley', 'Chelsea', 'Crystal-palace', 'Everton', 'Leeds', 'Leicester-city', 'Liverpool', 'Manchester-city', 'Manchester-united', 'Newcastle', 'Norwich', 'Southampton', 'Tottenham', 'Watford', 'West-ham', 'Wolves']

def predict_with_model(img_path):
    img = image.load_img(img_path, target_size=(140, 140))
    img = image.img_to_array(img)/255.0
    img = img.reshape(1, 140, 140, 3)

    prediction = model.predict(img)
    prediction = np.argmax(prediction)
    return labels[prediction]


if __name__ == '__main__':
    img_path_1 = 'C:\\Users\\Jobin\\Ds\\Datasets\\EPL_logo\\data\\test\\aston-villa\\0be6b4be-dfba-471d-b7fa-474181fdb4de.png'
    img_path_2 = 'C:\\Users\\Jobin\\Ds\\Datasets\\EPL_logo\\data\\test\\arsenal\\6bf7314b-b05c-4520-be3c-54c244fb713e.png'

    prediction_1 = predict_with_model(img_path_1)
    prediction_2 = predict_with_model(img_path_2)
    print(f'Prediction = {prediction_1}')
    print(f'Prediction = {prediction_2}')