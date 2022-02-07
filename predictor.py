import tensorflow as tf
import numpy as np

labels = ['Arsenal', 'Aston-villa', 'Brentford', 'Brighton', 'Burnley', 'Chelsea', 'Crystal-palace', 'Everton', 'Leeds', 'Leicester-city', 'Liverpool', 'Manchester-city', 'Manchester-united', 'Newcastle', 'Norwich', 'Southampton', 'Tottenham', 'Watford', 'West-ham', 'Wolves']

def predict_with_model(model, img_path):
    image = tf.io.read_file(img_path)
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    image = tf.image.resize(image, [140, 140])
    image = tf.expand_dims(image, axis=0)

    predictions = model.predict(image)
    predictions = np.argmax(predictions)
    return labels[predictions]


if __name__ == '__main__':
    img_path_1 = 'C:\\Users\\Jobin\\Ds\\Datasets\\EPL_logo\\data\\test\\aston-villa\\0be6b4be-dfba-471d-b7fa-474181fdb4de.png'
    img_path_2 = 'C:\\Users\\Jobin\\Ds\\Datasets\\EPL_logo\\data\\test\\arsenal\\6bf7314b-b05c-4520-be3c-54c244fb713e.png'
    model = tf.keras.models.load_model('./Models')

    prediction = predict_with_model(model, img_path_2)
    print(f'Prediction = {prediction}')