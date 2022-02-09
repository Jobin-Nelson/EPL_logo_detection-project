from flask import Flask, request, render_template
import predictor

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/sub', methods=['POST'])
def submit():
    if request.method == 'POST':
        img = request.files['Image']
        img_path = 'static/' + img.filename
        img.save(img_path)
        label = predictor.predict_with_model(img_path)
        return render_template('sub.html', label=label, img_path=img_path)


if __name__ == '__main__':
    app.run(debug=True)
