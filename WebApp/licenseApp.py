from flask import Flask, render_template, request
import os
from Prediction import deep_ocr, easy_ocr
from flask_bootstrap import Bootstrap
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, BooleanField
from wtforms.validators import InputRequired, Email, Length

# webserver gateway interface
app = Flask(__name__)
Bootstrap(app)



BASIC_PATH = os.getcwd()
UPLOAD_PATH = os.path.join(BASIC_PATH, 'static/upload/')


@app.route('/', methods=['POST', 'GET'])
def index():
    if request.method == 'POST':
        # get the uploaded file
        uploaded_file = request.files['image']
        # get the name of the uploaded file
        file_name = uploaded_file.filename
        # create the path for saving the uploaded file
        save_path = os.path.join(UPLOAD_PATH, file_name)
        # saving the uploaded file
        uploaded_file.save(save_path)
        print(file_name, 'was uploaded successfully!')
        #plate_number = deep_ocr(save_path, file_name)
        plate_number = easy_ocr(save_path, file_name)
        #print(plate_number)
        return render_template('index.html', upload=True, uploaded_image=file_name, text=plate_number)
    return render_template('index.html', upload=False)

if __name__ == "__main__":
    app.run(debug=True)
