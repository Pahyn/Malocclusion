import tensorflow as tf
from flask import Flask, render_template, request
from io import BytesIO
from keras.preprocessing import image
from keras.preprocessing.image import array_to_img, img_to_array
from keras.models import load_model
import cv2
import os
from cv2 import cv2
from PIL import Image
import numpy as np
from base64 import b64encode

from flask_bootstrap import Bootstrap
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileRequired, FileAllowed
from wtforms import SubmitField

# code which helps initialize our server
app = Flask(__name__)
app.config['SECRET_KEY'] = '123'

bootstrap = Bootstrap(app)

saved_model = tf.keras.models.load_model("models/model3.hdf5")
saved_model.make_predict_function()

class UploadForm(FlaskForm):
    photo = FileField('Upload an image',validators=[FileAllowed(['jpg', 'png', 'jpeg'], u'Image only!'), FileRequired(u'File was empty!')])
    submit = SubmitField(u'Predict')

def preprocess(img):
    width, height = img.shape[0], img.shape[1]
    img = image.array_to_img(img, scale=False)

    desired_width, desired_height = 128, 128

    if width < desired_width:
   	 desired_width = width
    start_x = np.maximum(0, int((width-desired_width)/2))
    img = img.crop((start_x, np.maximum(0, height-desired_height), start_x+desired_width, height))
    img = img.resize((desired_width, desired_height))
    img = image.img_to_array(img)
#    img = apply_mask(img)
    return img / 255.

def apply_mask(image, size=20, n_squares=1):
    h, w  = 128,128
    new_image = image
    for _ in range(n_squares):
        y = np.random.randint(h*0.35)
        x = np.random.randint(w*0.50)
        y1 = np.clip(y - size // 2, 0, h)
        y2 = np.clip(y + size // 2, 0, h)
        x1 = np.clip(x - size // 2, 0, w)
        x2 = np.clip(x + size // 2, 0, w)
        new_image[y1:y2,x1:x2,:] = 0
    return new_image

@app.route('/', methods=['GET','POST'])
def predict():
    form = UploadForm()
    if form.validate_on_submit():
   	 print(form.photo.data)
   	 image_stream = form.photo.data.stream
   	 original_img = Image.open(image_stream).convert('RGB')
   	 ori_img = Image.open(image_stream)
   	 img = image.img_to_array(original_img)
   	 img = preprocess(img)
   	 img = np.expand_dims(img, axis=0)
   	 prediction = saved_model.predict(img)
   	 result = prediction[0]*100 
   	 result_max = max(result)                       
   	 #result = str(np.argmax(prediction))

   	 byteIO = BytesIO()
   	 ori_img.save(byteIO, format=ori_img.format)
   	 byteArr = byteIO.getvalue()
   	 encoded = b64encode(byteArr)

   	 return render_template('result.html', result=result, result_max=result_max, encoded_photo=encoded.decode('ascii'), variable = prediction)

    return render_template('index.html', form=form)

if __name__ == '__main__':
    app.run(debug=True)
