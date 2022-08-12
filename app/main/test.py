"""An extension of the code for predicting the images fed to a neural network. The original code
can be found in the original project's directory. This one has been reworked to fit the purpose
of using in a Flask webapp
"""

# import dependency relating to TensorFlow model prediction
import os
 # for feeding image to neural network
from keras.preprocessing.image import load_img, img_to_array
from keras import models
import pickle
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # suppress tensorflow warnings

# for the birds
def predict_bird(file_path):
    # some parameters
    image_height = 128
    image_width = 128
    # load the model and dict for predictions
    model = models.load_model(r'app/main/model_artifacts/birds_final_model.h5')

    dict_path = r'app/main/model_artifacts/birds_dict.pickle'
    with open(dict_path, 'rb') as handle:
        class_dict = pickle.load(handle)

    # preprocess the image
    image = load_img(file_path, target_size=(image_height, image_width))

    # convert image to array & clip to 0-1 range
    image = img_to_array(image)
    image = np.array(image)
    image = image[:]/255
    image = np.expand_dims(image, axis=0)

    # feed image to model & predict
    pred = model.predict(image)
    pred = np.argmax(pred, axis=1)
    
    for key, value in class_dict.items():
        if pred == key:
            predicted_bird_specie = class_dict[key]
    return predicted_bird_specie

# for the pets
def predict_pets(file_path):
    # some parameters
    image_height = 150
    image_width = 150

    # load the model
    model = models.load_model(r'app/main/model_artifacts/dogscats_final.h5')

    # preprocess the image
    image = load_img(file_path, target_size=(image_height, image_width))

    # convert image to array & clip to 0-1 range
    image = img_to_array(image)
    image = np.array(image)
    image = image[:]/255
    image = np.expand_dims(image, axis=0)

    # feed image to model & predict
    predict_proba = model.predict([image])

     # translate for use
    if predict_proba < 0.5:
        return "a cat."
    else:
        return "a dog."


# for the tumors
def predict_tumors(file_path):
    # some parameters
    image_height = 128
    image_width = 128

    # load the model and dict for predictions
    model = models.load_model(r'app/main/model_artifacts/brain_tumor_baseline.h5')

    dict_path = r'app/main/model_artifacts/tumors_dict.pickle'
    with open(dict_path, 'rb') as handle:
        class_dict = pickle.load(handle)

    # preprocess the image
    image = load_img(file_path, target_size=(image_height, image_width))

    # convert image to array & clip to 0-1 range
    image = img_to_array(image)
    image = np.array(image)
    image = image[:]/255
    image = np.expand_dims(image, axis=0)
    
    # feed image to model & predict
    pred = model.predict(image)
    pred = np.argmax(pred, axis=1)

    for key, value in class_dict.items():
        if pred == key:
            predicted_tumor = class_dict[key]
    return predicted_tumor
