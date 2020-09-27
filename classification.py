from keras.applications import InceptionV3
from keras.applications.inception_v3 import preprocess_input as inception_v3_preprocessor
from keras.models import clone_model
from keras.models import Model
from keras.preprocessing import image as Image
import numpy as np
from joblib import load
import cv2
import pickle
# from IPython.display import clear_output

class Classifier:
    def __init__(self):
        self.inception_v3_dict = {}
        self.model = InceptionV3(weights='imagenet')
        self.model = Model(self.model.input, self.model.layers[-2].output)# output_shape de la penúltima capa(pooling) -> (,2048)
        self.inception_v3_dict["model"] = clone_model(self.model)
        self.inception_v3_dict["preprocesor"] = inception_v3_preprocessor
        self.inception_v3_dict["target_size"] = self.model.input_shape[1],self.model.input_shape[2] # default inceptionv3 input (299,299)
        # self.cls = load("logistic_clf_trained.joblib")
        with open("pickle_model.pkl", 'rb') as file:
            self.cls = pickle.load(file)

    def extract_features(self, image):
        model = self.inception_v3_dict["model"]
        preprocessor = self.inception_v3_dict["preprocesor"]
        target_size = self.inception_v3_dict["target_size"]
        # se carga la imagen y después se ajusta al input shape del modelo
        img = cv2.resize(cv2.imread(image),target_size)

        img_data = Image.img_to_array(img)
        img_data = np.expand_dims(img_data, axis=0)
        img_data = preprocessor(img_data)
        
        # extracción de features
        features = model.predict(img_data)
        feat = features[0]
        feat.shape = (1,feat.shape[0])
        return feat

    def predecir(self,image):
        return self.cls.predict(image)


