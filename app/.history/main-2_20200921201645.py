import keras_preprocessing
from keras_preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
from tkinter import filedialog
from tkinter import *
from PIL import ImageTk, Image
from keras_preprocessing import image
import numpy as np

import cv2
import skimage.io as io
import skimage.transform as trans
import shutil
from skimage import data
from skimage.io import imread, imshow
from keras.models import load_model


from skimage import data
from skimage.transform import (hough_line, hough_line_peaks, hough_circle, hough_circle_peaks)
from skimage.draw import circle_perimeter
from skimage.color import gray2rgb
from matplotlib import cm
from time import time


# sign_image = Label(ventana)
# segm = Label(ventana)

class Preprocess:
    def __init__(self,ventana):
        self.ventana=ventana
        self.sign_image = Label(self.ventana)
        self.segmented = Label(self.ventana)
        self.cargar_muestra = Button(self.ventana, text="Cargar muestra",command=self.upload_image)
        self.cargar_muestra.pack(side=BOTTOM, expand=True)
        self.sign_image.pack(side=BOTTOM,expand=True)
        self.segmented.pack(side=TOP,expand=True)
        self.segmented_sample_path = os.path.join(os.getcwd(),"output.png")
                
    def upload_image(self):
        try:
            original_sample_path=filedialog.askopenfilename()
            uploaded=Image.open(original_sample_path)
            uploaded.thumbnail(((self.ventana.winfo_width()/2.25),(self.ventana.winfo_height()/2.25)))
            im=ImageTk.PhotoImage(uploaded)
            self.sign_image.configure(image=im)
            self.sign_image.image=im

            # label.configure(text='')
            self.show_menu(original_sample_path)
        except:
            pass

    def show_menu(self,original_sample_path):
        # segmentación
        segment_button=Button(self.ventana,text="Iris Segmentation", command=lambda: self.segmentar(original_sample_path),padx=10,pady=5)
        segment_button.configure(background='#364156',
                            foreground='white',
                            font=('arial',10,'bold'))
        segment_button.place(relx=0.79,rely=0.40)

        #coords button
        coords_button=Button(self.ventana,text="Iris Coordinates", command=lambda: self.get_coords(self.segmented_sample_path),padx=10,pady=5)
        segment_button.configure(background='#364156',
                            foreground='white',
                            font=('arial',10,'bold'))
        segment_button.place(relx=0.79,rely=0.300)

    def get_coords(self, Segmentation):
        pass
    
    def ajustar_input(self,original_sample_path):
        img = cv2.imread(original_sample_path,0)
        render = img / 255
        render = trans.resize(render,(320,320))
        render = np.reshape(render,(1,)+render.shape)
        return render

    def morph_operator(self,result):
        img = cv2.resize(result[0][:,:,0], (320,280), interpolation=cv2.INTER_AREA)
        img = cv2.normalize(src=img, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        blur= cv2.GaussianBlur(img,(17,17),0)
        (thresh, binarized) = cv2.threshold(blur, 70, 255, cv2.THRESH_BINARY) # se binariza nuevament
        return binarized

    def segmentar(self,original_sample_path):
        global im
        top = Toplevel()
        top.title("U-net Output")
        model = load_model('Iris_unet_d5.h5')
        # predecimos(segementamos) la muestra
        result = model.predict(self.ajustar_input(original_sample_path))
        # reescalamos a tamaño original
        segmented = self.morph_operator(result)
        io.imsave(self.segmented_sample_path,segmented)
        im=ImageTk.PhotoImage(Image.open(self.segmented_sample_path))
        Label(top, image=im).pack()
 



if __name__ == '__main__':
    ventana = Tk()
    ventana.geometry("500x500")
    ventana.title("Iris Classifier")
    encabezado = Label(ventana, text="Iris Classifier")
    encabezado.configure(bg='#CDCDCD',
                        fg='#364156',
                        font=("Arial", 20))
    encabezado.pack()
    app = Preprocess(ventana)
    ventana.mainloop()