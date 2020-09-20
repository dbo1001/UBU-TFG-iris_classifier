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

ventana = Tk()
ventana.geometry("500x500")
ventana.title("Iris Classifier")
encabezado = Label(ventana, text="Iris Classifier")
encabezado.configure(bg='#CDCDCD',
                     fg='#364156',
                     font=("Arial", 20))
encabezado.pack()
sign_image = Label(ventana)
segm = Label(ventana)
def upload_image():
    try:
        file_path=filedialog.askopenfilename()
        uploaded=Image.open(file_path)
        uploaded.thumbnail(((ventana.winfo_width()/2.25),(ventana.winfo_height()/2.25)))
        im=ImageTk.PhotoImage(uploaded)
        sign_image.configure(image=im)
        sign_image.image=im

        # label.configure(text='')
        show_classify_button(file_path)
    except:
        pass

def show_classify_button(file_path):
    classify_b=Button(ventana,text="Classify Iris", command=lambda: clasificar(file_path),padx=10,pady=5)
    classify_b.configure(background='#364156',
                         foreground='white',
                         font=('arial',10,'bold'))
    classify_b.place(relx=0.79,rely=0.40)

def clasificar(file_path):
    model = load_model('Iris_unet_d5.h5')
    img = cv2.imread(file_path,0)
    render = img / 255
    render = trans.resize(render,(320,320))
    render = np.reshape(render,(1,)+render.shape)
    result = model.predict(render)
    result = cv2.resize(result[0][:,:,0], (320,280), interpolation=cv2.INTER_AREA)
    io.imsave("output.png",result)
    uploaded=Image.open("output.png")
    uploaded.thumbnail(((ventana.winfo_width()/2.25),(ventana.winfo_height()/2.25)))
    im=ImageTk.PhotoImage(uploaded)
    sign_image.configure(image=im)
    sign_image.image=im


cargar_muestra = Button(ventana, text="Cargar muestra",command=upload_image)
cargar_muestra.pack(side=BOTTOM, expand=True)
sign_image.pack(side=BOTTOM,expand=True)
segm.pack(side=TOP,expand=True)

ventana.mainloop()