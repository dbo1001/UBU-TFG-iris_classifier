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
        show_menu(file_path)
    except:
        pass

def show_menu(file_path):
    # segemntación
    segment_button=Button(ventana,text="Classify Iris", command=lambda: clasificar(file_path),padx=10,pady=5)
    segment_button.configure(background='#364156',
                         foreground='white',
                         font=('arial',10,'bold'))
    segment_button.place(relx=0.79,rely=0.40)


def ajustar_input(file_path):
    img = cv2.imread(file_path,0)
    render = img / 255
    render = trans.resize(render,(320,320))
    render = np.reshape(render,(1,)+render.shape)
    return render

def morph_operator(result):
    img = cv2.resize(img[0][:,:,0], (320,280), interpolation=cv2.INTER_AREA)
    img = cv2.normalize(src=img, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    blur= cv2.GaussianBlur(img,(17,17),0)
    (thresh, binarized) = cv2.threshold(blur, 70, 255, cv2.THRESH_BINARY) # se binariza nuevament
    return binarized

def clasificar(file_path):
    global im
    top = Toplevel()
    top.title("U-net Output")
    model = load_model('Iris_unet_d5.h5')
    # predecimos(segementamos) la muestra
    result = model.predict(ajustar_input(file_path))
    # reescalamos a tamaño original
    segmented = morph_operator(result)

    io.imsave("output.png",segmented)
    im=ImageTk.PhotoImage(Image.open("output.png"))
    Label(top, image=im).pack()
 

cargar_muestra = Button(ventana, text="Cargar muestra",command=upload_image)
cargar_muestra.pack(side=BOTTOM, expand=True)
sign_image.pack(side=BOTTOM,expand=True)
segm.pack(side=TOP,expand=True)

ventana.mainloop()