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
from skimage import data, exposure
from skimage.io import imread, imshow
from keras.models import load_model


from skimage import data
from skimage.transform import (hough_line, hough_line_peaks, hough_circle, hough_circle_peaks)
from skimage.draw import circle_perimeter
from skimage.color import gray2rgb
from matplotlib import cm
from time import time
from classification import Classifier
# sign_image = Label(ventana)
# segm = Label(ventana)

class Preprocess:
    def __init__(self,ventana, clasificador):
        self.clf = clasificador # clasificador preentrenado

        self.ventana=ventana
        self.ventana.resizable(0,0)

        # frame para la muestra
        self.top = Frame(self.ventana)
        self.top.config(bd = 15)
        self.top.pack(side="top")

        # frame par botoners cargar archivo y clasificar
        self.center = Frame(self.ventana)
        self.center.pack(side="top")

        # frame para imprimir el sujeto identificado
        self.resultado = Frame(self.ventana) 
        self.resultado.pack(side="top")
        self.resultado.config(bd = 15)

        # frame para botones de segmentación, coordeandas y normalización
        self.bottom = Frame(self.ventana) 
        self.bottom.pack(side="top")


        self.sign_image = Label(self.top)
        self.cargar_muestra = Button(self.ventana, text="Cargar muestra",command=self.upload_image,width=15, height=1)
        self.cargar_muestra.config(
            bg='white',
            font=('arial',10)
        )
        self.cargar_muestra.pack(in_=self.center, side="left")

        self.boundaries,self.centers = [],[]
        self.cx, self.cy, self.radius=(),(),()

        # nombre de las muestras
        self.segmented_sample_path = os.path.join(os.getcwd(),"output.png")
        self.canny_sample_path = os.path.join(os.getcwd(),"output-canny.png")
        self.coords_sample_path = os.path.join(os.getcwd(),"coords.png")
        self.norm_sample_path = os.path.join(os.getcwd(),"normalized.png")
        self.enh_sample_path = os.path.join(os.getcwd(),"enhanced.png")

        #botones
        self.clf_button = Button(self.ventana)
        self.segment_button = Button(self.ventana)
        self.coords_button = Button(self.ventana)
        self.norm_button =Button(self.ventana)
        self.resultado = Label(self.resultado)

    def upload_image(self):
        try:
            original_sample_path=filedialog.askopenfilename()
            uploaded=Image.open(original_sample_path)
            im=ImageTk.PhotoImage(uploaded)

            self.sign_image.configure(image=im)
            self.sign_image.image=im
            self.sign_image.pack(side = "top", fill = "both", expand = "yes")
            
            self.show_clf_button(original_sample_path)
        except:
            pass
    
    def show_clf_button(self, original_sample_path):
        """
        muestra botón clasificar
        """
        self.clf_button.configure(text="Clasificar", 
                            command=lambda: self.segmentar(original_sample_path),
                            width=15, height=1,
                            background='#364156',
                            foreground='white',
                            font=('arial',10))
        self.clf_button.pack(in_=self.center, side="right")

    def show_segment_button(self):
        """
        muestra botón segmentación
        """
        # segmentación
        self.segment_button.configure(text="Segmentar", command=lambda: self.print_button("Segmentación",self.segmented_sample_path),width=15, height=1,
            background='#364156',
                            foreground='white',
                            font=('arial',10))
        self.segment_button.pack(in_=self.bottom,side="left")

    def show_coords_button(self):
        """
        muestra botón coordenadas
        """
        #coords button
        self.coords_button.configure(text="Coordenadas", command=lambda: self.print_button("Bordes límbico y pupilar", self.coords_sample_path),width=15, height=1,
                            background='#364156',
                            foreground='white',
                            font=('arial',10))
        self.coords_button.pack(in_=self.bottom,side="left")

    def show_norm_button(self):
        """
        muestra botón normalización
        """

        self.norm_button.configure(text="Normalizar", command=lambda: self.print_button("Normalización", self.norm_sample_path),width=15, height=1,
                    background='#364156',
                            foreground='white',
                            font=('arial',10))
        self.norm_button.pack(in_=self.bottom,side="left")
    
    def print_button(self, name, path):
        global im
        top = Toplevel(self.ventana)
        top.title(name)
        im=ImageTk.PhotoImage(Image.open(path))
        panel = Label(top, image=im)
        panel.image = im
        panel.pack()
        # top.transient(self.ventana)
        top.focus_set()
        top.grab_set()
        # self.ventana.wait_window(top)

    ############### NORMALIZATION ###########################
    def crop_and_ecualization(self,normalized):
        """
        recorta la muestra normalizada y resalta la textura del iris
        """
        img = normalized
        h,w = img.shape
        roi = img[5:h, 0:int(512/2)]
        roi_enhanced = exposure.equalize_hist(roi)
        return roi_enhanced

    def iris_normalization(self):
        """
        normaliza el iris ("lo estira")
        """
        normalized = []
        cent=0
        for img in self.boundaries:
    #         img = normalized[name]
            #load pupil centers and radius of inner circles
            center_x = self.centers[cent][0]
            center_y = self.centers[cent][1]
            radius_pupil=int(self.centers[cent][2])
            
            iris_radius = 53 # width of space between inner and outer boundary
        
            #define equally spaced interval to iterate over
            nsamples = 360
            samples = np.linspace(0,2*np.pi, nsamples)[:-1]
            polar = np.zeros((iris_radius, nsamples))
            for r in range(iris_radius):
                for theta in samples:
                    #get x and y for values on inner boundary
                    x = (r+radius_pupil)*np.cos(theta)+center_x
                    y = (r+radius_pupil)*np.sin(theta)+center_y
                    x=int(x)
                    y=int(y)
                    try:
                    #convert coordinates
                        polar[r][int((theta*nsamples)/(2*np.pi))] = img[y][x]
                    except IndexError: #ignores values which lie out of bounds
                        pass
                    continue
            res = cv2.resize(polar,(512,64))
            normalized.append(res)
            cent+=1
        

        io.imsave(self.norm_sample_path,normalized[0])
        io.imsave(self.enh_sample_path,self.crop_and_ecualization(normalized[0]))
        self.boundaries = []
        self.centers = []

        self.print_sujeto()
        self.show_segment_button()
        self.show_coords_button()
        self.show_norm_button()
        


    def print_sujeto(self):
        """
        clasifica el iris 
        """
        print(self.enh_sample_path)
        # self.counter = self.counter + 1
        clasif = self.clf()
        features = clasif.extract_features(self.enh_sample_path)
        sujeto = clasif.predecir(features)
        print(sujeto)
        self.resultado.configure(text="Sujeto identificado: "+sujeto)
        # self.enh_sample_path = os.path.join(os.getcwd(), "enhanced_" + str(self.counter) + ".png")
        self.resultado.pack()

    ###############LOCATE COORDS########################
    def draw_circles(self,img, cx, cy, radii):
        '''
        A partir de los centros y el radio detectados dibuja el iris sobre la imagen que se le
        pasa como parámetro.
        '''
        image = img.copy()
        pupil = cv2.circle(image,(cx[0],cy[0]), radii[0]+3, (255, 0, 0), 2)
        iris = cv2.circle(image,(cx[1],cy[1]), radii[1], (255, 0, 0), 2)
        return image

    def get_circles(self, borde, original_sample_path):
        """
        obtiene las coordenadas de los bordes límbivo y pupilar
        """
        sample_ = cv2.imread(original_sample_path, 0)
        gray_img = cv2.imread(self.canny_sample_path, 0)
        
        hough_radii = np.arange(20, 80) # pupila por defecto
        if borde == "iris":
            hough_radii = np.arange(90, 160) # rango del iris
        
        hough_res = hough_circle(gray_img, hough_radii)
        accums, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii, total_num_peaks=1) 
        
        return [cx[0], cy[0], radii[0]]


    def get_coords(self, original_sample_path):
        """
        dibuja las coordenadas en la muestra
        """
        pupil_coord = self.get_circles("pupil",original_sample_path)
        iris_coord = self.get_circles("iris",original_sample_path)
        self.cx, self.cy, self.radius = list(zip(pupil_coord, iris_coord))
        texto_pupila = "center of pupil: " + "(" + str(self.cx[0]) +", "+str(self.cy[0])+")\n" + "radius of pupil: " + str(self.radius[0])
        texto_iris = "center of iris: " + "(" + str(self.cx[1]) +", "+str(self.cy[1])+")\n" + "radius of iris: " + str(self.radius[1])

        draw = self.draw_circles(cv2.imread(original_sample_path, 0),self.cx,self.cy,self.radius)
        self.boundaries.append(draw)
        self.centers.append(pupil_coord)
        io.imsave(self.coords_sample_path,draw)

        self.iris_normalization()

    ###### SEGMENTATION ######################

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
        model = load_model('Iris_unet_d5.h5')
        # predecimos(segementamos) la muestra
        result = model.predict(self.ajustar_input(original_sample_path))
        # reescalamos a tamaño original
        segmented = self.morph_operator(result)
        io.imsave(self.segmented_sample_path,segmented)
        cannied = cv2.Canny(segmented,10,255)# cambiar nombre variable
        io.imsave(self.canny_sample_path,cannied)

        self.get_coords(original_sample_path)
 



if __name__ == '__main__':
    ventana = Tk()
    ventana.geometry("500x500")
    ventana.title("Iris Classifier")
    ventana.iconbitmap("icon.ico")
    encabezado = Label(ventana, text="Iris Classifier")
    encabezado.configure(bg='#364156',
                        fg='white',
                        font=("Consolas", 20),
                        padx=500)
    encabezado.pack()
    app = Preprocess(ventana, Classifier)
    ventana.mainloop()