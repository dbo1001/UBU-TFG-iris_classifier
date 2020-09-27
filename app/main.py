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
    def __init__(self,ventana, clf):
        self.ventana=ventana
        self.clf = clf
        self.ventana.resizable(0,0)
        self.top = Frame(self.ventana)
        self.top.config(
            bd = 15
        )
        self.top.pack(side="top")
        self.center = Frame(self.ventana)
        self.center.pack(side="top")

        self.resultado = Frame(self.ventana) # para mostrar si acceso autorizado
        self.resultado.pack(side="top")
        # Label(self.resultado, text="USUARIO AUTORIZADO").pack()
        self.resultado.config(
            bd = 15
        )

        self.bottom = Frame(self.ventana) # botones de segmentación, coordeandas y normalización
        self.bottom.pack(side="top")


        self.sign_image = Label(self.top)
        self.cargar_muestra = Button(self.ventana, text="Cargar muestra",command=self.upload_image,width=15, height=1)
        self.cargar_muestra.config(
            bg='white',
            font=('arial',10)
        )
        self.cargar_muestra.pack(in_=self.center, side="left")
        # self.sign_image.pack(side=BOTTOM,expand=True)
        self.boundaries,self.centers = [],[]
        self.cx, self.cy, self.radius=(),(),()

        # nombre de las muestras
        self.segmented_sample_path = os.path.join(os.getcwd(),"output.png")
        self.canny_sample_path = os.path.join(os.getcwd(),"output-canny.png")
        self.coords_sample_path = os.path.join(os.getcwd(),"coords.png")
        self.norm_sample_path = os.path.join(os.getcwd(),"normalized.png")
        self.enh_sample_path = os.path.join(os.getcwd(),"enhanced.png")
                
    def upload_image(self):
        try:
            original_sample_path=filedialog.askopenfilename()
            uploaded=Image.open(original_sample_path)
            im=ImageTk.PhotoImage(uploaded)

            self.sign_image.configure(image=im)
            self.sign_image.image=im
            self.sign_image.pack(side = "top", fill = "both", expand = "yes")

            # label.configure(text='')
            self.show_clf_button(original_sample_path)
        except:
            pass
    
    def get_sujeto(self):
        features = self.clf.extract_features(self.enh_sample_path)
        sujeto = self.clf.predecir(features)
        return sujeto[0]

    def show_clf_button(self, original_sample_path):
        clf_button=Button(self.ventana,text="Log in", command=lambda: self.segmentar(original_sample_path),width=15, height=1)
        clf_button.configure(background='#364156',
                            foreground='white',
                            font=('arial',10))
        clf_button.pack(in_=self.center, side="right")

    def show_segment_button(self):
        # segmentación
        segment_button=Button(self.ventana,text="Segmentación", command=lambda: self.print_button("Segmentación",self.segmented_sample_path),width=15, height=1)
        segment_button.configure(background='#364156',
                            foreground='white',
                            font=('arial',10))
        segment_button.pack(in_=self.bottom,side="left")

    def show_coords_button(self):
        #coords button
        coords_button=Button(self.ventana,text="Iris Coordinates", command=lambda: self.print_button("Coordinates", self.coords_sample_path),width=15, height=1)
        coords_button.configure(background='#364156',
                            foreground='white',
                            font=('arial',10))
        coords_button.pack(in_=self.bottom,side="left")

    def show_norm_button(self):
        #coords button
        norm_button=Button(self.ventana,text="Normalization", command=lambda: self.print_button("Normalización", self.norm_sample_path),width=15, height=1)
        norm_button.configure(background='#364156',
                            foreground='white',
                            font=('arial',10))
        norm_button.pack(in_=self.bottom,side="left")
    
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
        img = normalized
        h,w = img.shape
        roi = img[5:h, 0:int(512/2)]
        roi_enhanced = exposure.equalize_hist(roi)
        return roi_enhanced

    def iris_normalization(self):
        global img_norm, img_enh
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

        self.show_segment_button()
        self.show_coords_button()
        self.show_norm_button()
        sujeto = self.get_sujeto()
        Label(self.resultado, text="Sujeto identificado: "+sujeto).pack()

    ###############LOCATE COORDS########################
    def draw_circles(self,img, cx, cy, radii):
        '''
        A partir de los centros y el radio detectados dibuja el iris sobre la imagen que se le
        pasa como parámetro.
        '''
        image = img.copy()
    #     cv2.circle(image,(cx,cy), radii, (255, 0, 0), 2)
        pupil = cv2.circle(image,(cx[0],cy[0]), radii[0]+3, (255, 0, 0), 2)
        iris = cv2.circle(image,(cx[1],cy[1]), radii[1], (255, 0, 0), 2)
        return image

    def get_circles(self, borde, original_sample_path):
        sample_ = cv2.imread(original_sample_path, 0)
        gray_img = cv2.imread(self.canny_sample_path, 0)
        
        hough_radii = np.arange(20, 80) # pupila por defecto
        if borde == "iris":
            hough_radii = np.arange(90, 160) # rango del iris
        
        hough_res = hough_circle(gray_img, hough_radii)
        accums, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii, total_num_peaks=1) 
        
        return [cx[0], cy[0], radii[0]]

    def get_coords(self, original_sample_path):
        # global img_coord
        # top = Toplevel()
        # top.title("Coordinates")
        
        pupil_coord = self.get_circles("pupil",original_sample_path)
        iris_coord = self.get_circles("iris",original_sample_path)
        self.cx, self.cy, self.radius = list(zip(pupil_coord, iris_coord))
        texto_pupila = "center of pupil: " + "(" + str(self.cx[0]) +", "+str(self.cy[0])+")\n" + "radius of pupil: " + str(self.radius[0])
        texto_iris = "center of iris: " + "(" + str(self.cx[1]) +", "+str(self.cy[1])+")\n" + "radius of iris: " + str(self.radius[1])

        draw = self.draw_circles(cv2.imread(original_sample_path, 0),self.cx,self.cy,self.radius)
        self.boundaries.append(draw)
        self.centers.append(pupil_coord)
        io.imsave(self.coords_sample_path,draw)
        # img_coord=ImageTk.PhotoImage(Image.open("coords.png"))
        # Label(top, image=img_coord).pack()
        # Label(top,text=texto_pupila).pack()
        # Label(top,text=texto_iris).pack()
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
        # global im
        # top = Toplevel()
        # top.title("U-net Output")
        model = load_model('Iris_unet_d5.h5')
        # predecimos(segementamos) la muestra
        result = model.predict(self.ajustar_input(original_sample_path))
        # reescalamos a tamaño original
        segmented = self.morph_operator(result)
        io.imsave(self.segmented_sample_path,segmented)
        cannied = cv2.Canny(segmented,10,255)# cambiar nombre variable
        io.imsave(self.canny_sample_path,cannied)
        # im=ImageTk.PhotoImage(Image.open(self.segmented_sample_path))
        # Label(top, image=im).pack()

        self.get_coords(original_sample_path)
 



if __name__ == '__main__':
    ventana = Tk()
    ventana.geometry("500x500")
    # ventana.title("Iris Classifier")
    encabezado = Label(ventana, text="Iris Classifier")
    encabezado.configure(bg='#364156',
                        fg='white',
                        font=("Consolas", 20),
                        padx=500)
    encabezado.pack()
    app = Preprocess(ventana, Classifier())
    ventana.mainloop()