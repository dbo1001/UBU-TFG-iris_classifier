Sistema clasificador de iris
-------------------------------
Tutor: José Francisco Diez Pastor

Los sistemas de seguridad basado en la biometría del iris están muy demandados debido a que permiten identificar inequívocamente a un individuo y por tanto su precisión es muy alta.

El proceso de reconocimiento se realiza en 3 fases:
- __Adquirir muestras__: en esta fase se recogen las imágenes del iris, ya sea mediante cámaras diseñadas especialmente para dicho propósito o cómo será nuestro caso, acudiendo a una base de datos existente.
- __Preprocesamiento de muestras__: esta fase se subdivide a su vez en 3 fases:

    - __Segmentación de imágenes__: se aisla por completo el iris intentando dejar fuera elementos innecesarios cómo pestañas, párpados, esclera, reflejos...
    - __Normalización de imágenes__: se establece un tamaño común para las iris segmentados, mediante polarización, en la que se estira el iris hasta que queda con forma rectangular.

    - __Extraer features__: en esta etapa se extraen los patrones característicos de cada iris ya sea mediante algoritmos mateméticos convencionales o con algoritmos de _machine learning_
- __Clasificación__: se compara los iris con una base de datos.

![Iris recognition](https://www.bayometric.com/wp-content/uploads/2016/06/an-iris-recognition-system.jpg)

