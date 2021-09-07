#------------------------------------------------------------------------------------------------------------------------
# TALLE 3: Paula Andrea Castro y Michael Hernando Contreras
# MainT3.py se llaman los métodos de Metodos.py y adicional se realiza la interpolación resultante del Punto 4 del taller
#------------------------------------------------------------------------------------------------------------------------

# Importación de librerias y métodos de la clase Descomp
from Metodos import Descomp
import numpy as np
import cv2
import sys
import os

""" 
    MainT3.py <path_to_image> <image_name>
"""

# Pulsar el botón verde en la barra superior para ejecutar el script.
if __name__ == '__main__':
    path = sys.argv[1]
    image_name = sys.argv[2]
    path_file = os.path.join(path, image_name)
    image = cv2.imread(path_file)
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Comprobar que la imagen es válida
    assert image is not None, "There is no image at {}".format(path_file)
# Llamado de la clase
    Desc = Descomp(image_gray)
# Visualizacion del punto 1 del taller 3 (descomentar para visualizar)
    #Desc.Diezmado() # (Quitar comentario para visualizar)
 # Visualización del punto 2 del taller 3
    #Desc.Interpolacion() #(Quitar comentario para visualizar)
# Visualización del punto 3 y 4
    Iim,N = Desc.Descomposicion()
    I = N*2
    #Intepolación final
    rows, cols = Iim.shape
    num_of_zeros = I
    image_zeros = np.zeros((num_of_zeros * rows, num_of_zeros * cols), dtype=Iim.dtype)
    image_zeros[::num_of_zeros, ::num_of_zeros] = Iim
    W = 2 * num_of_zeros + 1
    # Filtrado
    image_interpolated = cv2.GaussianBlur(image_zeros, (W, W), 10)
    image_interpolated *= num_of_zeros ** 2
    # Visualización imagen interpolada
    cv2.imshow("Imagen Original", image)
    cv2.imshow("ILL Interpolada", image_interpolated)
    cv2.waitKey(0)

