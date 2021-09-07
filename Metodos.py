#----------------------------------------------------------
# TALLE 3: Paula Andrea Castro y Michael Hernando Contreras
# Metodos.py contiene la clase Descomp que se compone de 3 métodos (diezmado, Interpolacion y Descomposicion)
#----------------------------------------------------------

# Importación de librerias
import cv2
import numpy as np

#Definición de la clase Descomp
class Descomp:
    #Definición del constructor
    def __init__(self, image):
        self.image_gray = image
    #Definción del método Diezmado que por un factor de D y utilizando FTT, retorna la imagen filtrada y diezmada
    def Diezmado(self):
        self.D = int(input('Ingrese el valor de D(entero positivo mayor a 1): '))
        # Aplica fft
        image_gray_fft = np.fft.fft2(self.image_gray)
        image_gray_fft_shift = np.fft.fftshift(image_gray_fft)
        # Visualización de fft
        image_gray_fft_mag = np.absolute(image_gray_fft_shift)
        image_fft_view = np.log(image_gray_fft_mag + 1)
        image_fft_view = image_fft_view / np.max(image_fft_view)
        # pre-cálculos
        num_rows, num_cols = (self.image_gray.shape[0], self.image_gray.shape[1])
        enum_rows = np.linspace(0, num_rows - 1, num_rows)
        enum_cols = np.linspace(0, num_cols - 1, num_cols)
        col_iter, row_iter = np.meshgrid(enum_cols, enum_rows)
        half_size = num_rows / 2  # here we assume num_rows = num_columns
        # Mascara de filtro pasa bajas
        low_pass_mask = np.zeros_like(self.image_gray) 
        freq_cut_off = 1 / self.D  # it should less than 1
        radius_cut_off = int(freq_cut_off * half_size)
        idx_lp = np.sqrt((col_iter - half_size) ** 2 + (row_iter - half_size) ** 2) < radius_cut_off
        low_pass_mask[idx_lp] = 1
        # Filtrado via FFT
        mask = low_pass_mask  # can also use high or band pass mask
        fft_filtered = image_gray_fft_shift * mask
        image_filtered = np.fft.ifft2(np.fft.fftshift(fft_filtered))
        image_filtered = np.absolute(image_filtered)
        image_filtered /= np.max(image_filtered)
        # Diezmado
        image_decimated = image_filtered[::self.D, ::self.D]
        #Visualización resultado del método
        cv2.imshow("Grey Image", self.image_gray)
        cv2.imshow("Filter frequency response", 255 * mask)
        cv2.imshow("Filtered image", image_decimated)
        cv2.waitKey(0)

    # Definción del método Interpolacion que por un factor de I y utilizando FTT, retorna la imagen filtrada e interpolada
    def Interpolacion(self):
        self.I = int(input('Ingrese el valor de I (entero positivo mayor a 1): '))
        # Aplica fft
        image_gray_fft = np.fft.fft2(self.image_gray)
        image_gray_fft_shift = np.fft.fftshift(image_gray_fft)
        # Visualización de fft
        image_gray_fft_mag = np.absolute(image_gray_fft_shift)
        image_fft_view = np.log(image_gray_fft_mag + 1)
        image_fft_view = image_fft_view / np.max(image_fft_view)
        # pre-cálculos
        num_rows, num_cols = (self.image_gray.shape[0], self.image_gray.shape[1])
        enum_rows = np.linspace(0, num_rows - 1, num_rows)
        enum_cols = np.linspace(0, num_cols - 1, num_cols)
        col_iter, row_iter = np.meshgrid(enum_cols, enum_rows)
        half_size = num_rows / 2  # here we assume num_rows = num_columns
        # Mascara de filtro pasa bajas
        low_pass_mask = np.zeros_like(self.image_gray)
        freq_cut_off = 1 / self.I  # it should less than 1
        radius_cut_off = int(freq_cut_off * half_size)
        idx_lp = np.sqrt((col_iter - half_size) ** 2 + (row_iter - half_size) ** 2) < radius_cut_off
        low_pass_mask[idx_lp] = 1
        # Filtrado via FFT
        mask = low_pass_mask  # can also use high or band pass mask
        fft_filtered = image_gray_fft_shift * mask
        image_filtered = np.fft.ifft2(np.fft.fftshift(fft_filtered))
        image_filtered = np.absolute(image_filtered)
        image_filtered /= np.max(image_filtered)
        # Interpolación
        # Insertar ceros
        rows, cols = self.image_gray.shape
        num_of_zeros = self.I - 1
        image_zeros = np.zeros((num_of_zeros * rows, num_of_zeros * cols), dtype=self.image_gray.dtype)
        image_zeros[::num_of_zeros, ::num_of_zeros] = self.image_gray
        W = 2 * num_of_zeros + 1
        # Filtrado
        image_interpolated = cv2.GaussianBlur(image_zeros, (W, W), 10)
        image_interpolated *= num_of_zeros ** 2
        #Visualización del resultado del método
        cv2.imshow("Grey Image",self.image_gray)
        cv2.imshow("Filter frequency response", 255 * mask)
        cv2.imshow("Interpolada", image_interpolated)
        cv2.imwrite("Interpolada.png", image_interpolated)
        cv2.waitKey(0)

    # Definción del método Descomposicion que recibe parámetro N (orden de la descomposición) e implementa un banco de filtros
    def Descomposicion(self):
        self.N = int(input('Ingrese el valor del orden de 1 a 3 : '))
        # Descomposición de Orden 1
        if self.N ==1:
            # Definición de los filtros
            H = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
            V = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
            D = np.array([[2, -1, -2], [-1, 4, -1], [-2, -1, 2]])
            L = np.array([[1 / 16, 1 / 8, 1 / 16], [1 / 8, 1 / 4, 1 / 8], [1 / 16, 1 / 8, 1 / 16]])
            # Punto b (Orden 1) convolución
            image_convolved1 = cv2.filter2D(self.image_gray, -1, L)
            image_convolved2 = cv2.filter2D(self.image_gray, -1, H)
            image_convolved3 = cv2.filter2D(self.image_gray, -1, V)
            image_convolved4 = cv2.filter2D(self.image_gray, -1, D)
            # Punto c (aplica diezmado D=2)
            D = 2
            IL = image_convolved1[::D, ::D]
            IH = image_convolved2[::D, ::D]
            IV = image_convolved3[::D, ::D]
            ID = image_convolved4[::D, ::D]
            #Visualizacion descomposición Orden 1
            cv2.imshow("Grey Image", self.image_gray)
            cv2.imshow("IH", IL)
            cv2.imshow("IV", IH)
            cv2.imshow("ID", IV)
            cv2.imshow("IL", ID)
            cv2.waitKey(3000)
            return IL, self.N
        # Descomposición de Orden 2
        elif self.N == 2:
            # Filtros
            H = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
            V = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
            D = np.array([[2, -1, -2], [-1, 4, -1], [-2, -1, 2]])
            L = np.array([[1 / 16, 1 / 8, 1 / 16], [1 / 8, 1 / 4, 1 / 8], [1 / 16, 1 / 8, 1 / 16]])
            # Orden 1 convoluación
            image_convolved1 = cv2.filter2D(self.image_gray, -1, L)
            image_convolved2 = cv2.filter2D(self.image_gray, -1, H)
            image_convolved3 = cv2.filter2D(self.image_gray, -1, V)
            image_convolved4 = cv2.filter2D(self.image_gray, -1, D)
            # Diezmado
            D = 2
            IL = image_convolved1[::D, ::D]
            IH = image_convolved2[::D, ::D]
            IV = image_convolved3[::D, ::D]
            ID = image_convolved4[::D, ::D]
            # Orden 2 convoluación
            image_convolved5 = cv2.filter2D(IL, -1, L)
            image_convolved6 = cv2.filter2D(IL, -1, H)
            image_convolved7 = cv2.filter2D(IL, -1, V)
            image_convolved8 = cv2.filter2D(IL, -1, D)
            # Diezmado
            ILL = image_convolved5[::D, ::D]
            IHL = image_convolved6[::D, ::D]
            IVL = image_convolved7[::D, ::D]
            IDL = image_convolved8[::D, ::D]
            #Visualizacion descomposición Orden 2
            cv2.imshow("Grey Image", self.image_gray)
            cv2.imshow("IHL", ILL)
            cv2.imshow("IVL", IHL)
            cv2.imshow("IDL", IVL)
            cv2.imshow("ILL", IDL)
            cv2.waitKey(3000)
            return ILL, self.N
        #Descomposicion orden 3
        elif self.N == 3:
            #Filtros
            H = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
            V = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
            D = np.array([[2, -1, -2], [-1, 4, -1], [-2, -1, 2]])
            L = np.array([[1 / 16, 1 / 8, 1 / 16], [1 / 8, 1 / 4, 1 / 8], [1 / 16, 1 / 8, 1 / 16]])
            #Orden 1 convolución
            image_convolved1 = cv2.filter2D(self.image_gray, -1, L)
            image_convolved2 = cv2.filter2D(self.image_gray, -1, H)
            image_convolved3 = cv2.filter2D(self.image_gray, -1, V)
            image_convolved4 = cv2.filter2D(self.image_gray, -1, D)
            # Diezmado
            D = 2
            IL = image_convolved1[::D, ::D]
            IH = image_convolved2[::D, ::D]
            IV = image_convolved3[::D, ::D]
            ID = image_convolved4[::D, ::D]
            # Orden 2 convolución
            image_convolved5 = cv2.filter2D(IL, -1, L)
            image_convolved6 = cv2.filter2D(IL, -1, H)
            image_convolved7 = cv2.filter2D(IL, -1, V)
            image_convolved8 = cv2.filter2D(IL, -1, D)
            # Diezmado
            ILL = image_convolved5[::D, ::D]
            IHL = image_convolved6[::D, ::D]
            IVL = image_convolved7[::D, ::D]
            IDL = image_convolved8[::D, ::D]
            # Orden 3 convolución
            image_convolved9 = cv2.filter2D(ILL, -1, L)
            image_convolved10 = cv2.filter2D(ILL, -1, H)
            image_convolved11 = cv2.filter2D(ILL, -1, V)
            image_convolved12 = cv2.filter2D(ILL, -1, D)
            # Diezmado
            IHLL = image_convolved9[::D, ::D]
            IVLL = image_convolved10[::D, ::D]
            IDLL = image_convolved11[::D, ::D]
            ILLL = image_convolved12[::D, ::D]
            #Visualizacion descomposición Orden 3
            cv2.imshow("Grey Image", self.image_gray)
            cv2.imshow("IHLL", ILLL)
            cv2.imshow("IVLL", IHLL)
            cv2.imshow("IDLL", IVLL)
            cv2.imshow("ILLL", IDLL)
            cv2.waitKey(3000)
            return ILLL, self.N

        else:
            print("Valor del Orden N Fuera del limite")
