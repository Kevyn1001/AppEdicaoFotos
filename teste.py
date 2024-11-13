import tkinter as tk
from tkinter import filedialog
import cv2
from PIL import Image, ImageTk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Variáveis globais para imagem e limiar
img_cv = None
threshold_value = 79

# Funções para criar as janelas de imagem original, histograma e imagem processada
def create_original_window():
    global original_image_canvas
    original_window = tk.Toplevel(root)
    original_window.title("Imagem Original")
    original_image_canvas = tk.Canvas(original_window, width=300, height=300, bg="gray")
    original_image_canvas.pack(padx=10, pady=10)

def create_histogram_window():
    global histogram_canvas, hist_frame
    histogram_window = tk.Toplevel(root)
    histogram_window.title("Histograma")
    hist_frame = tk.Frame(histogram_window, width=300, height=300, bg="gray")
    hist_frame.pack(padx=10, pady=10)
    histogram_canvas = FigureCanvasTkAgg(plt.figure(figsize=(4, 3)), master=hist_frame)
    histogram_canvas.get_tk_widget().pack()

def create_processed_window():
    global processed_image_canvas
    processed_window = tk.Toplevel(root)
    processed_window.title("Imagem Processada")
    processed_image_canvas = tk.Canvas(processed_window, width=300, height=300, bg="gray")
    processed_image_canvas.pack(padx=10, pady=10)

# Função para carregar a imagem
def load_image():
    global img_cv
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.bmp")])
    if file_path:
        img_cv = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        display_image(img_cv, original=True)
        update_histogram()

# Função para exibir a imagem no Tkinter
def display_image(img, original=False):
    img_pil = Image.fromarray(img)
    img_pil.thumbnail((300, 300)) 
    img_tk = ImageTk.PhotoImage(img_pil)

    if original:
        original_image_canvas.delete("all")
        original_image_canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
        original_image_canvas.image = img_tk
    else:
        processed_image_canvas.delete("all")
        processed_image_canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
        processed_image_canvas.image = img_tk

# Função para atualizar o histograma com a linha de limiar
def update_histogram():
    global histogram_canvas
    histogram_canvas.get_tk_widget().pack_forget()
    
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.hist(img_cv.ravel(), bins=256, range=(0, 256), color='gray')
    ax.axvline(threshold_value, color='red', linestyle='--')  # Linha de limiar
    ax.set_title("Histogram")
    ax.set_xlabel("Pixel Intensity")
    ax.set_ylabel("Frequency")

    histogram_canvas = FigureCanvasTkAgg(fig, master=hist_frame)
    histogram_canvas.draw()
    histogram_canvas.get_tk_widget().pack()

# Função para aplicar o threshold manual
def apply_manual_threshold(value):
    global threshold_value
    threshold_value = int(value)
    
    _, binarized_img = cv2.threshold(img_cv, threshold_value, 255, cv2.THRESH_BINARY)
    display_image(binarized_img, original=False)
    update_histogram()

# Função para aplicar o thresholding adaptativo
def apply_adaptive_threshold():
    if img_cv is None:
        return
    
    adaptive_img = cv2.adaptiveThreshold(img_cv, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY, 11, 2)
    display_image(adaptive_img, original=False)

# Função para aplicar o threshold binário automaticamente
def apply_auto_binary_threshold():
    global threshold_value
    if img_cv is None:
        return
    
    _, binarized_img = cv2.threshold(img_cv, threshold_value, 255, cv2.THRESH_BINARY)
    display_image(binarized_img, original=False)
    update_histogram()

# Função para aplicar filtros adicionais (Gaussian Blur, Average Blur, Laplacian, Sobel)
def apply_filter(filter_type):
    if img_cv is None:
        return

    if filter_type == "gaussian_blur":
        kernel_size = 15
        sigma = 2.0
        kernel = cv2.getGaussianKernel(kernel_size, sigma)
        kernel = kernel @ kernel.T
        filtered_img = cv2.filter2D(img_cv, -1, kernel)

    elif filter_type == "average_blur":
        kernel_size = 5
        kernel = np.ones((kernel_size, kernel_size), dtype=np.float32) / (kernel_size * kernel_size)
        filtered_img = cv2.filter2D(img_cv, -1, kernel)

    elif filter_type == "laplacian":
        laplacian_kernel = np.array([[0, 1, 0],
                                     [1, -4, 1],
                                     [0, 1, 0]], dtype=np.float32)
        filtered_img = cv2.filter2D(img_cv, -1, laplacian_kernel)

    elif filter_type == "sobel":
        sobelx = cv2.Sobel(img_cv, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(img_cv, cv2.CV_64F, 0, 1, ksize=3)
        sobel_magnitude = cv2.magnitude(sobelx, sobely)
        filtered_img = cv2.convertScaleAbs(sobel_magnitude)

    display_image(filtered_img, original=False)

# Funções de Operações Morfológicas
def apply_erosion():
    if img_cv is None:
        return
    
    kernel = np.ones((5, 5), np.uint8)
    eroded_img = cv2.erode(img_cv, kernel, iterations=1)
    display_image(eroded_img, original=False)

def apply_dilation():
    if img_cv is None:
        return
    
    kernel = np.ones((5, 5), np.uint8)
    dilated_img = cv2.dilate(img_cv, kernel, iterations=1)
    display_image(dilated_img, original=False)

def apply_opening():
    if img_cv is None:
        return
    
    kernel = np.ones((5, 5), np.uint8)
    opened_img = cv2.morphologyEx(img_cv, cv2.MORPH_OPEN, kernel)
    display_image(opened_img, original=False)

def apply_closing():
    if img_cv is None:
        return
    
    kernel = np.ones((5, 5), np.uint8)
    closed_img = cv2.morphologyEx(img_cv, cv2.MORPH_CLOSE, kernel)
    display_image(closed_img, original=False)

# Interface principal do Tkinter
root = tk.Tk()
root.title("Operações Morfológicas e Segmentação")

# Botão para carregar a imagem
load_button = tk.Button(root, text="Load Image", command=load_image)
load_button.pack(pady=5)

# Slider para ajustar o limiar manualmente
threshold_slider = tk.Scale(root, from_=0, to=255, orient="horizontal", command=apply_manual_threshold, label="Threshold Binário")
threshold_slider.set(threshold_value)
threshold_slider.pack(padx=5, pady=5)

# Botões de Segmentação
adaptive_thresh_button = tk.Button(root, text="Threshold Adaptativo", command=apply_adaptive_threshold)
adaptive_thresh_button.pack(pady=5)

auto_binary_thresh_button = tk.Button(root, text="Threshold Binário Automático", command=apply_auto_binary_threshold)
auto_binary_thresh_button.pack(pady=5)

# Botões de Operações Morfológicas
erosion_button = tk.Button(root, text="Erosão", command=apply_erosion)
erosion_button.pack(pady=5)

dilation_button = tk.Button(root, text="Dilatação", command=apply_dilation)
dilation_button.pack(pady=5)

opening_button = tk.Button(root, text="Abertura", command=apply_opening)
opening_button.pack(pady=5)

closing_button = tk.Button(root, text="Fechamento", command=apply_closing)
closing_button.pack(pady=5)

# Botões para aplicar filtros adicionais
gaussian_blur_button = tk.Button(root, text="Gaussian Blur", command=lambda: apply_filter("gaussian_blur"))
gaussian_blur_button.pack(pady=5)

average_blur_button = tk.Button(root, text="Average Blur", command=lambda: apply_filter("average_blur"))
average_blur_button.pack(pady=5)

laplacian_button = tk.Button(root, text="Laplacian", command=lambda: apply_filter("laplacian"))
laplacian_button.pack(pady=5)

sobel_button = tk.Button(root, text="Sobel", command=lambda: apply_filter("sobel"))
sobel_button.pack(pady=5)

# Criação das janelas para a imagem original, histograma e imagem processada
create_original_window()
create_histogram_window()
create_processed_window()

root.mainloop()
