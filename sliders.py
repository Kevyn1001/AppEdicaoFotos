import tkinter as tk
from tkinter import filedialog
import cv2
from PIL import Image, ImageTk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Variáveis globais
img_cv = None
threshold_value = 79
gaussian_kernel_size = 15
gaussian_sigma = 2.0
sobel_sensitivity = 1
morph_kernel_size = 5
adaptive_block_size = 11
adaptive_c = 2

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
    for widget in histogram_frame.winfo_children():
        widget.destroy()

    fig, ax = plt.subplots(figsize=(4, 3))
    ax.hist(img_cv.ravel(), bins=256, range=(0, 256), color='gray')
    ax.axvline(threshold_value, color='red', linestyle='--')  # Linha de limiar
    ax.set_title("Histogram")
    ax.set_xlabel("Pixel Intensity")
    ax.set_ylabel("Frequency")

    histogram_canvas = FigureCanvasTkAgg(fig, master=histogram_frame)
    histogram_canvas.draw()
    histogram_canvas.get_tk_widget().pack()

# Função para aplicar o filtro gaussiano
def apply_gaussian_blur():
    if img_cv is None:
        return
    kernel = cv2.getGaussianKernel(gaussian_kernel_size, gaussian_sigma)
    kernel = kernel @ kernel.T
    filtered_img = cv2.filter2D(img_cv, -1, kernel)
    display_image(filtered_img, original=False)

# Função para aplicar o filtro Sobel
def apply_sobel():
    if img_cv is None:
        return
    sobelx = cv2.Sobel(img_cv, cv2.CV_64F, 1, 0, ksize=sobel_sensitivity)
    sobely = cv2.Sobel(img_cv, cv2.CV_64F, 0, 1, ksize=sobel_sensitivity)
    sobel_magnitude = cv2.magnitude(sobelx, sobely)
    filtered_img = cv2.convertScaleAbs(sobel_magnitude)
    display_image(filtered_img, original=False)

# Funções para ajustar operações morfológicas
def apply_morph_operation(operation):
    if img_cv is None:
        return
    kernel = np.ones((morph_kernel_size, morph_kernel_size), np.uint8)
    if operation == "erosion":
        processed_img = cv2.erode(img_cv, kernel, iterations=1)
    elif operation == "dilation":
        processed_img = cv2.dilate(img_cv, kernel, iterations=1)
    elif operation == "opening":
        processed_img = cv2.morphologyEx(img_cv, cv2.MORPH_OPEN, kernel)
    elif operation == "closing":
        processed_img = cv2.morphologyEx(img_cv, cv2.MORPH_CLOSE, kernel)
    display_image(processed_img, original=False)

# Função para aplicar threshold adaptativo
def apply_adaptive_threshold():
    if img_cv is None:
        return
    adaptive_img = cv2.adaptiveThreshold(img_cv, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY, adaptive_block_size, adaptive_c)
    display_image(adaptive_img, original=False)

# Interface principal do Tkinter
root = tk.Tk()
root.title("HK PhotoEditor")
root.geometry("1200x700")

# Frame para exibir imagens
image_frame = tk.Frame(root, width=900, height=300, bg="white")
image_frame.pack(side=tk.TOP, fill=tk.BOTH, padx=10, pady=10)

# Canvas para imagem original
original_image_canvas = tk.Canvas(image_frame, width=300, height=300, bg="gray")
original_image_canvas.pack(side=tk.LEFT, padx=10)

# Frame para histograma
histogram_frame = tk.Frame(image_frame, width=300, height=300, bg="gray")
histogram_frame.pack(side=tk.LEFT, padx=10)

# Canvas para imagem processada
processed_image_canvas = tk.Canvas(image_frame, width=300, height=300, bg="gray")
processed_image_canvas.pack(side=tk.LEFT, padx=10)

# Frame para controles
controls_frame = tk.Frame(root)
controls_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=10)

# Botão para carregar a imagem
load_button = tk.Button(root, text="Load Image", command=load_image)
load_button.pack(pady=5)

# Sliders para ajustar parâmetros
tk.Scale(controls_frame, from_=3, to=31, resolution=2, orient="horizontal",
         label="Kernel Gaussiano", command=lambda val: setattr(globals(), 'gaussian_kernel_size', int(val))).pack(pady=5)
tk.Scale(controls_frame, from_=0.5, to=10, resolution=0.1, orient="horizontal",
         label="Sigma Gaussiano", command=lambda val: setattr(globals(), 'gaussian_sigma', float(val))).pack(pady=5)
tk.Scale(controls_frame, from_=1, to=5, orient="horizontal",
         label="Sensibilidade Sobel", command=lambda val: setattr(globals(), 'sobel_sensitivity', int(val))).pack(pady=5)
tk.Scale(controls_frame, from_=3, to=21, resolution=2, orient="horizontal",
         label="Kernel Morfológico", command=lambda val: setattr(globals(), 'morph_kernel_size', int(val))).pack(pady=5)
tk.Scale(controls_frame, from_=3, to=31, resolution=2, orient="horizontal",
         label="Blocos Threshold Adaptativo", command=lambda val: setattr(globals(), 'adaptive_block_size', int(val))).pack(pady=5)
tk.Scale(controls_frame, from_=-10, to=10, orient="horizontal",
         label="C Threshold Adaptativo", command=lambda val: setattr(globals(), 'adaptive_c', int(val))).pack(pady=5)

root.mainloop()
