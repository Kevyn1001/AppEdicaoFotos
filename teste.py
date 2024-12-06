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
root.title("HK PhotoEditor")
root.geometry("1200x600")

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

# Frame para botões e sliders
controls_frame = tk.Frame(root)
controls_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=350, pady=10)

# Botão para carregar a imagem
load_button = tk.Button(root, text="Load Image", command=load_image)
load_button.pack(pady=5)

# Organização dos botões em grid (3 botões por linha)
buttons = [
    ("Threshold Adaptativo", apply_adaptive_threshold),
    ("Threshold Automático", apply_auto_binary_threshold),
    ("Erosão", apply_erosion),
    ("Dilatação", apply_dilation),
    ("Abertura", apply_opening),
    ("Fechamento", apply_closing),
    ("Gaussian Blur", lambda: apply_filter("gaussian_blur")),
    ("Average Blur", lambda: apply_filter("average_blur")),
    ("Laplacian", lambda: apply_filter("laplacian")),
    ("Sobel", lambda: apply_filter("sobel")),
]

# Criando botões em grade
for i, (text, command) in enumerate(buttons):
    row, col = divmod(i, 3)  # Calcula a posição na grade (linha e coluna)
    tk.Button(controls_frame, text=text, command=command, width=20).grid(row=row, column=col, padx=5, pady=5)

# Slider para ajustar o limiar manualmente
threshold_slider = tk.Scale(
    controls_frame, 
    from_=0, 
    to=255, 
    orient="horizontal", 
    command=apply_manual_threshold, 
    label="Threshold Binário"
)
threshold_slider.set(threshold_value)
threshold_slider.grid(row=len(buttons) // 3 + 1, column=0, columnspan=3, pady=10)

root.mainloop()
