import tkinter as tk
from tkinter import filedialog
import cv2
from PIL import Image, ImageTk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Variáveis globais para imagem e limiar
img_cv = None
threshold_value = 127

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
    # Limpa o canvas do histograma
    histogram_canvas.get_tk_widget().pack_forget()
    
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.hist(img_cv.ravel(), bins=256, range=(0, 256), color='gray')
    ax.axvline(threshold_value, color='red', linestyle='--')  # Linha de limiar
    ax.set_title("Histogram")
    ax.set_xlabel("Pixel Intensity")
    ax.set_ylabel("Frequency")

    # Renderiza o histograma no Tkinter
    histogram_canvas = FigureCanvasTkAgg(fig, master=hist_frame)
    histogram_canvas.draw()
    histogram_canvas.get_tk_widget().pack()

# Função para aplicar o threshold manual
def apply_manual_threshold(value):
    global threshold_value
    threshold_value = int(value)  # Atualiza o valor do limiar
    
    _, binarized_img = cv2.threshold(img_cv, threshold_value, 255, cv2.THRESH_BINARY)
    display_image(binarized_img, original=False)
    update_histogram()  # Atualiza o histograma com a nova linha de limiar

# Função genérica para filtros (caso precise aplicar filtros adicionais)
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

# Interface do Tkinter com scrollbar
root = tk.Tk()
root.title("Threshold Manual")

# Frame principal com canvas e barra de rolagem
main_frame = tk.Frame(root)
main_frame.pack(fill="both", expand=True)

canvas = tk.Canvas(main_frame)
scrollbar = tk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
scrollable_frame = tk.Frame(canvas)

scrollable_frame.bind(
    "<Configure>",
    lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
)

canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
canvas.configure(yscrollcommand=scrollbar.set)

canvas.pack(side="left", fill="both", expand=True)
scrollbar.pack(side="right", fill="y")

# Canvas para exibir a imagem original e a imagem processada
original_image_canvas = tk.Canvas(scrollable_frame, width=300, height=300, bg="gray")
original_image_canvas.grid(row=0, column=0, padx=10, pady=10)

hist_frame = tk.Frame(scrollable_frame, width=300, height=300, bg="gray")  # Frame para o histograma
hist_frame.grid(row=0, column=1, padx=10, pady=10)
histogram_canvas = FigureCanvasTkAgg(plt.figure(figsize=(4, 3)), master=hist_frame)
histogram_canvas.get_tk_widget().pack()

processed_image_canvas = tk.Canvas(scrollable_frame, width=300, height=300, bg="gray")
processed_image_canvas.grid(row=0, column=2, padx=10, pady=10)

# Slider para ajustar o limiar manualmente
threshold_slider = tk.Scale(scrollable_frame, from_=0, to=255, orient="horizontal", command=apply_manual_threshold, label="Threshold")
threshold_slider.set(threshold_value)
threshold_slider.grid(row=1, column=0, columnspan=3, padx=10, pady=10, sticky="we")

# Botão para carregar a imagem
load_button = tk.Button(scrollable_frame, text="Load Image", command=load_image)
load_button.grid(row=2, column=0, columnspan=3, pady=10)

# Botões para aplicar filtros
control_frame = tk.Frame(scrollable_frame)
control_frame.grid(row=3, column=0, columnspan=3)

gaussian_blur_button = tk.Button(control_frame, text="Gaussian Blur", command=lambda: apply_filter("gaussian_blur"))
gaussian_blur_button.grid(row=0, column=0, padx=10)

average_blur_button = tk.Button(control_frame, text="Average Blur", command=lambda: apply_filter("average_blur"))
average_blur_button.grid(row=0, column=1, padx=10)

laplacian_button = tk.Button(control_frame, text="Laplacian", command=lambda: apply_filter("laplacian"))
laplacian_button.grid(row=0, column=2, padx=10)

sobel_button = tk.Button(control_frame, text="Sobel", command=lambda: apply_filter("sobel"))
sobel_button.grid(row=0, column=3, padx=10)

root.mainloop()
