import tkinter as tk
from tkinter import filedialog
import cv2
from PIL import Image, ImageTk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Variáveis globais
img_cv = None
processed_img = None
threshold_value = 79
lowpass_kernel_size = 5
highpass_kernel_size = 5
morphology_kernel_size = 5
morphology_step = 0
filter_step = 0
filter_step_highpass = 0

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
    global histogram_canvas, processed_img, lowpass_kernel_size, highpass_kernel_size, morphology_kernel_size, threshold_value
    for widget in histogram_frame.winfo_children():
        widget.destroy()

    # Use a imagem processada, se disponível, caso contrário, a imagem original
    img_to_analyze = processed_img if processed_img is not None else img_cv

    if img_to_analyze is not None:
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.hist(img_to_analyze.ravel(), bins=256, range=(0, 256), color='gray')
        ax.axvline(threshold_value, color='red', linestyle='--', label='Threshold')  # Linha do threshold
        ax.axvline(lowpass_kernel_size * 10, color='blue', linestyle='--', label='Kernel Passa-Baixa')  # Linha do passa-baixa
        ax.axvline(highpass_kernel_size * 10, color='green', linestyle='--', label='Kernel Passa-Alta')  # Linha do passa-alta
        ax.axvline(morphology_kernel_size * 10, color='purple', linestyle='--', label='Kernel Morfologia')  # Linha da morfologia
        ax.set_title("Histogram")
        ax.set_xlabel("Pixel Intensity")
        ax.set_ylabel("Frequency")
        ax.legend()

        histogram_canvas = FigureCanvasTkAgg(fig, master=histogram_frame)
        histogram_canvas.draw()
        histogram_canvas.get_tk_widget().pack()

# Função para aplicar filtros de passa-baixa
def apply_lowpass_filter(value):
    global processed_img, lowpass_kernel_size
    if img_cv is None:
        return
    lowpass_kernel_size = int(value)
    kernel_size = max(3, lowpass_kernel_size if lowpass_kernel_size % 2 == 1 else lowpass_kernel_size + 1)

    if filter_step % 2 == 0:  # Gaussian Blur
        sigma = 1.5
        kernel = cv2.getGaussianKernel(kernel_size, sigma)
        kernel = kernel @ kernel.T
        processed_img = cv2.filter2D(img_cv, -1, kernel)
    elif filter_step % 2 == 1:  # Average Blur
        kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size ** 2)
        processed_img = cv2.filter2D(img_cv, -1, kernel)

    display_image(processed_img, original=False)
    update_histogram()

# Função para aplicar filtros de passa-alta
def apply_highpass_filter(value):
    global processed_img, highpass_kernel_size
    if img_cv is None:
        return
    highpass_kernel_size = int(value)
    kernel_size = max(3, highpass_kernel_size if highpass_kernel_size % 2 == 1 else highpass_kernel_size + 1)

    if filter_step_highpass % 2 == 0:  # Sobel
        sobelx = cv2.Sobel(img_cv, cv2.CV_64F, 1, 0, ksize=kernel_size)
        sobely = cv2.Sobel(img_cv, cv2.CV_64F, 0, 1, ksize=kernel_size)
        processed_img = cv2.magnitude(sobelx, sobely)
        processed_img = cv2.convertScaleAbs(processed_img)
    elif filter_step_highpass % 2 == 1:  # Laplacian
        processed_img = cv2.Laplacian(img_cv, cv2.CV_64F, ksize=kernel_size)
        processed_img = cv2.convertScaleAbs(processed_img)

    display_image(processed_img, original=False)
    update_histogram()

# Função para alternar o tipo de filtro de passa-alta
def change_highpass_filter():
    global filter_step_highpass
    filter_step_highpass = (filter_step_highpass + 1) % 2
    highpass_label.config(text=["Sobel", "Laplacian"][filter_step_highpass])

# Função para alternar o tipo de filtro de passa-baixa
def change_lowpass_filter():
    global filter_step
    filter_step = (filter_step + 1) % 2
    filter_label.config(text=["Gaussian Blur", "Average Blur"][filter_step])

# Função para aplicar o threshold manual
def apply_manual_threshold(value):
    global threshold_value, processed_img
    if img_cv is None:
        return  # Verifica se a imagem foi carregada antes de aplicar o threshold
    threshold_value = int(value)
    _, binarized_img = cv2.threshold(img_cv, threshold_value, 255, cv2.THRESH_BINARY)
    if binarized_img is not None:
        display_image(binarized_img, original=False)
        update_histogram()

# Função para ajustar o slider de morfologia
def apply_morphology(value):
    global processed_img, morphology_kernel_size
    if img_cv is None:
        return
    morphology_kernel_size = int(value)
    kernel = np.ones((morphology_kernel_size, morphology_kernel_size), np.uint8)

    if morphology_step % 4 == 0:  # Erosão
        processed_img = cv2.erode(img_cv, kernel, iterations=1)
    elif morphology_step % 4 == 1:  # Dilatação
        processed_img = cv2.dilate(img_cv, kernel, iterations=1)
    elif morphology_step % 4 == 2:  # Abertura
        processed_img = cv2.morphologyEx(img_cv, cv2.MORPH_OPEN, kernel)
    elif morphology_step % 4 == 3:  # Fechamento
        processed_img = cv2.morphologyEx(img_cv, cv2.MORPH_CLOSE, kernel)

    display_image(processed_img, original=False)
    update_histogram()

# Função para mudar o tipo de morfologia ao pressionar um botão
def change_morphology():
    global morphology_step
    morphology_step = (morphology_step + 1) % 4
    morphology_label.config(text=["Erosão", "Dilatação", "Abertura", "Fechamento"][morphology_step])

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

# Frame para botões e sliders
controls_frame = tk.Frame(root)
controls_frame.pack(side=tk.TOP, fill=tk.Y, anchor='center', padx=10)

# Frame para histograma
histogram_frame = tk.Frame(image_frame, width=300, height=300, bg="gray")
histogram_frame.pack(side=tk.LEFT, padx=10)

# Canvas para imagem processada
processed_image_canvas = tk.Canvas(image_frame, width=300, height=300, bg="gray")
processed_image_canvas.pack(side=tk.LEFT, padx=10)

# Botão para carregar a imagem (colocado no controls_frame)
load_button = tk.Button(controls_frame, text="Load Image", command=load_image)
load_button.grid(row=0, column=0, columnspan=5, pady=10)

# Slider para ajustar o tamanho do kernel de filtros passa-alta
highpass_slider = tk.Scale(
    controls_frame,
    from_=3,
    to=21,
    orient="horizontal",
    command=apply_highpass_filter,
    label="Kernel Passa-Alta"
)
highpass_slider.set(5)
highpass_slider.grid(row=2, column=4, pady=5)

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
threshold_slider.grid(row=1, column=2, pady=5)

# Slider para ajustar o kernel da morfologia
morphology_slider = tk.Scale(
    controls_frame,
    from_=1,
    to=21,
    orient="horizontal",
    command=apply_morphology,
    label="Kernel Morfologia"
)
morphology_slider.set(5)
morphology_slider.grid(row=2, column=0, pady=1)

# Botão e rótulo para mudar o tipo de morfologia
morphology_label = tk.Label(controls_frame, text="Erosão")
morphology_label.grid(row=3, column=1)

morphology_button = tk.Button(controls_frame, text="Alterar Morfologia", command=change_morphology)
morphology_button.grid(row=3, column=0)

# Slider para ajustar o tamanho do kernel de filtros passa-baixa
filter_slider = tk.Scale(
    controls_frame,
    from_=3,
    to=21,
    orient="horizontal",
    command=apply_lowpass_filter,
    label="Kernel Passa-Baixa"
)
filter_slider.set(5)
filter_slider.grid(row=2, column=2, pady=5)

# Botão e rótulo para mudar o tipo de filtro de passa-baixa
filter_button = tk.Button(controls_frame, text="Alterar Filtro", command=change_lowpass_filter)
filter_button.grid(row=3, column=2)

filter_label = tk.Label(controls_frame, text="Gaussian Blur")
filter_label.grid(row=3, column=3)

# Botão e rótulo para mudar o tipo de filtro de passa-alta
highpass_button = tk.Button(controls_frame, text="Alterar Filtro", command=change_highpass_filter)
highpass_button.grid(row=3, column=4)

highpass_label = tk.Label(controls_frame, text="Sobel")
highpass_label.grid(row=3, column=6)


root.mainloop()
