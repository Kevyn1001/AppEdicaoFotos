import tkinter as tk
from tkinter import filedialog
import cv2
from PIL import Image, ImageTk
import numpy as np
import math

def load_image():
    global img_cv
    file_path = filedialog.askopenfilename()
    if file_path:
        img_cv = cv2.imread(file_path)
        display_image(img_cv, original=True)  # Exibe a imagem original
        refresh_canvas()

def display_image(img, original=False):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    
    # Obtém o tamanho da imagem original
    img_width, img_height = img_pil.size
    
    # Redimensiona a imagem para caber no canvas se for muito grande
    max_size = 500
    img_pil.thumbnail((max_size, max_size))  # Mantém a proporção
    img_tk = ImageTk.PhotoImage(img_pil)

    # Calcula a posição para centralizar a imagem dentro do canvas
    canvas_width, canvas_height = max_size, max_size
    x_offset = (canvas_width - img_pil.width) // 2
    y_offset = (canvas_height - img_pil.height) // 2

    if original:
        original_image_canvas.delete("all")  # Limpa a canvas
        original_image_canvas.image = img_tk  # Mantém a referência viva - garbage collection
        original_image_canvas.create_image(x_offset, y_offset, anchor=tk.NW, image=img_tk)
    else:
        edited_image_canvas.delete("all")  # Limpa a canvas
        edited_image_canvas.image = img_tk
        edited_image_canvas.create_image(x_offset, y_offset, anchor=tk.NW, image=img_tk)

def create_gaussian_kernel(size, sigma):
    """ Cria um kernel Gaussiano manualmente """
    kernel = np.zeros((size, size))
    center = size // 2
    sum_val = 0
    
    for i in range(size):
        for j in range(size):
            x = i - center
            y = j - center
            kernel[i, j] = math.exp(-(x**2 + y**2) / (2 * sigma**2))
            sum_val += kernel[i, j]
    
    # Normaliza o kernel
    kernel /= sum_val
    return kernel

def apply_convolution(image, kernel):
    """ Aplica a convolução usando o kernel fornecido à imagem """
    kernel_size = kernel.shape[0]
    pad_size = kernel_size // 2
    padded_img = np.pad(image, ((pad_size, pad_size), (pad_size, pad_size), (0, 0)), mode='constant')
    
    output_img = np.zeros_like(image)
    
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            for k in range(3):  # Para cada canal (R, G, B)
                output_img[i, j, k] = np.sum(padded_img[i:i+kernel_size, j:j+kernel_size, k] * kernel)
    
    return output_img

def create_mean_kernel(size):
    """ Cria um kernel de média (Filtro de Blur) manualmente """
    kernel = np.ones((size, size), dtype=np.float32)
    kernel /= size * size  # Normaliza para garantir que a soma seja 1
    return kernel

def apply_mean_blur_manual(image, kernel_size):
    """ Aplica o Filtro de Média (Blur) manualmente """
    kernel = create_mean_kernel(kernel_size)
    return apply_convolution(image, kernel)

def apply_filter(filter_type):
    if img_cv is None:
        return

    if filter_type == "gaussian_blur_manual":
        # Definindo os parâmetros para o filtro Gaussiano manual
        kernel_size = 15
        sigma = 2.0
        gaussian_kernel = create_gaussian_kernel(kernel_size, sigma)
        filtered_img = apply_convolution(img_cv, gaussian_kernel)

    elif filter_type == "average_blur":
        filtered_img = cv2.blur(img_cv, (5, 5))

    elif filter_type == "average_blur_manual":
        kernel_size = 5  # Tamanho do kernel 5x5
        filtered_img = apply_mean_blur_manual(img_cv, kernel_size)

    elif filter_type == "laplacian":
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        filtered_img = cv2.Laplacian(gray, cv2.CV_64F)
        filtered_img = cv2.convertScaleAbs(filtered_img)
        filtered_img = cv2.cvtColor(filtered_img, cv2.COLOR_GRAY2BGR)
    elif filter_type == "sobel":
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
        sobel_combined = cv2.sqrt(sobelx**2 + sobely**2)
        sobel_combined = cv2.convertScaleAbs(sobel_combined)
        filtered_img = cv2.cvtColor(sobel_combined, cv2.COLOR_GRAY2BGR)

    display_image(filtered_img, original=False)  # Exibe a imagem editada

def refresh_canvas():
    edited_image_canvas.delete("all")  # Limpa a canvas para exibir a nova imagem

# Definindo a GUI
root = tk.Tk()
root.title("Image Processing App")

# Define o tamanho da janela da aplicação
root.geometry("1200x600")

# Define a cor de fundo da janela
root.config(bg="#022a3b")

img_cv = None

# Cria o menu da aplicação
menu_bar = tk.Menu(root)
root.config(menu=menu_bar)

# File menu
file_menu = tk.Menu(menu_bar, tearoff=0)
menu_bar.add_cascade(label="File", menu=file_menu)
file_menu.add_command(label="Load Image", command=load_image)
file_menu.add_separator()
file_menu.add_command(label="Exit", command=root.quit)

# Filters menu
filters_menu = tk.Menu(menu_bar, tearoff=0)
menu_bar.add_cascade(label="Filters", menu=filters_menu)
filters_menu.add_command(label="Gaussian Blur (Manual)", command=lambda: apply_filter("gaussian_blur_manual"))
filters_menu.add_command(label="Average Blur (Manual)", command=lambda: apply_filter("average_blur_manual"))
filters_menu.add_command(label="Average Blur", command=lambda: apply_filter("average_blur"))
filters_menu.add_command(label="Laplacian", command=lambda: apply_filter("laplacian"))
filters_menu.add_command(label="Sobel", command=lambda: apply_filter("sobel"))

# Configura grid para centralizar o frame das imagens
root.grid_columnconfigure(0, weight=1)
root.grid_columnconfigure(2, weight=1)
root.grid_rowconfigure(0, weight=1)
root.grid_rowconfigure(2, weight=1)

# Cria um frame para as imagens
image_frame = tk.Frame(root, bg="#2e2e2e")
image_frame.grid(row=1, column=1, padx=20, pady=20)

# Cria a canvas para a imagem original com borda (sem background)
original_image_canvas = tk.Canvas(image_frame, width=500, height=500, bg="#2e2e2e", highlightthickness=1, highlightbackground="white")
original_image_canvas.grid(row=0, column=0, padx=10, pady=10)

# Cria a canvas para a imagem editada com borda (sem background)
edited_image_canvas = tk.Canvas(image_frame, width=500, height=500, bg="#2e2e2e", highlightthickness=1, highlightbackground="white")
edited_image_canvas.grid(row=0, column=1, padx=10, pady=10)

# Cria um painel de controle para os botões logo abaixo das imagens
control_frame = tk.Frame(root, bg="#022a3b")
control_frame.grid(row=2, column=1)  # Mantém o painel logo abaixo das imagens

# Adiciona botões para aplicar os filtros
gaussian_blur_button = tk.Button(control_frame, text="Gaussian Blur (Manual)", command=lambda: apply_filter("gaussian_blur_manual"))
gaussian_blur_button.grid(row=0, column=0, padx=10)

average_blur_button = tk.Button(control_frame, text="Average Blur", command=lambda: apply_filter("average_blur"))
average_blur_button.grid(row=0, column=1, padx=10)

average_blur_button_manual = tk.Button(control_frame, text="Average Blur (Manual)", command=lambda: apply_filter("average_blur_manual"))
average_blur_button_manual.grid(row=0, column=2, padx=10)

laplacian_button = tk.Button(control_frame, text="Laplacian", command=lambda: apply_filter("laplacian"))
laplacian_button.grid(row=0, column=3, padx=10)

sobel_button = tk.Button(control_frame, text="Sobel", command=lambda: apply_filter("sobel"))
sobel_button.grid(row=0, column=4, padx=10)

root.mainloop()
