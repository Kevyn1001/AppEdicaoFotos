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
    
    # Redimensiona a imagem se for muito grande
    max_size = 500
    img_pil.thumbnail((max_size, max_size)) 
    img_tk = ImageTk.PhotoImage(img_pil)
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
    # Cria um kernel Gaussiano manualmente
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

def apply_convolution_manual(image, kernel):
    # Aplica a convolução manualmente usando o kernel fornecido à imagem.
    rows = len(image)
    cols = len(image[0])
    k_size = len(kernel)
    offset = k_size // 2

    # Cria a matriz de saída com zeros
    output_image = np.zeros((rows, cols))  # Usamos o numpy para garantir a matriz 2D

    # Realiza a convolução
    for i in range(offset, rows - offset):
        for j in range(offset, cols - offset):
            sum_val = 0  # Reseta `sum_val` para cada posição de pixel
            for ki in range(-offset, offset + 1):
                for kj in range(-offset, offset + 1):
                    # Realiza a multiplicação pixel a pixel
                    pixel_val = image[i + ki][j + kj]
                    kernel_val = kernel[offset + ki][offset + kj]
                    sum_val += pixel_val * kernel_val
            output_image[i][j] = sum_val  # Garante que `sum_val` é um valor escalar
    return output_image

def apply_convolution_manual_simple(image, kernel):
    # Versão simplificada da convolução para filtros básicos.
    rows, cols, channels = image.shape
    k_size = kernel.shape[0]
    offset = k_size // 2
    output_image = np.zeros_like(image)  # Saída para imagem colorida

    for ch in range(channels):  # Para cada canal da imagem (R, G, B)
        for i in range(offset, rows - offset):
            for j in range(offset, cols - offset):
                sum_val = 0
                for ki in range(-offset, offset + 1):
                    for kj in range(-offset, offset + 1):
                        sum_val += image[i + ki, j + kj, ch] * kernel[offset + ki, offset + kj]
                output_image[i, j, ch] = sum_val
    return output_image

def convert_to_gray(image):
    # Converte a imagem colorida para tons de cinza
    rows, cols, _ = image.shape
    gray_image = np.zeros((rows, cols))  # Mudar para array do NumPy
    for i in range(rows):
        for j in range(cols):
            # Média dos canais R, G e B
            gray_image[i][j] = 0.2989 * image[i][j][2] + 0.5870 * image[i][j][1] + 0.1140 * image[i][j][0]
    return gray_image

def apply_laplacian_manual_v2(image):
    #Aplica o filtro Laplaciano manual 
    # Converte para escala de cinza (2D)
    gray_image = convert_to_gray(image)

    # Define o kernel Laplaciano
    laplacian_kernel = [[0, 1, 0],
                        [1, -4, 1],
                        [0, 1, 0]]

    # Aplica a convolução manualmente
    laplacian_img = apply_convolution_manual(gray_image, laplacian_kernel)

    # Ajusta os valores para estarem no intervalo 0-255
    laplacian_img = np.clip(laplacian_img, 0, 255)

    # Converte de volta para uma imagem RGB para exibir no Tkinter
    laplacian_rgb = np.zeros((len(laplacian_img), len(laplacian_img[0]), 3), dtype=np.uint8)
    for i in range(len(laplacian_img)):
        for j in range(len(laplacian_img[0])):
            laplacian_rgb[i][j] = [laplacian_img[i][j]] * 3
    return laplacian_rgb

def create_avarage_kernel(size):
    #Cria um kernel de média (Filtro de Blur) manualmente
    kernel = np.ones((size, size), dtype=np.float32)
    kernel /= size * size  # Normaliza para garantir que a soma seja 1
    return kernel

def apply_gaussian_blur_manual(image, kernel_size, sigma):
    kernel = create_gaussian_kernel(kernel_size, sigma)
    return apply_convolution_manual_simple(image, kernel)

def apply_avarage_blur_manual(image, kernel_size):
    # Cria um kernel de média com o tamanho fornecido
    kernel = create_avarage_kernel(kernel_size)
    
    return apply_convolution_manual_simple(image, kernel)


def create_sobel_kernels():
    Gx = np.array([[-1, 0, 1],
                  [-2, 0, 2],
                  [-1, 0, 1]], dtype=np.float32)
    
    Gy = np.array([[1, 2, 1],
                  [0, 0, 0],
                  [-1, -2, -1]], dtype=np.float32)
    return Gx, Gy

def apply_sobel_manual(image):
    # Converte para escala de cinza
    gray = convert_to_gray(image)

    # Cria os kernels Sobel
    Gx, Gy = create_sobel_kernels()
    
    # Aplica a convolução 
    sobelx = apply_convolution_manual(gray, Gx)
    sobely = apply_convolution_manual(gray, Gy)
    
    # Calcula a magnitude do gradiente
    sobel_magnitude = np.sqrt(sobelx**2 + sobely**2)
    
    # Normaliza para a faixa 0-255
    sobel_magnitude = np.clip(sobel_magnitude, 0, 255).astype(np.uint8)
    
    # Converte de volta para BGR para exibir
    sobel_bgr = cv2.cvtColor(sobel_magnitude, cv2.COLOR_GRAY2BGR)
    return sobel_bgr

def apply_filter(filter_type):
    if img_cv is None:
        return

    filtered_img = None

    if filter_type == "gaussian_blur_manual":
        kernel_size = 15
        sigma = 2.0
        filtered_img = apply_gaussian_blur_manual(img_cv, kernel_size, sigma)

    elif filter_type == "average_blur_manual":
        kernel_size = 5
        filtered_img = apply_avarage_blur_manual(img_cv, kernel_size)

    elif filter_type in ["laplacian_manual", "sobel_manual"]:
        if filter_type == "laplacian_manual":
            filtered_img = apply_laplacian_manual_v2(img_cv)
        elif filter_type == "sobel_manual":
            filtered_img = apply_sobel_manual(img_cv)

    display_image(filtered_img, original=False)

def refresh_canvas():
    edited_image_canvas.delete("all")  # Limpa a canvas para exibir a nova imagem

# GUI
root = tk.Tk()
root.title("HK ImageEditor")

# Tamanho da janela da aplicação
root.geometry("1200x600")

# Cor de fundo da janela
root.config(bg="#022a3b")

img_cv = None

# Menu da aplicação
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
filters_menu.add_command(label="Laplacian (Manual)", command=lambda: apply_filter("laplacian_manual"))
filters_menu.add_command(label="Sobel (Manual)", command=lambda: apply_filter("sobel_manual"))

# Grid para centralizar o frame das imagens
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

# Botões logo abaixo das imagens
control_frame = tk.Frame(root, bg="#022a3b")
control_frame.grid(row=2, column=1)  # Mantém o painel logo abaixo das imagens

# Botões para aplicar os filtros
gaussian_blur_button = tk.Button(control_frame, text="Gaussian Blur (Manual)", command=lambda: apply_filter("gaussian_blur_manual"))
gaussian_blur_button.grid(row=0, column=0, padx=10)

average_blur_button_manual = tk.Button(control_frame, text="Average Blur (Manual)", command=lambda: apply_filter("average_blur_manual"))
average_blur_button_manual.grid(row=0, column=2, padx=10)

laplacian_button = tk.Button(control_frame, text="Laplacian (Manual)", command=lambda: apply_filter("laplacian_manual"))
laplacian_button.grid(row=0, column=3, padx=10)

sobel_button_manual = tk.Button(control_frame, text="Sobel (Manual)", command=lambda: apply_filter("sobel_manual"))
sobel_button_manual.grid(row=0, column=4, padx=10)

root.mainloop()
