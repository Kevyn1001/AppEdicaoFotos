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
    
    img_width, img_height = img_pil.size
    
    max_size = 500
    img_pil.thumbnail((max_size, max_size)) 
    img_tk = ImageTk.PhotoImage(img_pil)
    canvas_width, canvas_height = max_size, max_size
    x_offset = (canvas_width - img_pil.width) // 2
    y_offset = (canvas_height - img_pil.height) // 2

    if original:
        original_image_canvas.delete("all")
        original_image_canvas.image = img_tk
        original_image_canvas.create_image(x_offset, y_offset, anchor=tk.NW, image=img_tk)
    else:
        edited_image_canvas.delete("all")
        edited_image_canvas.image = img_tk
        edited_image_canvas.create_image(x_offset, y_offset, anchor=tk.NW, image=img_tk)

def create_gaussian_kernel(size, sigma):
    # Cria um kernel Gaussiano usando cv2
    k = cv2.getGaussianKernel(size, sigma)
    kernel = k @ k.T  # Multiplica a coluna pelo transposto para obter o kernel 2D
    return kernel

def apply_filter(filter_type):
    if img_cv is None:
        return

    if filter_type == "gaussian_blur":
        kernel_size = 15
        sigma = 2.0
        kernel = create_gaussian_kernel(kernel_size, sigma)
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
        filtered_img = cv2.convertScaleAbs(sobel_magnitude)  # Converte para faixa de 0 a 255

    display_image(filtered_img, original=False)

def refresh_canvas():
    edited_image_canvas.delete("all")

# GUI
root = tk.Tk()
root.title("HK ImageEditor")

root.geometry("1200x600")
root.config(bg="#022a3b")

img_cv = None

menu_bar = tk.Menu(root)
root.config(menu=menu_bar)

file_menu = tk.Menu(menu_bar, tearoff=0)
menu_bar.add_cascade(label="File", menu=file_menu)
file_menu.add_command(label="Load Image", command=load_image)
file_menu.add_separator()
file_menu.add_command(label="Exit", command=root.quit)

filters_menu = tk.Menu(menu_bar, tearoff=0)
menu_bar.add_cascade(label="Filters", menu=filters_menu)
filters_menu.add_command(label="Gaussian Blur", command=lambda: apply_filter("gaussian_blur"))
filters_menu.add_command(label="Average Blur", command=lambda: apply_filter("average_blur"))
filters_menu.add_command(label="Laplacian", command=lambda: apply_filter("laplacian"))
filters_menu.add_command(label="Sobel", command=lambda: apply_filter("sobel"))

root.grid_columnconfigure(0, weight=1)
root.grid_columnconfigure(2, weight=1)
root.grid_rowconfigure(0, weight=1)
root.grid_rowconfigure(2, weight=1)

image_frame = tk.Frame(root, bg="#2e2e2e")
image_frame.grid(row=1, column=1, padx=20, pady=20)

original_image_canvas = tk.Canvas(image_frame, width=500, height=500, bg="#2e2e2e", highlightthickness=1, highlightbackground="white")
original_image_canvas.grid(row=0, column=0, padx=10, pady=10)

edited_image_canvas = tk.Canvas(image_frame, width=500, height=500, bg="#2e2e2e", highlightthickness=1, highlightbackground="white")
edited_image_canvas.grid(row=0, column=1, padx=10, pady=10)

control_frame = tk.Frame(root, bg="#022a3b")
control_frame.grid(row=2, column=1)

gaussian_blur_button = tk.Button(control_frame, text="Gaussian Blur", command=lambda: apply_filter("gaussian_blur"))
gaussian_blur_button.grid(row=0, column=0, padx=10)

average_blur_button = tk.Button(control_frame, text="Average Blur", command=lambda: apply_filter("average_blur"))
average_blur_button.grid(row=0, column=2, padx=10)

laplacian_button = tk.Button(control_frame, text="Laplacian", command=lambda: apply_filter("laplacian"))
laplacian_button.grid(row=0, column=3, padx=10)

sobel_button = tk.Button(control_frame, text="Sobel", command=lambda: apply_filter("sobel"))
sobel_button.grid(row=0, column=4, padx=10)

root.mainloop()
