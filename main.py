import tkinter as tk
from tkinter import filedialog
import cv2
from PIL import Image, ImageTk

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
    img_pil.thumbnail((max_size, max_size))  # Maintain aspect ratio
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

def apply_filter(filter_type):
    if img_cv is None:
        return
    if filter_type == "low_pass":
        filtered_img = cv2.GaussianBlur(img_cv, (15, 15), 0)
    elif filter_type == "high_pass":
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        filtered_img = cv2.Laplacian(gray, cv2.CV_64F)
        filtered_img = cv2.convertScaleAbs(filtered_img)
        filtered_img = cv2.cvtColor(filtered_img, cv2.COLOR_GRAY2BGR)
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
filters_menu.add_command(label="Low Pass Filter", command=lambda: apply_filter("low_pass"))
filters_menu.add_command(label="High Pass Filter", command=lambda: apply_filter("high_pass"))

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
low_pass_button = tk.Button(control_frame, text="Low Pass Filter", command=lambda: apply_filter("low_pass"))
low_pass_button.grid(row=0, column=0, padx=10)

high_pass_button = tk.Button(control_frame, text="High Pass Filter", command=lambda: apply_filter("high_pass"))
high_pass_button.grid(row=0, column=1, padx=10)

root.mainloop()
