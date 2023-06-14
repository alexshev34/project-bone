import os
import random
import tkinter
from tkinter import *
from tkinter import filedialog
import customtkinter as ctk
import pyautogui
import pygetwindow
from tkinter import ttk
from PIL import ImageTk, Image
from netconfig import NetConfig
from densenet import DenseNet
import cv2

from predictions import predict



project_folder = os.path.dirname(os.path.abspath(__file__))
folder_path = project_folder + '/images/'

filename = ""



class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        number = str(random.randint(100, 600)) 
        
        photo = Image.open(folder_path + "main.jpg")
        photo_resized = photo.resize((int(256 / photo.height * photo.width), 256))  # new width & height
        photo = ImageTk.PhotoImage(photo_resized)

    
        

        self.info_label = ctk.CTkLabel(master=self.main_frame, text="", wraplength=300, font=(ctk.CTkFont("Arial"), 18))
        self.info_label.pack(pady=10, padx=10)

        self.upload_btn = ctk.CTkButton(master=self.main_frame, text="Загрузить", fg_color="green", command=self.l_photo)
        self.upload_btn.pack(pady=0, padx=1)

        self.frame2 = ctk.CTkFrame(master=self.main_frame, fg_color="transparent", width=256, height=256)
        self.frame2.pack(pady=10, padx=1)

        

        self.photo_lb = ctk.CTkLabel(master=self.frame2, text="", image=photo)
        self.photo_lb.pack(pady=1, padx=10)


        self.predict_btn = ctk.CTkButton(master=self.main_frame, text="Прогноз", fg_color="green", command=self.pr_start)
        self.predict_btn.pack(pady=0, padx=1)


        self.title("Распознование переломов костей")
        self.iconbitmap('info.jpg')
        self.geometry(f"{645}x{950}")
        self.head_frame = ctk.CTkFrame(master=self)
        self.head_frame.pack(pady=20, padx=60, fill="both", expand=True)
        self.main_frame = ctk.CTkFrame(master=self)
        self.main_frame.pack(pady=20, padx=60, fill="both", expand=True)
        self.head_label = ctk.CTkLabel(master=self.head_frame, text="Распознавание переломов костей",font=(ctk.CTkFont("Arial"), 28))
        self.head_label.pack(pady=20, padx=10, anchor="nw", side="left")

        self.result_frame = ctk.CTkFrame(master=self.main_frame, fg_color="transparent", width=200, height=100)
        self.result_frame.pack(pady=5, padx=5)

        self.loader_label = ctk.CTkLabel(master=self.main_frame, width=100, height=100, text="")
        self.loader_label.pack(pady=3, padx=3)

        self.res1_label = ctk.CTkLabel(master=self.result_frame, text="")
        self.res1_label.pack(pady=0, padx=0)

        self.res2_label = ctk.CTkLabel(master=self.result_frame, text="")
        self.res2_label.pack(pady=0, padx=0)


     

        

    def el_good(self):
            self.res1_label.configure(text="Локоть", font=(ctk.CTkFont("Arial"), 24))
            self.res2_label.configure(text_color="GREEN", text="Положительный", font=(ctk.CTkFont("Arial"), 24))  

    def h_goode(self):
            self.res1_label.configure(text="Рука", font=(ctk.CTkFont("Arial"), 24))
            self.res2_label.configure(text_color="GREEN", text="Положительный", font=(ctk.CTkFont("Arial"), 24))   

    def sh_good(self):
            self.res1_label.configure(text="Плечо", font=(ctk.CTkFont("Arial"), 24))
            self.res2_label.configure(text_color="GREEN", text="Положительный", font=(ctk.CTkFont("Arial"), 24))   





    def l_photo(self):
        global filename
        f_types = [("All Files", "*.*")]
        filename = filedialog.askopenfilename(filetypes=f_types, initialdir=project_folder+'/test/Wrist/')
        self.res2_label.configure(text="")
        self.res1_label.configure(text="")
        self.photo_lb.configure(self.frame2, text="", image="")
        photo = Image.open(filename)
        photo_resized = photo.resize((int(256 / photo.height * photo.width), 256))  
        photo = ImageTk.PhotoImage(photo_resized)
        self.photo_lb.configure(self.frame2, image=photo, text="")
        self.photo_lb.image = photo
  

    def pr_start(self):
        global filename
        bone_type_total = predict(filename)
        total = predict(filename, bone_type_total)
        print(total)
        if total == 'fractured':
            self.res2_label.configure(text_color="RED", text="Отрицательный", font=(ctk.CTkFont("Arial"), 24))
        else:
            self.res2_label.configure(text_color="GREEN", text="Положительный", font=(ctk.CTkFont("Arial"), 24))
        bone_type_total = predict(filename, "Parts")


        if bone_type_total == "Elbow":
            self.res1_label.configure(text="Локоть", font=(ctk.CTkFont("Arial"), 24))  
        elif bone_type_total == "Hand":
            self.res1_label.configure(text="Рука", font=(ctk.CTkFont("Arial"), 24)) 
        elif bone_type_total == "Shoulder":
            self.res1_label.configure(text="Плечо", font=(ctk.CTkFont("Arial"), 24))  
        else:
            self.res1_label.configure(text="Кость", font=(ctk.CTkFont("Roboto"), 24))       





if __name__ == "__main__":
    app = App()
    app.mainloop()
