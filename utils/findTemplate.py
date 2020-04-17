import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
scale = 80
import os
meth = 'cv2.TM_CCOEFF_NORMED'

def selectFiles():
    return filedialog.askopenfilename(initialdir = os.getcwd(),title = "Select a template",filetypes = ( ("png files",".png"),("jpeg files","*.jpg")))


class FindTemp():
    def __init__(self, root, image_paths):
        self.root = root
        self.img_paths = image_paths
        self.count_img = 0
        self.scale_percent = scale/100
        self.root.protocol("WM_DELETE_WINDOW", self.abort)
        #self.root.bind("<Key>", self.dragOrResize)
        self.createTkImage()
        self.image_frame = tk.Frame(self.root, width=self.img_shape[1], height=self.img_shape[0], bd=1)
        self.image_frame.pack(side=tk.LEFT, expand = True, fill = tk.BOTH)
        self.canvas = tk.Canvas(self.image_frame, width=self.img_shape[1], height=self.img_shape[0], cursor="cross")
        self.canvas.pack(side="top", fill="both", expand=True)
        self.img_item = self.canvas.create_image(0, 0, anchor="nw", image=self.tk_im)
        self.canvas.pack(anchor=tk.NW)
        self.resetButton = tk.Button(self.image_frame, text="Reset", command=self.reset)
        self.resetButton.pack(side=tk.RIGHT, anchor=tk.SE)
        self.skipButton = tk.Button(self.image_frame, text="Skip", command=self.skip_image)
        self.skipButton.pack(side=tk.RIGHT, anchor=tk.SE)
        self.abortButton = tk.Button(self.image_frame, text="Exit", command=self.abort)
        self.abortButton.pack(side=tk.RIGHT, anchor=tk.SE)
        self.template_frame = tk.Frame(self.root, bd=1)
        self.template_frame.pack(side=tk.RIGHT)
        self.template_canvas = tk.Canvas(self.template_frame)
        self.template_canvas.pack()
        self.templateIN = False
        self.insTemplateButton = tk.Button(self.template_frame, text='Insert Template',
                                           command=self.insert_template).pack(side=tk.BOTTOM, anchor=tk.SE)
        self.coord_label = tk.Label(self.template_frame, text='(Top Left),(Bottom Right)')
        self.coord_label.pack(side=tk.BOTTOM)
        self.coord_value = tk.Text(self.template_frame, height=1, width=20)
        self.coord_value.pack(side=tk.BOTTOM, fill='x')

    def matchTemplate(self):
        w, h = self.img_template.shape[::-1]
        method = eval(meth)
        res = cv2.matchTemplate(self.img, self.img_template, method)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        top_left = max_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)
        self.rect = self.canvas.create_rectangle(top_left[0], top_left[1], bottom_right[0], bottom_right[1], outline='red', width=4, tags=("rec",))
        text = str(top_left)+','+str(bottom_right)
        self.coord_value.delete(1.0, tk.END)
        self.coord_value.insert(tk.END, text)

    def resize_image(self, input, scale_percent=0.8):
        width = int(input.shape[1] * scale_percent)
        height = int(input.shape[0] * scale_percent)
        dim = (width, height)
        return cv2.resize(input, dim, interpolation=cv2.INTER_AREA)

    def insert_template(self):
        path = selectFiles()
        if len(path)>0:
            if self.templateIN:
                self.reset()
            self.img_template = self.resize_image(cv2.imread(path))
            template = Image.fromarray(cv2.cvtColor(self.img_template, cv2.COLOR_BGR2RGB).astype('uint8'), 'RGB')
            self.img_template = cv2.cvtColor(self.img_template, cv2.COLOR_BGR2GRAY)
            self.template = ImageTk.PhotoImage(template)
            self.template_canvas.create_image(0,0, anchor="nw",image=self.template)
            self.matchTemplate()
            self.templateIN = True


    def createTkImage(self):
        path = self.img_paths[self.count_img]
        _, self.img_name = os.path.split(path)
        self.img = self.resize_image(cv2.imread(path))
        self.img_shape = self.img.shape
        img = Image.fromarray(cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB).astype('uint8'), 'RGB')
        self.tk_im = ImageTk.PhotoImage(img)
        self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)

    def changeImage(self):
        self.reset()
        self.count_img += 1
        if self.count_img < len(self.img_paths):
            self.createTkImage()
            self.canvas.itemconfigure(self.img_item, image=self.tk_im)
        else:
            self.abort()

    def abort(self):
        self.root.destroy()

    def reset(self):
        self.canvas.delete(self.rect)
        self.template_canvas.delete('all')
        self.templateIN = False


    def skip_image(self):
        self.changeImage()



