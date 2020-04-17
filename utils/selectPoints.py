import tkinter as tk
from tkinter import filedialog
import cv2
from PIL import Image, ImageTk
from utils.selectRect import computeHomograpy
import os
scale = 20
scale_h = 80
class PointSelector(object):
    def __init__(self, root, images):
        self.points = []
        self.count_points = 0
        self.root = root
        self.img_paths = images
        self.count_img = 0
        self.scale_percent = scale/100
        self.createTkImage()
        self.image_frame = tk.Frame(self.root, width=self.img_shape[1], height=self.img_shape[0], bd=1)
        self.image_frame.pack(side=tk.LEFT, expand=True)
        self.canvas = tk.Canvas(self.image_frame, width=self.img_shape[1], height=self.img_shape[0], cursor="circle")
        self.canvas.pack(side="top", fill="both", expand=True)
        self.canvas.bind("<ButtonPress-1>", self.on_button_press)
        self.img_item = self.canvas.create_image(0, 0, anchor="nw", image=self.tk_im)
        self.cropButton = tk.Button(self.image_frame, text="OK", command=self.confirm)
        self.root.bind('<Return>', lambda event=None: self.cropButton.invoke())
        self.cropButton.pack(side=tk.RIGHT, anchor=tk.SE)
        self.resetButton = tk.Button(self.image_frame, text="Reset", command=self.reset)
        self.resetButton.pack(side=tk.RIGHT, anchor=tk.SE)
        self.skipButton = tk.Button(self.image_frame, text="Skip", command=self.skip_image)
        self.skipButton.pack(side=tk.RIGHT, anchor=tk.SE)
        self.skipButton = tk.Button(self.image_frame, text="Exit", command=self.abort)
        self.skipButton.pack(side=tk.RIGHT, anchor=tk.SE)
        self.root.protocol("WM_DELETE_WINDOW", self.abort)
        self.template_frame = tk.Frame(self.root)
        self.template_frame.pack(side=tk.RIGHT, expand=True)
        self.template_canvas = tk.Canvas(self.template_frame)
        self.template_canvas.pack(side="top", fill="both", expand=True)
        self.template = Image.open(os.path.join(os.getcwd(), 'templates', 'template.png'))
        self.template = ImageTk.PhotoImage(self.template)
        self.template_canvas.create_image(0, 0, anchor="nw", image=self.template)
        self.instruction_label = tk.Label(self.template_frame, text="Select five points like in figure using muose")
        self.instruction_label.pack(side=tk.BOTTOM)
        self.root.update_idletasks()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry('{}x{}+{}+{}'.format(width, height, x, y))

    def createTkImage(self):
        path = self.img_paths[self.count_img]
        self.original_img = cv2.imread(path)
        _, self.img_name = os.path.split(path)
        width = int(self.original_img.shape[1] * self.scale_percent)
        height = int(self.original_img.shape[0] * self.scale_percent)
        dim = (width, height)
        img = cv2.resize(self.original_img, dim, interpolation=cv2.INTER_AREA)
        self.img_shape = img.shape
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype('uint8'), 'RGB')
        self.tk_im = ImageTk.PhotoImage(img)


    def changeImage(self):
        self.reset()
        self.count_img += 1
        if self.count_img<len(self.img_paths):
            self.createTkImage()
            self.canvas.itemconfigure(self.img_item, image=self.tk_im)
        else:
            self.root.destroy()

    def abort(self):
        self.root.destroy()

    def on_button_press(self, event):
        if self.count_points<5:
            x = event.x
            y = event.y
            self.points.append(((1/self.scale_percent) * x, (1/self.scale_percent) * y))
            self.count_points += 1
            r = 5
            self.canvas.create_oval(x-r, y-r, x+r, y+r, outline='green', fill='green', tags=("circle",))

    def confirm(self):
        if self.count_points == 5:
            homog = computeHomograpy(self.original_img, self.points)
            width = int(homog.shape[1] * scale_h/100)
            height = int(homog.shape[0] * scale_h/100)
            dim = (width, height)
            homog = cv2.resize(homog, dim, interpolation=cv2.INTER_AREA)
            show_homog = homog.copy()
            cv2.putText(show_homog, 'Confirm: Y/N', (10, 128), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
            while True:
                cv2.imshow('homography', show_homog)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("y"):
                    confirmed = 1
                    break
                elif key == ord("n"):
                    confirmed = 0
                    break
            cv2.destroyAllWindows()
            if confirmed:
                self.saveImage(homog)
                self.changeImage()
            else:
                self.reset()

    def saveImage(self, img):
        directory = filedialog.asksaveasfilename(initialdir=os.getcwd(), initialfile=self.img_name)
        cv2.imwrite(directory, img)


    def reset(self):
        self.canvas.delete("circle")
        self.count_points = 0
        self.points = []

    def skip_image(self):
        self.points = []
        self.changeImage()


