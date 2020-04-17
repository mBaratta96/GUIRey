import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import math
from utils import removeScore
from utils.cosineSim import cosine, hausdorff, chamfer, histGrad
from numpy import abs
scale = 80

or_points = [
    [324,119,378,373],
    [375, 540,794, 661],
    [742, 287, 829, 373],
    [617, 383, 847, 534],
    [852, 229, 1056, 531]
]
label_dict={
    'cross.png':0,
    'crossdown.png':1,
    'face.png':2,
    'rail.png':3,
    'rombo.png':4
}
import os

def selectFiles():
    return filedialog.askopenfilename(initialdir = os.getcwd(),title = "Select a template",filetypes = ( ("png files",".png"),("jpeg files","*.jpg")))


class SimMeasures():
    def __init__(self, root, image_paths):
        self.root = root
        self.img_paths = image_paths
        self.count_img = 0
        self.scale_percent = scale/100
        for i in range(len(or_points)):
            or_points[i] = [int(p * self.scale_percent) for p in or_points[i]]
        self.root.protocol("WM_DELETE_WINDOW", self.abort)
        self.x = self.y = 0
        self.createTkImage()
        self.image_frame = tk.Frame(self.root, width=self.img_shape[1], height=self.img_shape[0], bd=1)
        self.image_frame.pack(side=tk.LEFT, expand = True, fill = tk.BOTH)
        self.canvas = tk.Canvas(self.image_frame, width=self.img_shape[1], height=self.img_shape[0], cursor="cross")
        self.canvas.pack(side="top", fill="both", expand=True)
        self.img_item = self.canvas.create_image(0, 0, anchor="nw", image=self.tk_im)
        self.canvas.bind('<Double-Button-1>', self.dragOrResize)
        self.start_x = None
        self.start_y = None
        self._drag_data = {"x": 0, "y": 0, "item": None}
        self.canvas.tag_bind("rec", "<ButtonPress-1>", self.start)
        self.canvas.tag_bind("rec", "<ButtonRelease-1>", self.stop)
        self.canvas.tag_bind("rec", "<B1-Motion>", self.move)
        self.dragging = False
        self.resizing = False
        self.modality = True
        self.canvas.pack(anchor=tk.NW)
        self.resetButton = tk.Button(self.image_frame, text="Reset", command=self.reset)
        self.resetButton.pack(side=tk.RIGHT, anchor=tk.SE)
        self.skipButton = tk.Button(self.image_frame, text="Skip", command=self.skip_image)
        self.skipButton.pack(side=tk.RIGHT, anchor=tk.SE)
        self.abortButton = tk.Button(self.image_frame, text="Exit", command=self.abort)
        self.abortButton.pack(side=tk.RIGHT, anchor=tk.SE)
        self.measures_frame = tk.Frame(self.root, bd=1)
        self.measures_frame.pack(side=tk.RIGHT)
        self.measures_canvas = tk.Canvas(self.measures_frame)
        self.measures_canvas.pack()
        self.templateIN = False
        self.insTemplateButton = tk.Button(self.measures_frame, text='Insert Template',
                                           command=self.insert_template).pack(side=tk.BOTTOM, anchor=tk.SE)
        self.cosine_selected = tk.IntVar()
        self.cosine_label = tk.Checkbutton(self.measures_frame, text='Cosine Distance', variable=self.cosine_selected, command=self.computeCosine)
        self.cosine_label.pack(side=tk.BOTTOM)
        self.cosine_value = tk.Text(self.measures_frame, height=1, width=20)
        self.cosine_value.pack(side=tk.BOTTOM, fill='x')
        self.haus_selected = tk.IntVar()
        self.haus_label = tk.Checkbutton(self.measures_frame, text='Hausdorff distance', variable=self.haus_selected, command=self.computeHaus)
        self.haus_label.pack(side=tk.BOTTOM)
        self.haus_value = tk.Text(self.measures_frame, height=1, width=20)
        self.haus_value.pack(side=tk.BOTTOM, fill='x')
        self.chamfer_selected = tk.IntVar()
        self.chamfer_label = tk.Checkbutton(self.measures_frame, text='Chamfer distance', variable=self.chamfer_selected, command=self.computeChamfer)
        self.chamfer_label.pack(side=tk.BOTTOM)
        self.chamfer_value = tk.Text(self.measures_frame, height=1, width=20)
        self.chamfer_value.pack(side=tk.BOTTOM, fill='x')
        self.HOG_selected = tk.IntVar()
        self.HOG_label = tk.Checkbutton(self.measures_frame, text='Histogram of Gradients',
                                            variable=self.HOG_selected, command=self.computeHOG)
        self.HOG_label.pack(side=tk.BOTTOM)
        self.HOG_value = tk.Text(self.measures_frame, height=1, width=20)
        self.HOG_value.pack(side=tk.BOTTOM, fill='x')
        self.distance_selected = tk.IntVar()
        self.distance_label = tk.Checkbutton(self.measures_frame, text='Distance from original',
                                            variable=self.distance_selected, command=self.computeDistanceOriginal)
        self.distance_label.pack(side=tk.BOTTOM)
        self.distance_value = tk.Text(self.measures_frame, height=1, width=20)
        self.distance_value.pack(side=tk.BOTTOM, fill='x')
        self.computeSimilarityGlobal()
        self.root.update_idletasks()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry('{}x{}+{}+{}'.format(width, height, x, y))



    def insert_template(self):
        path = selectFiles()
        if len(path)>0:
            self.reset()
            self.template_name = os.path.split(path)[1]
            img_template = cv2.imread(path)
            width = int(img_template.shape[1] * self.scale_percent)
            height = int(img_template.shape[0] * self.scale_percent)
            dim = (width, height)
            self.img_template = cv2.resize(img_template, dim, interpolation=cv2.INTER_AREA)
            template = Image.fromarray(cv2.cvtColor(self.img_template, cv2.COLOR_BGR2RGB).astype('uint8'), 'RGB')
            self.img_template = cv2.cvtColor(self.img_template, cv2.COLOR_BGR2GRAY)
            self.template = ImageTk.PhotoImage(template)
            self.measures_canvas.create_image(0,0, anchor="nw",image=self.template)
            w = int(self.img_template.shape[1])
            h = int(self.img_template.shape[0])
            self.rect = self.canvas.create_rectangle(50, 50, 50+w, 50+h, outline='red', width=4, tags=("rec",))
            self.no_score = self.getROI()
            self.templateIN = True
        else:
            messagebox.showerror("Error", "You must select a template")



    def createTkImage(self):
        path = self.img_paths[self.count_img]
        self.img = cv2.imread(path)
        for x in or_points:
            self.img = cv2.rectangle(self.img, (x[0], x[1]), (x[2], x[3]), color=(0,0,0))
        self.img_shape = self.img.shape
        img = Image.fromarray(cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB).astype('uint8'), 'RGB')
        self.tk_im = ImageTk.PhotoImage(img)

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

    def start(self, event):
        """Begining drag of an object"""
        # record the item and its location
        self._drag_data["item"] = self.canvas.find_closest(event.x, event.y)[0]
        self._drag_data["x"] = event.x
        self._drag_data["y"] = event.y
        if self.modality:
            self.dragging = True
        else:
            self.resizing = True


    def stop(self, event):
        self._drag_data["item"] = None
        self._drag_data["x"] = 0
        self._drag_data["y"] = 0
        if self.modality:
            self.dragging = False
        else:
            self.resizing = False


    def move(self, event):

        delta_x = event.x - self._drag_data["x"]
        delta_y = event.y - self._drag_data["y"]

        if self.modality:
            self.canvas.move(self._drag_data["item"], delta_x, delta_y)
        else:
            if len(self.canvas.coords(self._drag_data["item"]))==4:
                x0, y0, x1, y1 = self.canvas.coords(self._drag_data["item"])
                if(abs(self._drag_data["x"]-x0)<20):
                    self.canvas.coords(self._drag_data["item"], x0+delta_x, y0, x1, y1)
                elif(abs(self._drag_data["x"]-x1)<20):
                    self.canvas.coords(self._drag_data["item"], x0, y0, x1+delta_x, y1)
                elif(abs(self._drag_data["y"]-y0)<20):
                    self.canvas.coords(self._drag_data["item"], x0, y0+delta_y, x1, y1)
                elif(abs(self._drag_data["y"]-y1)<20):
                    self.canvas.coords(self._drag_data["item"], x0, y0, x1, y1+delta_y)

        self._drag_data["x"] = event.x
        self._drag_data["y"] = event.y
        self.no_score = self.getROI()
        self.computeSimilarityGlobal()

    def dragOrResize(self, event):
        if self.modality:
            self.canvas.itemconfig(self.rect, outline="blue")
        else:
            self.canvas.itemconfig(self.rect, outline="red")
        self.modality = not self.modality

    def getROI(self):
        x0, y0, x1, y1 = self.canvas.coords(self.rect)
        cropped = self.img[int(y0):int(y1), int(x0):int(x1)]
        no_score = cv2.cvtColor(removeScore.removeScore(cropped), cv2.COLOR_BGR2GRAY)
        return no_score

    def computeCosine(self):
        self.cosine_value.delete(1.0, tk.END)
        if self.cosine_selected.get() and self.templateIN:
            self.cosine_value.insert(tk.END, str(cosine(self.img_template, self.no_score)))
        else:
            self.cosine_value.insert(tk.END, '0.0')

    def computeHaus(self):
        self.haus_value.delete(1.0, tk.END)
        if self.haus_selected.get() and self.templateIN:
            self.haus_value.insert(tk.END, str(hausdorff(self.img_template, self.no_score)))
        else:
            self.haus_value.insert(tk.END, '0.0')

    def computeChamfer(self):
        self.chamfer_value.delete(1.0, tk.END)
        if self.chamfer_selected.get() and self.templateIN:
            self.chamfer_value.insert(tk.END, str(chamfer(self.img_template, self.no_score)))
        else:
            self.chamfer_value.insert(tk.END, '0.0')

    def computeHOG(self):
        self.HOG_value.delete(1.0, tk.END)
        if self.HOG_selected.get() and self.templateIN:
            self.HOG_value.insert(tk.END, str(histGrad(self.img_template, self.no_score)))
        else:
            self.HOG_value.insert(tk.END, '0.0')

    def computeDistanceOriginal(self):
        self.distance_value.delete(1.0, tk.END)
        if self.distance_selected.get() and self.templateIN:
            original_coord = or_points[label_dict[self.template_name]]
            rect_coord = self.canvas.coords(self.rect)
            centerROI = (int((rect_coord[2]+rect_coord[0])//2), int((rect_coord[3]+rect_coord[1])//2))
            center_or = ((original_coord[2]+original_coord[0])//2, (original_coord[3]+original_coord[1])//2)
            dist = math.hypot(center_or[0]-centerROI[0], center_or[1]-centerROI[1])
            self.distance_value.insert(tk.END, str(dist))
            self.templateDrawn = True
        else:
            self.distance_value.insert(tk.END, '0.0')


    def computeSimilarityGlobal(self):
        self.computeCosine()
        self.computeHaus()
        self.computeChamfer()
        self.computeHOG()
        self.computeDistanceOriginal()

    def reset(self):
        if self.templateIN:
            self.canvas.delete(self.rect)
        self.measures_canvas.delete('all')
        self.templateIN = False
        self.computeSimilarityGlobal()

    def skip_image(self):
        self.changeImage()



