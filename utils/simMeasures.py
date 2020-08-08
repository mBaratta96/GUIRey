import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import math
from utils import removeScore
from utils.cosineSim import cosine, hausdorff, chamfer, histGrad
import numpy as np
import os
from utils.edgeDetection import extractDrawing
from matplotlib import pyplot as plt
#from keras.models import load_model

scale = 100
shape = (100, 100, 1)
or_points = [
    [324, 119, 378, 373],
    [375, 540, 794, 661],
    [742, 287, 829, 373],
    [617, 383, 847, 534],
    [852, 229, 1056, 531]
]
label_dict = {
    'cross.png': 0,
    'crossdown.png': 1,
    'face.png': 2,
    'rail.png': 3,
    'rombo.png': 4
}


def selectFiles():
    return filedialog.askopenfilename(initialdir=os.getcwd(), title="Select a template",
                                      filetypes=(("png files", ".png"), ("jpeg files", "*.jpg")))


def background_thumbnail(template, modality, thumbnail_size=(200, 200)):
    foreground = Image.fromarray(template).convert(modality)
    background = Image.new(modality, thumbnail_size, "white")
    foreground.thumbnail(thumbnail_size)
    (w, h) = foreground.size
    upper_left = (int((thumbnail_size[0] - w) / 2), int((thumbnail_size[1] - h) / 2))
    background.paste(foreground, upper_left)
    return np.array(background)


class SimMeasures():
    def __init__(self, root, image_paths):
        self.root = root
        self.img_paths = image_paths
        self.count_img = 0
        self.root.protocol("WM_DELETE_WINDOW", self.abort)
        self.x = self.y = 0
        self.img_or_points = [None] * len(or_points)
        self.createTkImage()
        self.image_frame = tk.Frame(self.root, width=self.img_shape[1], height=self.img_shape[0], bd=1)
        self.image_frame.pack(side=tk.LEFT, expand=True, fill=tk.BOTH)
        self.canvas = tk.Canvas(self.image_frame, width=self.img_shape[1], height=self.img_shape[0], cursor="cross")
        self.canvas.pack(side="top", fill="both", expand=True)
        self.img_item = self.canvas.create_image(0, 0, anchor="nw", image=self.tk_im)
        self.canvas.bind('<Double-Button-1>', self.dragOrResize)
        self.root.bind('<Left>', lambda e: self.left())
        self.root.bind('<Right>', lambda e: self.right())
        self.root.bind('<Up>', lambda e: self.up())
        self.root.bind('<Down>', lambda e: self.down())
        self.start_x = None
        self.start_y = None
        self.model = None
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
        self.cosine_label = tk.Checkbutton(self.measures_frame, text='Cosine Distance', variable=self.cosine_selected,
                                           command=self.computeCosine)
        self.cosine_label.pack(side=tk.BOTTOM)
        self.cosine_value = tk.Text(self.measures_frame, height=1, width=20)
        self.cosine_value.pack(side=tk.BOTTOM, fill='x')
        self.haus_selected = tk.IntVar()
        self.haus_label = tk.Checkbutton(self.measures_frame, text='Hausdorff distance', variable=self.haus_selected,
                                         command=self.computeHaus)
        self.haus_label.pack(side=tk.BOTTOM)
        self.haus_value = tk.Text(self.measures_frame, height=1, width=20)
        self.haus_value.pack(side=tk.BOTTOM, fill='x')
        self.chamfer_selected = tk.IntVar()
        self.chamfer_label = tk.Checkbutton(self.measures_frame, text='Chamfer distance',
                                            variable=self.chamfer_selected, command=self.computeChamfer)
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
        '''self.model_selected = tk.IntVar()
        self.model_label = tk.Checkbutton(self.measures_frame, text='Score from model',
                                          variable=self.model_selected, command=self.computeModelScore)
        self.model_label.pack(side=tk.BOTTOM)
        self.model_value = tk.Text(self.measures_frame, height=1, width=20)
        self.model_value.pack(side=tk.BOTTOM, fill='x')'''
        self.computeSimilarityGlobal()
        self.root.update_idletasks()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry('{}x{}+{}+{}'.format(width, height, x, y))

    def insert_template(self):
        path = selectFiles()
        if len(path) > 0:
            self.reset()
            self.template_name = os.path.split(path)[1]
            img_template = cv2.imread(path)
            width = int(img_template.shape[1] * self.scale_percent_w)
            height = int(img_template.shape[0] * self.scale_percent_h)
            dim = (width, height)
            self.img_template = cv2.resize(img_template, dim, interpolation=cv2.INTER_AREA)
            template = Image.fromarray(cv2.cvtColor(self.img_template, cv2.COLOR_BGR2RGB).astype('uint8'), 'RGB')
            self.img_template = cv2.cvtColor(self.img_template, cv2.COLOR_BGR2GRAY)
            self.input_template = background_thumbnail(self.img_template, 'L', (shape[0], shape[1]))
            self.input_template = self.input_template.astype('float32')
            self.input_template /= 255
            self.input_template = np.reshape(self.input_template, shape)
            self.template = ImageTk.PhotoImage(template)
            self.measures_canvas.create_image(0, 0, anchor="nw", image=self.template)
            w = int(self.img_template.shape[1])
            h = int(self.img_template.shape[0])
            self.rect = self.canvas.create_rectangle(50, 50, 50 + w, 50 + h, outline='red', width=4, tags=("rec",))
            self.no_score = self.getROI()
            self.templateIN = True
        else:
            messagebox.showerror("Error", "You must select a template")

    def createTkImage(self):
        path = self.img_paths[self.count_img]
        self.img = cv2.imread(path)
        self.img_shape = self.img.shape
        self.scale_percent_w = (self.img_shape[1] / 1360)
        self.scale_percent_h = (self.img_shape[0] / 768)
        for i in range(len(or_points)):
            self.img_or_points[i] = or_points[i].copy()
            self.img_or_points[i][0] *= self.scale_percent_w
            self.img_or_points[i][2] *= self.scale_percent_w
            self.img_or_points[i][1] *= self.scale_percent_h
            self.img_or_points[i][3] *= self.scale_percent_h
            self.img_or_points[i] = np.around(self.img_or_points[i]).astype(int)
        img = self.img.copy()
        for x in self.img_or_points:
            img = cv2.rectangle(img, (x[0], x[1]), (x[2], x[3]), color=(0, 0, 0))
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype('uint8'), 'RGB')
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
        if self.templateIN:
            self._drag_data["x"] = event.x
            self._drag_data["y"] = event.y
            if self.modality:
                self.dragging = True
            else:
                self.resizing = True

    def stop(self, event):
        if self.templateIN:
            self._drag_data["x"] = 0
            self._drag_data["y"] = 0
            if self.modality:
                self.dragging = False
            else:
                self.resizing = False

    def move(self, event):
        if self.templateIN:
            delta_x = event.x - self._drag_data["x"]
            delta_y = event.y - self._drag_data["y"]

            if self.modality:
                self.canvas.move(self.rect, delta_x, delta_y)
            else:
                if len(self.canvas.coords(self.rect)) == 4:
                    x0, y0, x1, y1 = self.canvas.coords(self.rect)
                    if abs(self._drag_data["x"] - x0) < 20:
                        self.canvas.coords(self.rect, x0 + delta_x, y0, x1, y1)
                    elif abs(self._drag_data["x"] - x1) < 20:
                        self.canvas.coords(self.rect, x0, y0, x1 + delta_x, y1)
                    elif abs(self._drag_data["y"] - y0) < 20:
                        self.canvas.coords(self.rect, x0, y0 + delta_y, x1, y1)
                    elif abs(self._drag_data["y"] - y1) < 20:
                        self.canvas.coords(self.rect, x0, y0, x1, y1 + delta_y)

            self._drag_data["x"] = event.x
            self._drag_data["y"] = event.y
            self.no_score = self.getROI()
            self.computeSimilarityGlobal()

    def dragOrResize(self, event):
        if self.templateIN:
            if self.modality:
                self.canvas.itemconfig(self.rect, outline="blue")
            else:
                self.canvas.itemconfig(self.rect, outline="red")
            self.modality = not self.modality

    def left(self):
        if self.templateIN:
            delta_x = -1
            delta_y = 0
            if self.modality:
                self.canvas.move(self.rect, delta_x, delta_y)
            else:
                x0, y0, x1, y1 = self.canvas.coords(self.rect)
                self.canvas.coords(self.rect, x0 + delta_x, y0, x1, y1)

    def right(self):
        if self.templateIN:
            delta_x = 1
            delta_y = 0
            if self.modality:
                self.canvas.move(self.rect, delta_x, delta_y)
            else:
                x0, y0, x1, y1 = self.canvas.coords(self.rect)
                self.canvas.coords(self.rect, x0, y0, x1 + delta_x, y1)

    def up(self):
        if self.templateIN:
            delta_x = 0
            delta_y = -1
            if self.modality:
                self.canvas.move(self.rect, delta_x, delta_y)
            else:
                x0, y0, x1, y1 = self.canvas.coords(self.rect)
                self.canvas.coords(self.rect, x0, y0 + delta_y, x1, y1)

    def down(self):
        if self.templateIN:
            delta_x = 0
            delta_y = 1
            if self.modality:
                self.canvas.move(self.rect, delta_x, delta_y)
            else:
                x0, y0, x1, y1 = self.canvas.coords(self.rect)
                self.canvas.coords(self.rect, x0, y0, x1, y1 + delta_y)

    def getROI(self):
        x0, y0, x1, y1 = self.canvas.coords(self.rect)
        x0 = int(min(self.img_shape[1], max(0, x0)))
        y0 = int(min(self.img_shape[0], max(0, y0)))
        y1 = int(min(self.img_shape[0], max(0, y1)))
        x1 = int(min(self.img_shape[1], max(0, x1)))
        cropped = self.img[y0:y1, x0:x1]
        if cropped.shape[0] > 0 and cropped.shape[1] > 0:
            no_score = cv2.cvtColor(removeScore.removeScore(cropped), cv2.COLOR_BGR2GRAY)
        else:
            no_score = None
        return no_score

    def computeCosine(self):
        self.cosine_value.delete(1.0, tk.END)
        if self.cosine_selected.get() and self.templateIN and self.no_score is not None:
            self.cosine_value.insert(tk.END, str(cosine(self.img_template, self.no_score)))
        else:
            self.cosine_value.insert(tk.END, '0.0')

    def computeHaus(self):
        self.haus_value.delete(1.0, tk.END)
        if self.haus_selected.get() and self.templateIN and self.no_score is not None:
            self.haus_value.insert(tk.END, str(hausdorff(self.img_template, self.no_score)))
        else:
            self.haus_value.insert(tk.END, '0.0')

    def computeChamfer(self):
        self.chamfer_value.delete(1.0, tk.END)
        if self.chamfer_selected.get() and self.templateIN and self.no_score is not None:
            self.chamfer_value.insert(tk.END, str(chamfer(self.img_template, self.no_score)))
        else:
            self.chamfer_value.insert(tk.END, '0.0')

    def computeHOG(self):
        self.HOG_value.delete(1.0, tk.END)
        if self.HOG_selected.get() and self.templateIN and self.no_score is not None:
            self.HOG_value.insert(tk.END, str(histGrad(self.img_template, self.no_score)))
        else:
            self.HOG_value.insert(tk.END, '0.0')

    def computeDistanceOriginal(self):
        self.distance_value.delete(1.0, tk.END)
        if self.distance_selected.get() and self.templateIN and self.no_score is not None:
            original_coord = self.img_or_points[label_dict[self.template_name]]
            rect_coord = self.canvas.coords(self.rect)
            centerROI = (int((rect_coord[2] + rect_coord[0]) // 2), int((rect_coord[3] + rect_coord[1]) // 2))
            center_or = ((original_coord[2] + original_coord[0]) // 2, (original_coord[3] + original_coord[1]) // 2)
            dist = math.hypot(center_or[0] - centerROI[0], center_or[1] - centerROI[1])
            self.distance_value.insert(tk.END, str(dist))
        else:
            self.distance_value.insert(tk.END, '0.0')

    def computeModelScore(self):
        self.model_value.delete(1.0, tk.END)
        if self.model_selected.get() and self.templateIN and self.no_score is not None:
            if self.model is None:
                self.model = load_model
            threshed = np.array(extractDrawing(self.no_score))
            img_input = background_thumbnail(threshed, 'L', (shape[0], shape[1]))
            img_input = img_input.astype('float32')
            img_input /= 255
            img_input = np.reshape(img_input, shape)
            fig, ax = plt.subplots(nrows=1, ncols=2)
            ax.ravel()[0].imshow(img_input[:, :, 0], cmap='gray')
            ax.ravel()[1].imshow(self.input_template[:, :, 0], cmap='gray')
            plt.show()
            plt.close('all')
            self.model_value.insert(tk.END, str(5))
        else:
            self.model_value.insert(tk.END, '0.0')

    def computeSimilarityGlobal(self):
        self.computeCosine()
        self.computeHaus()
        self.computeChamfer()
        self.computeHOG()
        self.computeDistanceOriginal()
        self.computeModelScore()

    def reset(self):
        if self.templateIN:
            self.canvas.delete(self.rect)
        self.measures_canvas.delete('all')
        self.templateIN = False
        self.computeSimilarityGlobal()

    def skip_image(self):
        self.changeImage()
