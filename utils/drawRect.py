import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
from utils import removeScore
from numpy import abs
scale = 80
import os

class CropImage():
    def __init__(self, root, image_paths):
        self.root = root
        self.img_paths = image_paths
        self.count_img = 0
        self.scale_percent = scale/100
        self.root.protocol("WM_DELETE_WINDOW", self.abort)
        self.x = self.y = 0
        self.createTkImage()
        self.image_frame = tk.Frame(self.root, width=self.img_shape[1], height=self.img_shape[0], bd=1)
        self.image_frame.pack(side=tk.LEFT)
        self.canvas = tk.Canvas(self.image_frame, width=self.img_shape[1], height=self.img_shape[0], cursor="cross")
        self.canvas.pack(side="top", fill="both", expand=True)
        self.img_item = self.canvas.create_image(0, 0, anchor="nw", image=self.tk_im)
        self.canvas.bind("<ButtonPress-1>", self.on_button_press)
        self.canvas.bind("<B1-Motion>", self.on_move_press)
        self.canvas.bind("<ButtonRelease-1>", self.on_button_release)
        self.canvas.bind('<Double-Button-1>', self.dragOrResize)
        self.rect = None
        self.start_x = None
        self.start_y = None
        self._drag_data = {"x": 0, "y": 0, "item": None}
        self.canvas.tag_bind("rec", "<ButtonPress-1>", self.start)
        self.canvas.tag_bind("rec", "<ButtonRelease-1>", self.stop)
        self.canvas.tag_bind("rec", "<B1-Motion>", self.move)
        self.dragging = False
        self.resizing = False
        self.modality = True
        self.drawn = False
        self.canvas.pack(anchor=tk.NW)
        self.cropButton = tk.Button(self.image_frame, text="Crop", command=self.get_rating)
        self.cropButton.pack(side=tk.RIGHT, anchor=tk.SE)
        self.root.bind('<Return>', lambda event=None: self.cropButton.invoke())
        self.resetButton = tk.Button(self.image_frame, text="Reset", command=self.reset)
        self.resetButton.pack(side=tk.RIGHT, anchor=tk.SE)
        self.skipButton = tk.Button(self.image_frame, text="Skip", command=self.skip_image)
        self.skipButton.pack(side=tk.RIGHT, anchor=tk.SE)
        self.abortButton = tk.Button(self.image_frame, text="Exit", command=self.abort)
        self.abortButton.pack(side=tk.RIGHT, anchor=tk.SE)
        self.template_frame = tk.Frame(self.root)
        self.template_frame.pack(side=tk.RIGHT)
        self.instruction_label = tk.Label(self.template_frame, text="Select Region Of\n Interest with mouse.\nDouble-"
                                                                    "click on rectangle\no modify ROI:\n Red border = Move ROI\n"
                                                                    "Blue border = Reshape ROI")
        self.instruction_label.pack(side=tk.BOTTOM)
        self.root.update_idletasks()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry('{}x{}+{}+{}'.format(width, height, x, y))

    def createTkImage(self):
        path = self.img_paths[self.count_img]
        self.img = cv2.imread(path)
        _, self.img_name = os.path.split(path)
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


    def on_button_press(self, event):
        # save mouse drag start position
        if not self.dragging and not self.resizing and not self.drawn:
            self.start_x = event.x
            self.start_y = event.y
            # create rectangle if not yet exist
            #if not self.rect:
            self.rect = self.canvas.create_rectangle(self.x, self.y, 1, 1, outline='red', width=4, tags=("rec",))

    def on_move_press(self, event):
        if not self.dragging and not self.resizing and not self.drawn:
            curX, curY = (event.x, event.y)
            # expand rectangle as you drag the mouse
            self.canvas.coords(self.rect, self.start_x, self.start_y, curX, curY)

    def on_button_release(self, event):
        self.drawn = True
        #self.n_feature += 1
        pass

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
        """End drag of an object"""
        # reset the drag information
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

    def dragOrResize(self, event):
        if self.modality:
            self.canvas.itemconfig(self.rect, outline="blue")
        else:
            self.canvas.itemconfig(self.rect, outline="red")
        self.modality = not self.modality


    def saveImage(self, img):
        index = self.img_name.find('.png')
        if index == -1:
            index = self.img_name.find('.jpg')
        head = self.img_name[:index]
        tail = self.img_name[index:]
        file_name_default = head+'_'+str(self.rating)+tail
        directory = filedialog.asksaveasfilename(initialdir=os.getcwd(), initialfile=file_name_default)
        if len(directory)>0:
            cv2.imwrite(directory, img)


    def crop(self):
        self.rating = self.panel.get()
        self.window.destroy()
        x0, y0, x1, y1 = self.canvas.coords(self.rect)
        self.canvas.delete(self.rect)
        self.drawn = False
        cropped = self.img[int(y0):int(y1), int(x0):int(x1)]
        no_score = removeScore.removeScore(cropped)
        self.saveImage(no_score)
        self.changeImage()


    def get_rating(self):
        self.window = tk.Toplevel()
        message = tk.Message(self.window, text="insert score")
        message.pack()
        self.panel = tk.Entry(self.window)
        self.panel.pack()
        button = tk.Button(self.window, text="OK", command=self.crop)
        self.window.bind('<Return>', lambda event=None: button.invoke())
        button.pack()

    def reset(self):
        self.canvas.delete(self.rect)
        self.drawn = False

    def skip_image(self):
        self.changeImage()