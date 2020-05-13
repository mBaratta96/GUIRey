import tkinter as tk
from tkinter import filedialog
import cv2
import os
from PIL import Image, ImageTk
from sklearn.mixture import GaussianMixture
import numpy as np
from utils.edgeDetection import maxDeviationThresh



def selectFiles():
    return filedialog.askopenfilenames(initialdir=os.getcwd(), title="Select images",
                                       filetypes=(("png files", ".png"), ("jpeg files", "*.jpg")))


class RemoveLine:
    def __init__(self, root, image_paths):
        self.root = root
        self.img_paths = image_paths
        self.count_img = 0
        self.createTkImage()
        self.image_frame = tk.Frame(self.root, width=self.img_shape[1], height=self.img_shape[0], bd=1)
        self.image_frame.pack(side=tk.LEFT, expand=True, fill=tk.BOTH)
        self.canvas = tk.Canvas(self.image_frame, width=self.img_shape[1], height=self.img_shape[0], cursor="cross")
        self.canvas.pack(side="top", fill="both", expand=True)
        self.img_item = self.canvas.create_image(0, 0, anchor="nw", image=self.tk_im)
        self.confirmButton = tk.Button(self.image_frame, text="OK", command=self.save)
        self.root.bind('<Return>', lambda event=None: self.confirmButton.invoke())
        self.confirmButton.pack(side=tk.RIGHT, anchor=tk.SE)
        self.template_frame = tk.Frame(self.root)
        self.template_frame.pack(side=tk.RIGHT)
        self.n_components = tk.Entry(self.template_frame, bd=3)
        self.n_components.insert(tk.END, '3')
        self.n_components.pack(side=tk.BOTTOM)
        self.component_label = tk.Label(self.template_frame, text='write number of components')
        self.component_label.pack(side=tk.BOTTOM)
        self.removed = None
        self.median_slide = tk.Scale(self.template_frame, from_=0, to=9, orient=tk.HORIZONTAL,
                                     command=lambda e: self.apply_filter())
        self.median_slide.set(2)
        self.median_slide.pack(side=tk.BOTTOM)
        self.median_label = tk.Label(self.template_frame, text='select filter kernel')
        self.median_label.pack(side=tk.BOTTOM)
        self.thresh_slide = tk.Scale(self.template_frame, from_=-20, to=20, orient=tk.HORIZONTAL,
                                     command=lambda e: self.confirm())
        self.thresh_slide.set(0)
        self.thresh_slide.pack(side=tk.BOTTOM)
        self.thresh_label = tk.Label(self.template_frame, text='select thresh')
        self.thresh_label.pack(side=tk.BOTTOM)
        self.root.update_idletasks()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry('{}x{}+{}+{}'.format(width, height, x, y))

    def abort(self):
        self.root.destroy()

    def createTkImage(self):
        path = self.img_paths[self.count_img]
        _, self.img_name = os.path.split(path)
        self.img = cv2.bilateralFilter(cv2.imread(path, cv2.IMREAD_GRAYSCALE), 10, 15, 15)
        self.img_shape = self.img.shape
        img = Image.fromarray(self.img.astype('uint8'), 'L')
        self.tk_im = ImageTk.PhotoImage(img)

    def changeImage(self):
        self.reset()
        self.count_img += 1
        if self.count_img < len(self.img_paths):
            self.createTkImage()
            self.canvas.itemconfigure(self.img_item, image=self.tk_im)
        else:
            self.abort()

    def apply_filter(self):
        if self.removed is not None:
            if self.median_slide.get() > 0:
                '''k_size = (int(2*self.median_slide.get()+1), int(2*self.median_slide.get()+1))
                anchor = (int(self.median_slide.get()), int(self.median_slide.get()))
                element = cv2.getStructuringElement(cv2.MORPH_RECT, k_size, anchor)

                self.filtered = cv2.dilate(self.removed, element)'''
                self.filtered = cv2.medianBlur(self.removed, int(2 * self.median_slide.get()) + 1)
            else:
                self.filtered = self.removed

            img = Image.fromarray(self.filtered.astype('uint8'), 'L')
            self.tk_im = ImageTk.PhotoImage(img)
            self.canvas.itemconfigure(self.img_item, image=self.tk_im)

    def confirm(self):
        dst = self.img.copy()
        hist, bins = np.histogram(dst[dst > 0].flatten(), range(257))
        thresh_val = maxDeviationThresh(hist)
        hist = hist[:thresh_val]
        #plt.stem(hist)
        #plt.show()
        n_components = int(self.n_components.get())
        gmm = GaussianMixture(n_components=n_components)
        mask = (dst > 0) & (dst < thresh_val)
        gmm = gmm.fit(np.expand_dims(dst[mask].ravel(), 1))
        bins = bins[bins < thresh_val]
        results = gmm.predict(np.expand_dims(bins[1:], 1))
        means = gmm.means_
        min_idx = np.argmin(means)
        
        '''for i in range(n_components):
            arg = np.where(results == i)[0]
            line_prop = 'C{}-'.format(i)
            marker_prop = 'C{}o'.format(i)
            plt.stem(arg, hist[arg], linefmt=line_prop, markerfmt=marker_prop, use_line_collection=True)
        plt.show()
        plt.close('all')'''
        max_occ = np.bincount(dst[dst > 0]).argmax()
        values = np.asarray(np.where(results == min_idx))[0]
        thresh = int(self.thresh_slide.get())
        if thresh > 0:
            add = np.arange(max(values)+1, max(values)+thresh+1, step=1)
            values = np.concatenate((values, add))
        elif thresh < 0:
            values = values[:thresh]
        mask = np.isin(dst, values)
        dst[mask] = max_occ
        self.removed = dst
        self.apply_filter()

    def save(self):
        index = self.img_name.find('.png')
        if index == -1:
            index = self.img_name.find('.jpg')
        head = self.img_name[:index]
        tail = self.img_name[index:]
        file_name_default = head + '_' + 'no_line' + tail
        directory = filedialog.asksaveasfilename(initialdir=os.getcwd(), initialfile=file_name_default)
        if len(directory) > 0:
           cv2.imwrite(directory, self.filtered)

if __name__ == '__main__':
    root = tk.Tk()
    image_paths = selectFiles()
    if len(image_paths) > 0:
        app = RemoveLine(root, image_paths)
        root.mainloop()
