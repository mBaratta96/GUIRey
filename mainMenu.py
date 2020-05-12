import tkinter as tk
from tkinter import filedialog, messagebox
import os
from utils.selectPoints import PointSelector
from utils.drawRect import CropImage
from utils.simMeasures import SimMeasures
from utils.findTemplate import FindTemp
from utils.lineRemoval import RemoveLine


def selectFiles():
    return filedialog.askopenfilenames(initialdir=os.getcwd(), title="Select images",
                                       filetypes=(("png files", ".png"), ("jpeg files", "*.jpg")))


class MainMenu(object):

    def __init__(self, root):
        self.root = root
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.title("Rey-Osferrith Image Operations")
        self.root.geometry("500x400")
        self.menu_frame = tk.Frame(self.root).pack()
        self.omogButton = tk.Button(self.menu_frame, text="Compute Homography", command=self.openHomography).pack(
            fill='both')
        self.cropTemplateButton = tk.Button(self.menu_frame, text="Crop Image", command=self.openCropRegion).pack(
            fill='both')
        self.analizeScoreButton = tk.Button(self.menu_frame, text="Compute Similarity Measures",
                                            command=self.openSimMeasures).pack(
            fill='both')
        self.findTempButton = tk.Button(self.menu_frame, text="Find Cropped Image",
                                        command=self.openTempMatching).pack(fill='both')
        self.removeLineButton = tk.Button(self.menu_frame, text="Remove Line (Work In Progress)",
                                          command=self.removeLine).pack(fill='both')

    def on_closing(self):
        if messagebox.askokcancel("Quit", "Do you want to quit?"):
            self.root.destroy()

    def openHomography(self):
        image_paths = selectFiles()
        if len(image_paths) > 0:
            self.tl = tk.Toplevel(self.root)
            self.win = PointSelector(self.tl, image_paths)
            self.tl.wait_window()

    def openCropRegion(self):
        image_paths = selectFiles()
        if len(image_paths) > 0:
            self.tl = tk.Toplevel(self.root)
            self.win = CropImage(self.tl, image_paths)
            self.tl.wait_window()

    def openSimMeasures(self):
        image_paths = selectFiles()
        if len(image_paths) > 0:
            self.tl = tk.Toplevel(self.root)
            self.win = SimMeasures(self.tl, image_paths)
            self.tl.wait_window()

    def openTempMatching(self):
        image_paths = selectFiles()
        if len(image_paths) > 0:
            self.tl = tk.Toplevel(self.root)
            self.win = FindTemp(self.tl, image_paths)
            self.tl.wait_window()

    def removeLine(self):
        image_paths = selectFiles()
        if len(image_paths) > 0:
            self.tl = tk.Toplevel(self.root)
            self.win = RemoveLine(self.tl, image_paths)
            self.tl.wait_window()


if __name__ == '__main__':
    root = tk.Tk()
    app = MainMenu(root)
    root.mainloop()
