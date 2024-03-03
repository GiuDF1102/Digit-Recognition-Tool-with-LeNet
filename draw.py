import tkinter as tk
from tkinter import messagebox

import cv2
from model.LeNet5 import predict, LeNet5
import os 
from PIL import Image, ImageOps
import numpy as np
import torch

class DigitRecognitionApp:
    def __init__(self, master, model):
        self.master = master
        self.model = model
        self.master.title("Digit Recognition App")

        self.canvas = tk.Canvas(self.master, width=200, height=200, bg="white", bd=2, relief=tk.RAISED)
        self.canvas.pack(pady=10)

        self.canvas.bind("<B1-Motion>", self.draw)
        self.canvas.bind("<ButtonRelease-1>", self.recognize_digit)

        self.prediction_label = tk.Label(self.master, text="", font=("Arial", 18))
        self.prediction_label.pack(pady=10)

    def draw(self, event):
        x1, y1 = (event.x - 10), (event.y - 10)
        x2, y2 = (event.x + 10), (event.y + 10)
        self.canvas.create_oval(x1, y1, x2, y2, fill="black")

    def clear_canvas(self):
        self.canvas.delete("all")
        self.prediction_label.config(text="")

    def recognize_digit(self, event=None):
        # Get digit.EPS
        self.canvas.postscript(file="digit.eps")  # Save canvas as EPS
        digit = Image.open("digit.eps")
        digit = digit.convert("L")  # Convert to grayscale
        digit = ImageOps.invert(digit)  # Invert colors
        digit = digit.resize((28, 28))  # Resize
        digit.save("digit.jpg")  # Save as PNG
        digit = cv2.imread("digit.jpg", cv2.IMREAD_GRAYSCALE)
        digit = digit.reshape(1, 1, 28, 28)
        digit = digit.astype(np.float32)
        digit = torch.from_numpy(digit)

        prediction = predict(self.model, digit, 'cpu')

        self.prediction_label.config(text=f"Prediction: {prediction[0]}")

        self.master.after(2000, self.clear_canvas)
    
    def clear_prediction_label(self):
        self.prediction_label.config(text="")



def main(model):
    root = tk.Tk()
    app = DigitRecognitionApp(root, model)
    root.mainloop()

if __name__ == "__main__":
    #Load LeNet model
    model = LeNet5()

    cwd = os.getcwd()
    model_path = os.path.join(cwd, 'lenet5.pth')
    model = model.load_model(model_path)

    main(model)
