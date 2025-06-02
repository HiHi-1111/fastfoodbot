import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import os
from io import BytesIO
import win32clipboard
from order_processor import order_processor, identify_ingredient
from PIL import ImageGrab

templates = ["x1.png", "x2.png", "patty.png"]

class ImageMatcher:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Image Matcher")
        self.current_image = None
        self.items = ["cheese", "lettuce", "tomato", "onion", "patty"]
        
        # Create main frame
        self.main_frame = ttk.Frame(self.root, padding="10")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Create canvas for image display
        self.canvas = tk.Canvas(self.main_frame, width=400, height=400, bg='white')
        self.canvas.grid(row=0, column=0, pady=5)
        self.canvas.bind('<Button-1>', self.paste_image)
        
        # Create style for green outline button
        self.style = ttk.Style()
        self.style.configure('GreenOutline.TButton', foreground='black', borderwidth=3, relief='solid')
        self.style.map('GreenOutline.TButton',
                       bordercolor=[('!disabled', 'green'), ('disabled', 'grey')],
                       highlightcolor=[('!disabled', 'green'), ('disabled', 'grey')])

        # Create identify button
        self.identify_button = ttk.Button(self.main_frame, text="Identify", command=self.identify_image, state='disabled')
        self.identify_button.grid(row=1, column=0, pady=5)
        
        # Create result label
        self.result_label = ttk.Label(self.main_frame, text="")
        self.result_label.grid(row=2, column=0, pady=5)

    def paste_image(self, event=None):
        print("image box clicked!")  # Print message to terminal
        try:

            image = ImageGrab.grabclipboard()
        
            if image is None:
                self.result_label['text'] = "No image in clipboard"
                return

            self.current_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # Resize image to fit canvas while maintaining aspect ratio
            display_size = (400, 400)
            image.thumbnail(display_size, Image.LANCZOS)
            
            # Convert to PhotoImage for display
            self.photo = ImageTk.PhotoImage(image)
            self.canvas.create_image(200, 200, image=self.photo, anchor='center')
            
            # Enable identify button
            self.identify_button['state'] = 'normal'

        except:
            self.result_label['text'] = "Error: Could not paste image"

    def identify_image(self):
        if self.current_image is None:
            self.result_label["text"] = "Nothing to see yet."
            return

        # Process the current image.
        # res = order_processor(self.current_image)
        item = identify_ingredient(self.current_image)

        if item > -1:
            display = f"Result:\n{self.items[item]}"
        self.result_label['text'] = display


    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    app = ImageMatcher()
    app.run()