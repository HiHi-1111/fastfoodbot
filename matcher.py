import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import os
from io import BytesIO
import win32clipboard
from order_processor import order_processor, identify_ingredient, split_order_items
from text_find import find_text
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

        # Create frame for sub-images
        self.sections_frame = ttk.Frame(self.main_frame)
        self.sections_frame.grid(row=3, column=0, pady=10)
        
        # Create label for sections heading
        self.sections_label = ttk.Label(self.sections_frame, text="Relevant sections:")
        self.sections_label.grid(row=0, column=0, columnspan=10, pady=(0,5), sticky='w')

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

        # Calculate proportional coordinates for the slice
        height, width = self.current_image.shape[:2]
    
        # Original coordinates were for 2560x1369 image
        # Convert to proportions
        x1_prop = 421/2560  # Left x coordinate
        x2_prop = 2116/2560  # Right x coordinate
        y1_prop = 232/1369  # Top y coordinate
        y2_prop = 545/1369  # Bottom y coordinate
        
        # Calculate actual coordinates for current image
        x1 = int(width * x1_prop)
        x2 = int(width * x2_prop)
        y1 = int(height * y1_prop)
        y2 = int(height * y2_prop)
        
        # Extract the relevant portion
        relevant_portion = self.current_image[y1:y2, x1:x2]

        # Clear previous sub-images
        for widget in self.sections_frame.winfo_children():
            if widget != self.sections_label:
                widget.destroy()

        # Identify any text in this portion.
        with_text = find_text(relevant_portion, "With")

        display = "Result:\n"

        if with_text:
            display += "With  !!"

        else:
            all_items = split_order_items(relevant_portion)
            
            # Display sub-images
            for i, item_image in enumerate(all_items):
                # Convert CV2 image to PIL Image
                item_rgb = cv2.cvtColor(item_image, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(item_rgb)
                
                # Resize to thumbnail
                display_size = (100, 100)
                pil_image.thumbnail(display_size, Image.LANCZOS)
                
                # Create and store PhotoImage (need to store to prevent garbage collection)
                photo = ImageTk.PhotoImage(pil_image)
                setattr(self, f'photo_{i}', photo)
                
                # Create canvas and display image
                canvas = tk.Canvas(self.sections_frame, width=100, height=100, bg='white')
                canvas.grid(row=1, column=i, padx=5)
                canvas.create_image(50, 50, image=photo, anchor='center')

            # Continue with identification
            for item in all_items:
                item = identify_ingredient(item)  # Note: now passing individual item image
                if item > -1:
                    display += "\n" + self.items[item]

        self.result_label['text'] = display


    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    app = ImageMatcher()
    app.run()