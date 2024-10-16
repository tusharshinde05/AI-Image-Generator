import tkinter as tk
import customtkinter as ctk
from PIL import Image, ImageTk
from authtoken import auth_token

import torch
from torch import autocast
from diffusers import StableDiffusionPipeline

# Create the app window
app = ctk.CTk()  # Use CTk for main window if using CustomTkinter
app.geometry("532x632")
app.title("Stable Bud")
ctk.set_appearance_mode("dark")

# Prompt entry field
prompt = ctk.CTkEntry(master=app, height=40, width=512, font=("Arial", 20), text_color="black", fg_color="white")
prompt.place(x=10, y=10)

# Image display label
lmain = ctk.CTkLabel(master=app, height=512, width=512)
lmain.place(x=10, y=110)

# Model setup
modelid = "CompVis/stable-diffusion-v1-4"
device = "cuda" if torch.cuda.is_available() else "cpu"  # Check if CUDA is available
dtype = torch.float16 if device == "cuda" else torch.float32
pipe = StableDiffusionPipeline.from_pretrained(modelid, torch_dtype = dtype, use_auth_token=auth_token)
pipe.to(device)

def generate():
    # Generate image from prompt
    with autocast(device):
        image = pipe(prompt.get(), guidance_scale=8.5).images[0]  # Use .images[0] to get the PIL image directly
    
    image.save('generatedimage.png')  # Save the image
    
    # Display the generated image in the Tkinter window
    img = ImageTk.PhotoImage(image)
    lmain.configure(image=img)
    lmain.image = img  # Keep a reference to avoid garbage collection

# Generate button
trigger = ctk.CTkButton(master=app, height=40, width=120, font=("Arial", 20), text_color="white", fg_color="blue", command=generate)
trigger.configure(text="Generate")
trigger.place(x=206, y=60)

# Start the main application loop
app.mainloop()
