import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import torch
from torchvision import transforms
from age_model import AgeModel
from drowsiness_util import detect_faces_and_drowsiness

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AgeModel().to(device)
model.load_state_dict(torch.load("D:/Ai_Tool01/computer vision/Pranjul/age_model.pt", map_location=device))
model.eval()
#"D:\Ai_Tool01\computer vision\Pranjul\age_model.pt"
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((100, 100)),
    transforms.ToTensor()
])

class DrowsinessApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Drowsiness and Age Detection")

        self.canvas = tk.Canvas(root, width=800, height=600)
        self.canvas.pack()

        self.btn_frame = tk.Frame(root)
        self.btn_frame.pack(pady=10)

        self.img_btn = tk.Button(self.btn_frame, text="Load Image", command=self.load_image)
        self.img_btn.grid(row=0, column=0, padx=5)

        self.vid_btn = tk.Button(self.btn_frame, text="Load Video", command=self.load_video)
        self.vid_btn.grid(row=0, column=1, padx=5)

        self.frame = None

    def load_image(self):
        path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if not path:
            return

        image = cv2.imread(path)
        output_img, sleepy_count, ages = detect_faces_and_drowsiness(image, model, transform, device)

        # Show image
        img_rgb = cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        img_tk = ImageTk.PhotoImage(img_pil.resize((800, 600)))

        self.canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
        self.canvas.image = img_tk

        messagebox.showinfo("Result", f"Detected {sleepy_count} drowsy people. Ages: {ages}")

    def load_video(self):
        path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi")])
        if not path:
            return

        cap = cv2.VideoCapture(path)

        sleepy_count = 0
        total_drowsy_ages = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            output_img, frame_sleepy, ages = detect_faces_and_drowsiness(frame, model, transform, device)
            sleepy_count += frame_sleepy
            total_drowsy_ages += ages

            img_rgb = cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)
            img_tk = ImageTk.PhotoImage(img_pil.resize((800, 600)))

            self.canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
            self.canvas.image = img_tk
            self.root.update()

        cap.release()
        messagebox.showinfo("Result", f"Total Drowsy People: {sleepy_count}\nAges: {total_drowsy_ages}")

if __name__ == '__main__':
    root = tk.Tk()
    app = DrowsinessApp(root)
    root.mainloop()
