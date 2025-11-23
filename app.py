import torch
import torch.nn.functional as F
from torchvision import transforms
import cv2
import numpy as np
from PIL import Image
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import ImageTk
import threading
from model import ASLNet

class ASLApp:
    def __init__(self, root):
        self.root = root
        self.root.title("ASL Sign Language to Text Converter")
        self.root.geometry("1200x700")
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.classes = []
        self.is_camera_running = False
        self.cap = None
        self.current_text = ""
        
        self.transform = transforms.Compose([
            transforms.Resize((200, 200)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.setup_ui()
        self.load_model()
    
    def setup_ui(self):
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        title_label = ttk.Label(main_frame, text="ASL Sign Language to Text", font=('Arial', 20, 'bold'))
        title_label.grid(row=0, column=0, columnspan=3, pady=10)
        
        device_label = ttk.Label(main_frame, text=f"Device: {self.device}", font=('Arial', 10))
        device_label.grid(row=1, column=0, columnspan=3)
        
        self.video_label = ttk.Label(main_frame, text="Video Feed", relief=tk.SUNKEN)
        self.video_label.grid(row=2, column=0, columnspan=2, padx=5, pady=5, sticky='nsew')
        
        control_frame = ttk.Frame(main_frame)
        control_frame.grid(row=3, column=0, columnspan=2, pady=10)
        
        self.start_camera_btn = ttk.Button(control_frame, text="Start Camera", command=self.start_camera)
        self.start_camera_btn.grid(row=0, column=0, padx=5)
        
        self.stop_camera_btn = ttk.Button(control_frame, text="Stop Camera", command=self.stop_camera, state='disabled')
        self.stop_camera_btn.grid(row=0, column=1, padx=5)
        
        self.load_image_btn = ttk.Button(control_frame, text="Load Image", command=self.load_image)
        self.load_image_btn.grid(row=0, column=2, padx=5)
        
        self.load_video_btn = ttk.Button(control_frame, text="Load Video", command=self.load_video)
        self.load_video_btn.grid(row=0, column=3, padx=5)
        
        self.clear_btn = ttk.Button(control_frame, text="Clear Text", command=self.clear_text)
        self.clear_btn.grid(row=0, column=4, padx=5)
        
        text_frame = ttk.LabelFrame(main_frame, text="Detected Text", padding="10")
        text_frame.grid(row=2, column=2, rowspan=2, padx=5, pady=5, sticky='nsew')
        
        self.text_display = tk.Text(text_frame, width=40, height=20, font=('Arial', 12), wrap=tk.WORD)
        self.text_display.pack(fill=tk.BOTH, expand=True)
        
        scrollbar = ttk.Scrollbar(text_frame, command=self.text_display.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.text_display.config(yscrollcommand=scrollbar.set)
        
        self.prediction_label = ttk.Label(main_frame, text="Current Prediction: -", font=('Arial', 14, 'bold'))
        self.prediction_label.grid(row=4, column=0, columnspan=3, pady=10)
        
        self.confidence_label = ttk.Label(main_frame, text="Confidence: -", font=('Arial', 11))
        self.confidence_label.grid(row=5, column=0, columnspan=3)
        
        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.columnconfigure(2, weight=1)
        main_frame.rowconfigure(2, weight=1)
    
    def load_model(self):
        try:
            checkpoint = torch.load('asl_model.pth', map_location=self.device)
            self.classes = checkpoint['classes']
            self.model = ASLNet(num_classes=len(self.classes)).to(self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            messagebox.showinfo("Success", "Model loaded successfully!")
        except Exception as e:
            messagebox.showwarning("Model Not Found", 
                                 f"Could not load model: {str(e)}\n\nPlease train the model first using train.py")
    
    def predict(self, image):
        if self.model is None:
            return None, 0
        
        with torch.no_grad():
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            output = self.model(image_tensor)
            probabilities = F.softmax(output, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
            return self.classes[predicted.item()], confidence.item() * 100
    
    def start_camera(self):
        if self.model is None:
            messagebox.showerror("Error", "Please load a model first!")
            return
        
        self.is_camera_running = True
        self.start_camera_btn.config(state='disabled')
        self.stop_camera_btn.config(state='normal')
        self.cap = cv2.VideoCapture(0)
        threading.Thread(target=self.process_camera, daemon=True).start()
    
    def stop_camera(self):
        self.is_camera_running = False
        self.start_camera_btn.config(state='normal')
        self.stop_camera_btn.config(state='disabled')
        if self.cap:
            self.cap.release()
    
    def process_camera(self):
        last_prediction = None
        stable_count = 0
        
        while self.is_camera_running:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)
            
            prediction, confidence = self.predict(pil_image)
            
            if prediction and confidence > 70:
                if prediction == last_prediction:
                    stable_count += 1
                    if stable_count >= 15:
                        self.add_character(prediction)
                        stable_count = 0
                else:
                    last_prediction = prediction
                    stable_count = 0
                
                cv2.putText(frame, f'{prediction}: {confidence:.1f}%', 
                          (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                self.prediction_label.config(text=f"Current Prediction: {prediction}")
                self.confidence_label.config(text=f"Confidence: {confidence:.2f}%")
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            img = img.resize((640, 480))
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)
    
    def load_image(self):
        if self.model is None:
            messagebox.showerror("Error", "Please load a model first!")
            return
        
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
        )
        
        if file_path:
            image = Image.open(file_path).convert('RGB')
            prediction, confidence = self.predict(image)
            
            if prediction:
                self.add_character(prediction)
                self.prediction_label.config(text=f"Current Prediction: {prediction}")
                self.confidence_label.config(text=f"Confidence: {confidence:.2f}%")
                
                cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                cv_image = cv2.resize(cv_image, (640, 480))
                cv2.putText(cv_image, f'{prediction}: {confidence:.1f}%', 
                          (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                frame_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                imgtk = ImageTk.PhotoImage(image=img)
                self.video_label.imgtk = imgtk
                self.video_label.configure(image=imgtk)
    
    def load_video(self):
        if self.model is None:
            messagebox.showerror("Error", "Please load a model first!")
            return
        
        file_path = filedialog.askopenfilename(
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv")]
        )
        
        if file_path:
            threading.Thread(target=self.process_video, args=(file_path,), daemon=True).start()
    
    def process_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        last_prediction = None
        stable_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)
            
            prediction, confidence = self.predict(pil_image)
            
            if prediction and confidence > 70:
                if prediction == last_prediction:
                    stable_count += 1
                    if stable_count >= 15:
                        self.add_character(prediction)
                        stable_count = 0
                else:
                    last_prediction = prediction
                    stable_count = 0
                
                cv2.putText(frame, f'{prediction}: {confidence:.1f}%', 
                          (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                self.prediction_label.config(text=f"Current Prediction: {prediction}")
                self.confidence_label.config(text=f"Confidence: {confidence:.2f}%")
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            img = img.resize((640, 480))
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)
            
            cv2.waitKey(30)
        
        cap.release()
    
    def add_character(self, char):
        if char == 'space':
            self.current_text += ' '
        elif char == 'del':
            self.current_text = self.current_text[:-1]
        elif char == 'nothing':
            pass
        else:
            self.current_text += char
        
        self.text_display.delete('1.0', tk.END)
        self.text_display.insert('1.0', self.current_text)
    
    def clear_text(self):
        self.current_text = ""
        self.text_display.delete('1.0', tk.END)

if __name__ == '__main__':
    root = tk.Tk()
    app = ASLApp(root)
    root.mainloop()