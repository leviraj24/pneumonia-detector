# src/app.py
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import torch
from torchvision import transforms, models
import os
import threading
import shutil
from datetime import datetime
import numpy as np
import cv2
import csv

# --- Try to import Grad-CAM ---
try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    from pytorch_grad_cam.utils.image import show_cam_on_image
    GRADCAM_AVAILABLE = True
except ImportError:
    GRADCAM_AVAILABLE = False
    print("WARNING: 'pytorch-grad-cam' is not installed. Explanation feature will be disabled.")

class PneumoniaDetectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🫁 Pneumonia Detection & Explainable AI System")
        self.root.geometry("1050x750")
        self.root.minsize(1000, 700)

        # --- Color Palette & Fonts ---
        self.colors = {
            "bg_main": "#f0f2f5",
            "bg_frame": "#ffffff",
            "bg_header": "#2d3e50",
            "text_light": "#ffffff",
            "text_dark": "#34495e",
            "text_secondary": "#5d6d7e",
            "accent_blue": "#3a98d9",
            "accent_green": "#27ae60",
            "accent_orange": "#f39c12",
            "accent_red": "#e74c3c",
            "accent_success": "#2ecc71",
        }
        self.fonts = {
            "title": ("Segoe UI", 20, "bold"),
            "header": ("Segoe UI", 14, "bold"),
            "button": ("Segoe UI", 12, "bold"),
            "label": ("Segoe UI", 12, "bold"),
            "text": ("Segoe UI", 10),
        }
        
        self.root.configure(bg=self.colors["bg_main"])

        self.model_path = os.path.join("models", "best_efficientnet_b0.pth")
        self.feedback_dir = os.path.join("data", "feedback")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)), 
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        self.current_image_path = None
        self.current_prediction_idx = None
        self.current_prediction_str = None
        self.input_tensor = None

        self.load_model()
        self.setup_ui()
        self.update_status("Welcome! Please browse an image to begin.")

    def load_model(self):
        try:
            self.model = models.efficientnet_b0(weights=None)
            self.model.classifier[1] = torch.nn.Linear(self.model.classifier[1].in_features, 2)
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            self.model.to(self.device).eval()
            print(f"✅ Model loaded successfully from {self.model_path}")
        except Exception as e:
            messagebox.showerror("Model Error", f"Could not load model: {e}\nPlease run train_advanced.py first.")
            self.root.destroy()

    def setup_ui(self):
        # Main container
        main_container = tk.Frame(self.root, bg=self.colors["bg_main"])
        main_container.pack(fill="both", expand=True, padx=20, pady=20)
        
        # Title
        title_label = tk.Label(main_container, text="Pneumonia Detection & XAI", 
                              font=self.fonts["title"], fg=self.colors["text_dark"], bg=self.colors["bg_main"])
        title_label.pack(pady=(0, 20))
        
        # Content frame
        content_frame = tk.Frame(main_container, bg=self.colors["bg_main"])
        content_frame.pack(fill="both", expand=True)
        
        # --- Left Panel (Image) ---
        left_panel = tk.Frame(content_frame, bg=self.colors["bg_frame"], relief="solid", bd=1)
        left_panel.pack(side="left", fill="both", expand=True, padx=(0, 10))
        
        tk.Label(left_panel, text=" Chest X-Ray Image", 
                 font=self.fonts["header"], fg=self.colors["text_dark"], bg=self.colors["bg_frame"], anchor="w").pack(fill="x", padx=15, pady=10)
        
        self.image_label = tk.Label(left_panel, text="Click 'Browse Image' to select an X-ray", 
                                   bg="#e5e8e8", fg=self.colors["text_secondary"], font=self.fonts["text"])
        self.image_label.pack(fill="both", expand=True, padx=15, pady=15)
        
        # --- Right Panel (Controls) ---
        right_panel = tk.Frame(content_frame, bg=self.colors["bg_frame"], relief="solid", bd=1, width=350)
        right_panel.pack(side="right", fill="y", padx=(10, 0))
        right_panel.pack_propagate(False)
        
        control_content = tk.Frame(right_panel, bg=self.colors["bg_frame"])
        control_content.pack(fill="both", expand=True, padx=20, pady=20)
        
        # Buttons
        self.browse_btn = self._create_styled_button(control_content, "📁 Browse Image", self.browse_image, self.colors["accent_blue"])
        self.analyze_btn = self._create_styled_button(control_content, "🔬 Analyze Image", self.analyze_image, self.colors["accent_green"], state="disabled")
        self.explain_btn = self._create_styled_button(control_content, "🧠 Show Explanation", self.show_explanation, self.colors["accent_orange"], state="disabled")
        
        if not GRADCAM_AVAILABLE:
            self.explain_btn.config(text="⚠️ Grad-CAM Not Available", bg="#95a5a6", cursor="arrow", state="disabled")
        
        # --- Results Section ---
        results_frame = tk.LabelFrame(control_content, text=" Analysis Results ", font=self.fonts["label"], bg=self.colors["bg_frame"], fg=self.colors["text_dark"], relief="solid", bd=1)
        results_frame.pack(fill="x", pady=(20, 15), ipady=10)
        
        self.prediction_label = tk.Label(results_frame, text="Awaiting Analysis", font=("Segoe UI", 16, "bold"), bg=self.colors["bg_frame"], fg=self.colors["text_secondary"])
        self.prediction_label.pack(pady=(10, 5))
        
        confidence_frame = tk.Frame(results_frame, bg=self.colors["bg_frame"])
        confidence_frame.pack(fill="x", padx=15, pady=(5, 15))
        
        style = ttk.Style()
        style.theme_use('clam')
        style.configure("green.Horizontal.TProgressbar", background=self.colors["accent_green"])
        style.configure("red.Horizontal.TProgressbar", background=self.colors["accent_red"])
        
        self.progress = ttk.Progressbar(confidence_frame, mode='determinate', style="green.Horizontal.TProgressbar")
        self.progress.pack(side="left", fill="x", expand=True, padx=(0, 10))
        self.confidence_label = tk.Label(confidence_frame, text="--%", font=("Segoe UI", 12, "bold"), bg=self.colors["bg_frame"], fg=self.colors["text_dark"], width=5)
        self.confidence_label.pack(side="right")
        
        # --- Feedback Section ---
        feedback_frame = tk.LabelFrame(control_content, text=" Provide Feedback ", font=self.fonts["label"], bg=self.colors["bg_frame"], fg=self.colors["text_dark"], relief="solid", bd=1)
        feedback_frame.pack(fill="x", ipady=10)
        
        tk.Label(feedback_frame, text="Was this prediction correct?", font=self.fonts["text"], bg=self.colors["bg_frame"], fg=self.colors["text_dark"]).pack(pady=5)
        
        feedback_btns_frame = tk.Frame(feedback_frame, bg=self.colors["bg_frame"])
        feedback_btns_frame.pack(fill="x", padx=10, pady=10)
        
        self.correct_btn = tk.Button(feedback_btns_frame, text="✓ Correct", font=self.fonts["text"], bg=self.colors["accent_success"], fg="white", state="disabled", relief="flat", command=lambda: self.handle_feedback(True))
        self.correct_btn.pack(side="left", fill="x", expand=True, padx=(0, 5), ipady=5)
        self.incorrect_btn = tk.Button(feedback_btns_frame, text="✗ Incorrect", font=self.fonts["text"], bg=self.colors["accent_red"], fg="white", state="disabled", relief="flat", command=lambda: self.handle_feedback(False))
        self.incorrect_btn.pack(side="right", fill="x", expand=True, padx=(5, 0), ipady=5)

        # --- Status Bar ---
        self.status_bar = tk.Label(self.root, text="", font=("Segoe UI", 9), relief="sunken", anchor="w", bd=1, bg="#fafafa", fg="#333")
        self.status_bar.pack(side="bottom", fill="x")

    def _create_styled_button(self, parent, text, command, color, state="normal"):
        """Helper to create and pack a styled button with hover effects."""
        hover_color = self._adjust_brightness(color, -0.2)
        
        button = tk.Button(parent, text=text, command=command, font=self.fonts["button"], 
                           bg=color, fg=self.colors["text_light"], relief="flat",
                           pady=12, state=state, cursor="hand2")
        button.pack(fill="x", pady=5)
        
        button.bind("<Enter>", lambda e, c=hover_color: e.widget.config(bg=c) if e.widget['state'] != 'disabled' else None)
        button.bind("<Leave>", lambda e, c=color: e.widget.config(bg=c) if e.widget['state'] != 'disabled' else None)
        return button

    def _adjust_brightness(self, hex_color, factor):
        """Adjusts hex color brightness. factor < 0 darkens, > 0 lightens."""
        if hex_color.startswith('#'):
            hex_color = hex_color[1:]
        r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
        
        r = int(max(0, min(255, r + 255 * factor)))
        g = int(max(0, min(255, g + 255 * factor)))
        b = int(max(0, min(255, b + 255 * factor)))
        
        return f"#{r:02x}{g:02x}{b:02x}"

    def update_status(self, message):
        self.status_bar.config(text=f"  {message}")
        
    def get_target_layer(self):
        """Get the last convolutional block for EfficientNet."""
        return self.model.features[-1]

    def show_explanation(self):
        if not GRADCAM_AVAILABLE:
            messagebox.showerror("Missing Library", "Please run: pip install pytorch-grad-cam")
            return
        if self.input_tensor is None or self.current_prediction_idx is None:
            messagebox.showwarning("No Analysis", "Please analyze an image first.")
            return

        self.update_status("Generating explanation... This may take a moment.")
        threading.Thread(target=self._generate_explanation_worker, daemon=True).start()

    def _generate_explanation_worker(self):
        try:
            target_layer = self.get_target_layer()
            targets = [ClassifierOutputTarget(self.current_prediction_idx)]
            
            original_pil = Image.open(self.current_image_path).convert("RGB")
            
            # --- FIX: Resize to model's input size (224x224) for Grad-CAM calculation ---
            vis_image_rgb = np.array(original_pil.resize((224, 224))) / 255.0

            with GradCAM(model=self.model, target_layers=[target_layer]) as cam:
                grayscale_cam = cam(input_tensor=self.input_tensor.clone(), targets=targets)[0, :]
            
            # Now both vis_image_rgb and grayscale_cam are compatible
            heatmap_image = show_cam_on_image(vis_image_rgb, grayscale_cam, use_rgb=True, image_weight=0.5)
            original_image_display = (vis_image_rgb * 255).astype(np.uint8)

            # --- Interactive Visualization Window ---
            show_heatmap = True
            window_name = 'Grad-CAM Analysis'
            display_size = (512, 512) # Define a larger size for the display window
            
            while True:
                # Select the base 224x224 image
                base_frame = heatmap_image.copy() if show_heatmap else original_image_display.copy()
                
                # --- FIX: Upscale the final image for better viewing ---
                display_frame = cv2.resize(base_frame, display_size, interpolation=cv2.INTER_AREA)

                # Add text overlays to the upscaled frame
                pred_color = (60, 60, 231) if self.current_prediction_str == "PNEUMONIA" else (96, 174, 39)
                confidence, _ = torch.max(torch.softmax(self.model(self.input_tensor), dim=1), 1)

                # Info Box
                cv2.rectangle(display_frame, (0, 0), (display_size[0], 90), (30, 30, 30), -1)
                cv2.putText(display_frame, f"Prediction: {self.current_prediction_str}", (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, pred_color, 2)
                cv2.putText(display_frame, f"Confidence: {confidence.item():.2%}", (15, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                
                # Instructions
                instruction_text = "Press 'H' to Toggle Heatmap | 'S' to Save | 'Q' to Quit"
                cv2.rectangle(display_frame, (0, display_size[1] - 30), (display_size[0], display_size[1]), (30, 30, 30), -1)
                cv2.putText(display_frame, instruction_text, (20, display_size[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)

                cv2.imshow(window_name, cv2.cvtColor(display_frame, cv2.COLOR_RGB2BGR))
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    break
                elif key == ord('h'):
                    show_heatmap = not show_heatmap
                elif key == ord('s'):
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    filename = f"gradcam_{self.current_prediction_str}_{timestamp}.png"
                    save_path = os.path.join("explanations", filename)
                    os.makedirs("explanations", exist_ok=True)
                    cv2.imwrite(save_path, cv2.cvtColor(display_frame, cv2.COLOR_RGB2BGR))
                    print(f"💾 Explanation saved: {save_path}")
                    # Show temporary saved message on screen
                    save_feedback_frame = display_frame.copy()
                    cv2.putText(save_feedback_frame, "SAVED!", (int(display_size[0]/2)-70, int(display_size[1]/2)), cv2.FONT_HERSHEY_DUPLEX, 1.5, (96, 255, 96), 3)
                    cv2.imshow(window_name, cv2.cvtColor(save_feedback_frame, cv2.COLOR_RGB2BGR))
                    cv2.waitKey(1000)

            cv2.destroyAllWindows()
            self.root.after(0, self.update_status, "Ready.")
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.root.after(0, messagebox.showerror, "Grad-CAM Error", f"Failed to generate explanation:\n{e}")

    def handle_feedback(self, is_correct):
        if not self.current_image_path or self.current_prediction_str is None: return

        feedback_label = self.current_prediction_str if is_correct else ("NORMAL" if self.current_prediction_str == "PNEUMONIA" else "PNEUMONIA")
        
        # Save the image
        image_filename = os.path.basename(self.current_image_path)
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        unique_filename = f"{timestamp}_{image_filename}"
        dest_folder = os.path.join(self.feedback_dir, feedback_label)
        os.makedirs(dest_folder, exist_ok=True)
        dest_path = os.path.join(dest_folder, unique_filename)
        
        try:
            shutil.copy2(self.current_image_path, dest_path)
            message = f"Feedback saved. Image logged as '{feedback_label}'."
            messagebox.showinfo("Feedback Saved", message)
            
            # Log to CSV
            log_path = os.path.join(self.feedback_dir, "feedback_log.csv")
            log_header = ["timestamp", "image_path", "prediction", "feedback", "final_label", "saved_path"]
            log_exists = os.path.isfile(log_path)
            
            with open(log_path, 'a', newline='') as f:
                writer = csv.writer(f)
                if not log_exists:
                    writer.writerow(log_header)
                writer.writerow([datetime.now(), self.current_image_path, self.current_prediction_str, 
                                 "correct" if is_correct else "incorrect", feedback_label, dest_path])
            
            self.update_status(message)
        except Exception as e:
            messagebox.showerror("Error Saving Feedback", f"Could not save file.\nError: {e}")

        self.correct_btn.config(state="disabled")
        self.incorrect_btn.config(state="disabled")

    def browse_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if file_path:
            self.current_image_path = file_path
            self.display_image(file_path)
            self.analyze_btn.config(state="normal")
            self.reset_results()
            self.update_status(f"Loaded: {os.path.basename(file_path)}. Ready to analyze.")

    def display_image(self, image_path):
        try:
            pil_image = Image.open(image_path)
            w, h = pil_image.size
            max_size = self.image_label.winfo_width() - 30
            ratio = max_size / max(w, h)
            new_w, new_h = int(w * ratio), int(h * ratio)
            
            pil_image = pil_image.resize((new_w, new_h), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(pil_image)
            
            self.image_label.configure(image=photo, text="", bg=self.colors["bg_frame"])
            self.image_label.image = photo
        except Exception as e:
            messagebox.showerror("Image Error", f"Could not load image: {e}")

    def reset_results(self):
        self.prediction_label.config(text="Awaiting Analysis", fg=self.colors["text_secondary"])
        self.confidence_label.config(text="--%")
        self.progress['value'] = 0
        self.correct_btn.config(state="disabled")
        self.incorrect_btn.config(state="disabled")
        self.explain_btn.config(state="disabled")
        self.current_prediction_idx = self.current_prediction_str = self.input_tensor = None

    def analyze_image(self):
        if not self.current_image_path or not self.model: return
        
        self.analyze_btn.config(state="disabled", text="🔬 Analyzing...")
        self.update_status("Analyzing image...")
        self.progress.config(mode="indeterminate")
        self.progress.start()
        threading.Thread(target=self._analyze_worker, daemon=True).start()

    def _analyze_worker(self):
        try:
            image = Image.open(self.current_image_path).convert("RGB")
            self.input_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(self.input_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
            
            self.current_prediction_idx = predicted.item()
            classes = ["NORMAL", "PNEUMONIA"]
            self.current_prediction_str = classes[self.current_prediction_idx]
            conf_score = confidence.item()
            
            self.root.after(0, self._update_results, self.current_prediction_str, conf_score)
        except Exception as e:
            error_msg = f"Analysis failed: {str(e)}"
            self.root.after(0, lambda: messagebox.showerror("Analysis Error", error_msg))
            self.root.after(0, self.update_status, "Analysis failed. Please try another image.")

    def _update_results(self, prediction, confidence):
        self.progress.stop()
        self.progress.config(mode="determinate")
        self.analyze_btn.config(state="normal", text="🔬 Analyze Image")
        
        color = self.colors["accent_red"] if prediction == "PNEUMONIA" else self.colors["accent_green"]
        style = "red.Horizontal.TProgressbar" if prediction == "PNEUMONIA" else "green.Horizontal.TProgressbar"
        
        self.prediction_label.config(text=prediction, fg=color)
        self.confidence_label.config(text=f"{confidence*100:.1f}%")
        self.progress.config(style=style)
        self.progress['value'] = confidence * 100
        
        self.correct_btn.config(state="normal")
        self.incorrect_btn.config(state="normal")
        if GRADCAM_AVAILABLE: self.explain_btn.config(state="normal")
        
        self.update_status("Analysis complete. You can now view the explanation or provide feedback.")

def main():
    print("🚀 Starting Pneumonia Detection App...")
    print(f"🖥️ Using device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    
    root = tk.Tk()
    app = PneumoniaDetectorApp(root)
    
    try:
        root.mainloop()
    except KeyboardInterrupt:
        print("\n👋 Application closed by user")
    except Exception as e:
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()