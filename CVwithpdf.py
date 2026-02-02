import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import os
import threading
from ultralytics import YOLO
import easyocr
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch

# === Model Paths ===
PRETRAINED_MODEL_PATH = "yolov8n.pt"  # Pretrained YOLOv8
CUSTOM_MODEL_PATH = r"C:\Python Projects\custom_training\yolov8_custom\weights\best.pt"  # Your trained model
CONF_THRESHOLD = 0.10

# === Detection Settings ===
CLASS_REPLACE = {"tv": "desktop", "backpack": "bag"}
IGNORE_CLASSES = {"person", "car", "potted plant", "microwave", "chair", "refrigerator"}
STANDARD_LIST = {"laptop", "desktop", "keyboard", "mouse", "file", "page", "filetray",
                 "calander", "headphone", "mobilephone", "nameplate", "book", "document", "stickynote"}
THUMBNAIL_SIZE = (120, 90)

# === EasyOCR Setup ===
model_dir = os.path.join(os.path.expanduser("~"), ".EasyOCR", "model")
ocr_reader = easyocr.Reader(['en'], gpu=False, model_storage_directory=model_dir, download_enabled=False)


class DeskApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🧠 Desk 5S Object Detector")
        self.root.geometry("1200x750")
        self.root.configure(bg="#eef3f9")
        self.root.option_add("*Font", ("Segoe UI", 10))

        self.current_image_path = None
        self.annotated_image = None
        self.report = None
        self.nameplate_text = None  # For OCR results

        # Will hold detections for PDF export
        self.last_standard = []
        self.last_nonstandard = []
        self.last_violations = []

        # Style
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TButton", padding=6, relief="flat", background="#3b82f6", foreground="white")
        style.map("TButton", background=[("active", "#2563eb")])

        # --- Top Button Panel ---
        top = ttk.Frame(root)
        top.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)

        ttk.Button(top, text="📷 Capture from Camera", command=self.open_camera).pack(side=tk.LEFT, padx=5)
        ttk.Button(top, text="🖼️ Choose Image", command=self.choose_image).pack(side=tk.LEFT, padx=5)
        self.btn_process = ttk.Button(top, text="⚙ Process Image", command=self.process_current_image)
        self.btn_process.pack(side=tk.LEFT, padx=5)
        ttk.Button(top, text="💾 Save Report", command=self.save_report).pack(side=tk.LEFT, padx=5)
        ttk.Button(top, text="❌ Exit", command=root.quit).pack(side=tk.RIGHT, padx=5)

        # --- Main Split Frame (Image Preview + Report) ---
        main_frame = ttk.Frame(root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Left: Image Preview
        left_frame = ttk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        self.preview_label = ttk.Label(left_frame, text="No image selected", anchor="center", background="white")
        self.preview_label.pack(fill=tk.BOTH, expand=True)

        # Right: Report Section
        right_frame = ttk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        report_label = ttk.Label(right_frame, text="📋 Desk Report", font=("Segoe UI", 12, "bold"))
        report_label.pack(anchor="w", pady=(0, 5))

        self.text_report = tk.Text(right_frame, wrap=tk.WORD, font=("Consolas", 10),
                                   bg="white", relief=tk.SUNKEN, height=8)
        self.text_report.pack(fill=tk.X, padx=5, pady=5)

        gallery_label = ttk.Label(right_frame, text="🔎 Detection Galleries", font=("Segoe UI", 11, "bold"))
        gallery_label.pack(anchor="w", pady=(8, 4))

        self.gallery_notebook = ttk.Notebook(right_frame)
        self.gallery_notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.tab_standard = ttk.Frame(self.gallery_notebook)
        self.gallery_notebook.add(self.tab_standard, text="Standard Items")
        self.tab_nonstandard = ttk.Frame(self.gallery_notebook)
        self.gallery_notebook.add(self.tab_nonstandard, text="Non-Standard Items")
        self.tab_violations = ttk.Frame(self.gallery_notebook)
        self.gallery_notebook.add(self.tab_violations, text="Violations")

        self.standard_canvas, self.standard_inner = self._make_scrollable(self.tab_standard)
        self.nonstandard_canvas, self.nonstandard_inner = self._make_scrollable(self.tab_nonstandard)
        self.viol_canvas, self.viol_inner = self._make_scrollable(self.tab_violations)

        self.status_var = tk.StringVar(value="Loading models...")
        status_bar = ttk.Label(root, textvariable=self.status_var, anchor=tk.W, relief=tk.SUNKEN)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        threading.Thread(target=self._bg_load_models, daemon=True).start()

    def _make_scrollable(self, parent):
        canvas = tk.Canvas(parent, bg="#ffffff", highlightthickness=0)
        vsb = ttk.Scrollbar(parent, orient="horizontal", command=canvas.xview)
        inner = ttk.Frame(canvas)
        canvas.create_window((0, 0), window=inner, anchor='nw')

        def _on_config(event):
            canvas.configure(scrollregion=canvas.bbox("all"))

        inner.bind("<Configure>", _on_config)
        canvas.configure(xscrollcommand=vsb.set)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vsb.pack(side=tk.BOTTOM, fill=tk.X)
        return canvas, inner

    def set_status(self, text):
        self.status_var.set(text)
        self.root.update_idletasks()

    def _bg_load_models(self):
        try:
            self.model_pretrained = YOLO(PRETRAINED_MODEL_PATH)
            self.model_custom = YOLO(CUSTOM_MODEL_PATH)
            self.set_status("✅ Both models loaded and ready.")
        except Exception as e:
            self.set_status(f"❌ Model load failed: {e}")

    def choose_image(self):
        path = filedialog.askopenfilename(title="Select image",
                                          filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp")])
        if not path:
            return
        self.current_image_path = path
        self.set_status(f"Selected: {os.path.basename(path)}")
        self.show_preview(path)

    def open_camera(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            messagebox.showerror("Camera Error", "Could not open webcam.")
            return
        self.set_status("Camera opened. Press 'c' to capture, 'q' to cancel.")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            cv2.imshow("Camera - press 'c' to capture or 'q' to cancel", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('c'):
                save_path = os.path.join(os.getcwd(), "captured_image.jpg")
                cv2.imwrite(save_path, frame)
                self.current_image_path = save_path
                self.show_preview(save_path)
                self.set_status(f"Captured image: {save_path}")
                break
            if key == ord('q'):
                self.set_status("Camera capture cancelled.")
                break
        cap.release()
        cv2.destroyAllWindows()

    def show_preview(self, image_path):
        try:
            pil = Image.open(image_path)
            pil.thumbnail((800, 600))
            self.preview_imgtk = ImageTk.PhotoImage(pil)
            self.preview_label.configure(image=self.preview_imgtk, text="")
        except Exception as e:
            self.preview_label.configure(text=f"Preview error: {e}")

    def process_current_image(self):
        if not self.current_image_path:
            messagebox.showwarning("No image", "Choose or capture an image first.")
            return
        self.set_status("Processing image...")
        self.btn_process.config(state=tk.DISABLED)
        threading.Thread(target=self._process_thread, daemon=True).start()

    def _process_thread(self):
        try:
            results_pre = self.model_pretrained(self.current_image_path, conf=CONF_THRESHOLD, verbose=False)[0]
            results_custom = self.model_custom(self.current_image_path, conf=CONF_THRESHOLD, verbose=False)[0]
            img_bgr = cv2.imread(self.current_image_path)
            h, w = img_bgr.shape[:2]

            detections = []

            def extract_dets(result, model_names):
                dets = []
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    label = model_names[cls_id].lower()
                    if label in IGNORE_CLASSES:
                        continue
                    label = CLASS_REPLACE.get(label, label)
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w - 1, x2), min(h - 1, y2)
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    conf = float(box.conf[0])
                    dets.append({"label": label, "conf": conf,
                                 "bbox": (x1, y1, x2, y2), "center": (cx, cy)})
                return dets

            detections += extract_dets(results_pre, self.model_pretrained.names)
            detections += extract_dets(results_custom, self.model_custom.names)

            # --- EasyOCR for Name Plate ---
            self.nameplate_text = None
            for d in detections:
                if d["label"] == "nameplate":
                    x1, y1, x2, y2 = d["bbox"]
                    crop_img = img_bgr[y1:y2, x1:x2] if y2 > y1 and x2 > x1 else img_bgr
                    ocr_results = ocr_reader.readtext(crop_img)
                    if ocr_results:
                        best = max(ocr_results, key=lambda r: r[2])
                        self.nameplate_text = best[1]
                    break

            standard_items = [d for d in detections if d["label"] in STANDARD_LIST]
            nonstandard_items = [d for d in detections if d["label"] not in STANDARD_LIST]
            violations = self._check_violations(detections, w, h)

            # Save for PDF report later
            self.last_standard = standard_items
            self.last_nonstandard = nonstandard_items
            self.last_violations = violations

            # Annotate
            annot = img_bgr.copy()
            for d in detections:
                x1, y1, x2, y2 = d["bbox"]
                lbl = d["label"]
                conf = d["conf"]
                color = (0, 200, 0) if lbl in STANDARD_LIST else (0, 0, 200)
                cv2.rectangle(annot, (x1, y1), (x2, y2), color, 12)
                cv2.putText(annot, f"{lbl} {conf:.2f}", (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 8, color, 14)
            out_path = os.path.join(os.getcwd(), "annotated_output.jpg")
            cv2.imwrite(out_path, annot)
            self.annotated_image = out_path
            self.show_preview(out_path)

            report_text = self._generate_report_text(detections, standard_items, nonstandard_items, violations)
            self.report = report_text
            self._display_report(report_text, standard_items, nonstandard_items, violations, img_bgr)

            self.set_status("✅ Processing completed.")
        except Exception as e:
            messagebox.showerror("Processing error", str(e))
            self.set_status("❌ Processing failed.")
        finally:
            self.btn_process.config(state=tk.NORMAL)

    def _check_violations(self, detections, w, h):
        centers = {}
        for d in sorted(detections, key=lambda x: -x["conf"]):
            lbl = d["label"]
            if lbl not in centers:
                centers[lbl] = {"center": d["center"], "bbox": d["bbox"], "conf": d["conf"]}
        violations = []

        if "mouse" in centers:
            cx, cy = centers["mouse"]["center"]
            if cx >= int(0.66 * w):
                violations.append({"rule": "Mouse should be on the RIGHT side of the desk",
                                   "label": "mouse", "bbox": centers["mouse"]["bbox"]})
        if "desktop" in centers:
            cx, cy = centers["desktop"]["center"]
            if not (w // 3 < cx < 2 * w // 3):
                violations.append({"rule": "Desktop should be at the CENTER of desk",
                                   "label": "desktop", "bbox": centers["desktop"]["bbox"]})
        if "laptop" in centers:
            cx, cy = centers["laptop"]["center"]
            if not (w // 3 < cx < 2 * w // 3):
                violations.append({"rule": "Laptop should be at the CENTER of desk",
                                   "label": "laptop", "bbox": centers["laptop"]["bbox"]})
        if "keyboard" in centers:
            k_cx, k_cy = centers["keyboard"]["center"]
            kb_bbox = centers["keyboard"]["bbox"]
            in_front_ok = False
            if "desktop" in centers:
                d_cx, d_cy = centers["desktop"]["center"]
                if k_cy > d_cy + 10:
                    in_front_ok = True
            if not in_front_ok and "laptop" in centers:
                l_cx, l_cy = centers["laptop"]["center"]
                if k_cy > l_cy + 10:
                    in_front_ok = True
            if not in_front_ok:
                violations.append({"rule": "Keyboard should be positioned in FRONT of the Desktop or Laptop",
                                   "label": "keyboard", "bbox": kb_bbox})
        return violations

    def _generate_report_text(self, all_dets, std_items, nonstd_items, violations):
        total = len(all_dets)
        std_count = len(std_items)
        nonstd_count = len(nonstd_items)
        std_pct = (std_count / total * 100) if total else 0
        nonstd_pct = (nonstd_count / total * 100) if total else 0

        lines = []
        lines.append("📋 Desk Analysis Report")
        lines.append("=" * 40)

        if self.nameplate_text:
            lines.append("\n🔤 Person NAME: " + self.nameplate_text)
        else:
            lines.append("  None")
    
        lines.append(f"Total Objects Detected: {total}")
        lines.append(f"Standard Objects: {std_count} ({std_pct:.2f}%)")
        lines.append(f"Non-Standard Objects: {nonstd_count} ({nonstd_pct:.2f}%)")
        lines.append("")
        lines.append("✔ Standard Items:")
        if std_items:
            for d in std_items:
                lines.append(f"  - {d['label']} (conf={d['conf']:.2f})")
        else:
            lines.append("  None")
        lines.append("")
        lines.append("✖ Non-Standard Items:")
        if nonstd_items:
            for d in nonstd_items:
                lines.append(f"  - {d['label']} (conf={d['conf']:.2f})")
        else:
            lines.append("  None")
        lines.append("")
        lines.append("⚠ Violations:")
        if violations:
            for v in violations:
                lines.append(f"  - {v['rule']} (object: {v['label']})")
        else:
            lines.append("  None ✅")
        '''if self.nameplate_text:
            lines.append("\n🔤 Name Plate OCR: " + self.nameplate_text)'''
        return "\n".join(lines)

    def _display_report(self, report_text, standard_items, nonstandard_items, violations, orig_img):
        self.text_report.delete(1.0, tk.END)
        self.text_report.insert(tk.END, report_text)

        for widget in self.standard_inner.winfo_children():
            widget.destroy()
        for widget in self.nonstandard_inner.winfo_children():
            widget.destroy()
        for widget in self.viol_inner.winfo_children():
            widget.destroy()

        def add_thumb(parent, label_text, crop_bgr, extra_text=None):
            try:
                crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
                pil = Image.fromarray(crop_rgb)
                pil.thumbnail(THUMBNAIL_SIZE)
                imgtk = ImageTk.PhotoImage(pil)
            except:
                pil = Image.new("RGB", THUMBNAIL_SIZE, (200, 200, 200))
                imgtk = ImageTk.PhotoImage(pil)
            frame = ttk.Frame(parent, relief=tk.RIDGE, padding=4)
            frame.pack(side=tk.LEFT, padx=6, pady=6)
            lbl_img = ttk.Label(frame, image=imgtk)
            lbl_img.image = imgtk
            lbl_img.pack()
            ttk.Label(frame, text=label_text).pack()
            if extra_text:
                ttk.Label(frame, text=extra_text, foreground="red", wraplength=THUMBNAIL_SIZE[0]).pack()

        if standard_items:
            for d in standard_items:
                x1, y1, x2, y2 = d["bbox"]
                crop = orig_img[y1:y2, x1:x2] if y2 > y1 and x2 > x1 else orig_img
                add_thumb(self.standard_inner, f"{d['label']} ({d['conf']:.2f})", crop)
        else:
            ttk.Label(self.standard_inner, text="No standard items detected").pack(padx=6, pady=6)

        if nonstandard_items:
            for d in nonstandard_items:
                x1, y1, x2, y2 = d["bbox"]
                crop = orig_img[y1:y2, x1:x2] if y2 > y1 and x2 > x1 else orig_img
                add_thumb(self.nonstandard_inner, f"{d['label']} ({d['conf']:.2f})", crop)
        else:
            ttk.Label(self.nonstandard_inner, text="No non-standard items detected").pack(padx=6, pady=6)

        if violations:
            for v in violations:
                x1, y1, x2, y2 = v["bbox"]
                crop = orig_img[y1:y2, x1:x2] if y2 > y1 and x2 > x1 else orig_img
                add_thumb(self.viol_inner, v["label"], crop, extra_text=v["rule"])
        else:
            ttk.Label(self.viol_inner, text="No violations detected").pack(padx=6, pady=6)

    def save_report(self):
        if not self.report:
            messagebox.showwarning("No report", "No report to save. Process an image first.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".pdf",
                                            filetypes=[("PDF file", "*.pdf")],
                                            initialfile="desk_report.pdf")
        if not path:
            return
        try:
            c = canvas.Canvas(path, pagesize=A4)
            width, height = A4
            margin = 1 * inch
            y_pos = height - margin

            text_obj = c.beginText(margin, y_pos)
            text_obj.setFont("Helvetica", 11)
            for line in self.report.split("\n"):
                if y_pos < margin + 100:
                    c.drawText(text_obj)
                    c.showPage()
                    text_obj = c.beginText(margin, height - margin)
                    text_obj.setFont("Helvetica", 11)
                    y_pos = height - margin
                text_obj.textLine(line)
                y_pos -= 14
            c.drawText(text_obj)

            if self.annotated_image and os.path.exists(self.annotated_image):
                img_width = width - 2 * inch
                c.drawImage(self.annotated_image, margin, margin + 3 * inch,
                            width=img_width, preserveAspectRatio=True, mask='auto')

            def embed_thumbs_grid(section_title, items, y_start):
                nonlocal c
                if not items:
                    return y_start
                y = y_start - 40
                c.setFont("Helvetica-Bold", 12)
                c.drawString(margin, y, section_title)
                y -= 20
                cols = 3
                thumb_w, thumb_h = 100, 80
                x = margin
                col_count = 0
                for d in items:
                    x1, y1, x2, y2 = d["bbox"]
                    crop = cv2.imread(self.current_image_path)[y1:y2, x1:x2]
                    if crop is None or crop.size == 0:
                        continue
                    tmp_path = f"_tmp_{d['label']}.jpg"
                    cv2.imwrite(tmp_path, crop)
                    try:
                        if y < margin + thumb_h:
                            c.showPage()
                            y = height - margin - thumb_h - 20
                            x = margin
                            col_count = 0
                        c.drawImage(tmp_path, x, y - thumb_h, width=thumb_w, height=thumb_h)
                        x += thumb_w + 20
                        col_count += 1
                        if col_count >= cols:
                            col_count = 0
                            x = margin
                            y -= thumb_h + 20
                    finally:
                        if os.path.exists(tmp_path):
                            os.remove(tmp_path)
                return y - thumb_h - 20

            y_pos = embed_thumbs_grid("✔ Standard Items", self.last_standard, y_pos)
            y_pos = embed_thumbs_grid("✖ Non-Standard Items", self.last_nonstandard, y_pos)
            y_pos = embed_thumbs_grid("⚠ Violations", self.last_violations, y_pos)

            c.save()
            messagebox.showinfo("Saved", f"PDF report saved to {path}")
            self.set_status(f"Report saved: {os.path.basename(path)}")
        except Exception as e:
            messagebox.showerror("Save Error", f"Failed to save report: {e}")


if __name__ == "__main__":
    root = tk.Tk()
    app = DeskApp(root)
    root.mainloop()
