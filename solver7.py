import ctypes
try:
    ctypes.windll.user32.SetProcessDPIAware()
except Exception:
    pass

import cv2
import numpy as np
import tkinter as tk
from tkinter import messagebox
from PIL import ImageGrab, Image, ImageTk
import time
import customtkinter as ctk

import os

# ===== SETTINGS =====
TILE_SIZE = 220
N_OPTIONS = 3
# Ensure path is absolute to avoid relative path issues when CWD changes
PUZZLE_IMAGE_PATH = os.path.abspath("puzzle.png") 
DISPLAY_MAX_W = 1200  # max window width for preview
DISPLAY_MAX_H = 800   # max window height for preview

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

# ================= IMAGE PROCESSING =================
def remove_white_background(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
    b, g, r, a = cv2.split(image)
    white_mask = (b > 240) & (g > 240) & (r > 240)
    image[white_mask] = [0, 0, 0, 0]
    return image

def crop_to_content(image):
    alpha = image[:, :, 3]
    coords = cv2.findNonZero(alpha)
    if coords is None:
        return image
    x, y, w, h = cv2.boundingRect(coords)
    return image[y:y+h, x:x+w]

def resize_to_target(image, size):
    return cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)

def resize_for_display(img, max_w=DISPLAY_MAX_W, max_h=DISPLAY_MAX_H):
    h, w = img.shape[:2]
    scale = min(max_w / w, max_h / h, 1.0)
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

# ================= TEMPLATE MATCHING =================
def overlay_transparent(background, overlay, x, y):
    h, w = overlay.shape[:2]
    overlay_rgb = overlay[:, :, :3]
    mask = overlay[:, :, 3] / 255.0 if overlay.shape[2] == 4 else np.ones((h, w))
    bg_crop = background[y:y+h, x:x+w]
    for c in range(3):
        bg_crop[:, :, c] = (1 - mask) * bg_crop[:, :, c] + mask * overlay_rgb[:, :, c]
    background[y:y+h, x:x+w] = bg_crop

def find_top_matches(puzzle_img, piece_img, tile_size=TILE_SIZE, n_options=N_OPTIONS):
    piece_gray = cv2.cvtColor(piece_img[:, :, :3], cv2.COLOR_BGR2GRAY)
    puzzle_gray = cv2.cvtColor(puzzle_img, cv2.COLOR_BGR2GRAY)
    mask = piece_img[:, :, 3] if piece_img.shape[2] == 4 else None
    result = cv2.matchTemplate(puzzle_gray, piece_gray, cv2.TM_CCOEFF_NORMED, mask=mask)
    matches = []
    result_copy = result.copy()
    for _ in range(n_options):
        _, max_val, _, max_loc = cv2.minMaxLoc(result_copy)
        x, y = max_loc
        matches.append((max_val, x, y))
        x1 = max(0, x - tile_size//2)
        y1 = max(0, y - tile_size//2)
        x2 = min(result.shape[1], x + tile_size//2)
        y2 = min(result.shape[0], y + tile_size//2)
        result_copy[y1:y2, x1:x2] = -1
    return matches

# ================= BOARD SELECTOR =================
class ROISelector:
    def __init__(self, parent, screen_img):
        self.win = tk.Toplevel(parent)
        self.win.attributes("-fullscreen", True)
        self.win.attributes("-topmost", True)
        self.win.overrideredirect(True)

        rgb = cv2.cvtColor(screen_img, cv2.COLOR_BGR2RGB)
        self.tk_img = ImageTk.PhotoImage(Image.fromarray(rgb))

        self.canvas = tk.Canvas(self.win, cursor="cross", highlightthickness=0, bd=0)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.create_image(0, 0, anchor="nw", image=self.tk_img)
        self.canvas.create_text(20, 20,
            text="Drag over the PUZZLE BOARD AREA | ENTER = Confirm | ESC = Cancel",
            fill="#00FF00", font=("Helvetica", 18, "bold"), anchor="nw")

        self.rect = None
        self.start_x = self.start_y = None
        self.roi = None

        self.canvas.bind("<ButtonPress-1>",   self.on_press)
        self.canvas.bind("<B1-Motion>",       self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        self.win.bind("<Return>", self.confirm)
        self.win.bind("<Escape>", lambda e: self.cancel())

    def on_press(self, event):
        self.start_x, self.start_y = event.x, event.y
        if self.rect:
            self.canvas.delete(self.rect)
        self.rect = self.canvas.create_rectangle(
            self.start_x, self.start_y, self.start_x, self.start_y,
            outline="#00FF00", width=4)

    def on_drag(self, event):
        self.canvas.coords(self.rect, self.start_x, self.start_y, event.x, event.y)

    def on_release(self, event):
        x1 = min(self.start_x, event.x)
        y1 = min(self.start_y, event.y)
        x2 = max(self.start_x, event.x)
        y2 = max(self.start_y, event.y)
        self.roi = (x1, y1, x2 - x1, y2 - y1)
        self.canvas.delete("lbl")
        self.canvas.create_text(x1, max(y1 - 15, 20),
            text=f"{x2-x1} x {y2-y1} px — Press ENTER to Confirm",
            fill="#00FF00", font=("Helvetica", 14, "bold"), anchor="sw", tags="lbl")

    def confirm(self, event=None):
        if self.roi and self.roi[2] > 0 and self.roi[3] > 0:
            self.win.destroy()

    def cancel(self):
        self.roi = None
        self.win.destroy()

# ================= SCREEN OVERLAY =================
class ScreenOverlay(tk.Toplevel):
    def __init__(self, parent, screen_x, screen_y, pw, ph, duration=10000):
        super().__init__(parent)
        self.alpha_val = 0.9
        self.overrideredirect(True)
        self.attributes("-topmost", True)
        self.attributes("-alpha", 0.9)
        self.config(bg="white")
        self.attributes("-transparentcolor", "white")
        self.geometry(f"{pw}x{ph}+{screen_x}+{screen_y}")
        
        c = tk.Canvas(self, width=pw, height=ph, bg="white", highlightthickness=0)
        c.pack()
        # Animated-like thick border logic
        c.create_rectangle(0, 0, pw, ph, outline="#00FF00", width=12)
        if pw > 60 and ph > 30:
            c.create_text(pw // 2, ph // 2, text="PLACE PIECE", fill="#00FF00",
                          font=("Helvetica", max(10, ph // 6), "bold"))
        self.after(duration, self.destroy)
        self.after(200, self._make_click_through)

    def _make_click_through(self):
        try:
            # Use top-level handle for style changes
            frame = self.wm_frame()
            hwnd = int(frame, 16) if frame.startswith('0x') else self.winfo_id()
            style = ctypes.windll.user32.GetWindowLongW(hwnd, -20)
            ctypes.windll.user32.SetWindowLongW(hwnd, -20, style | 0x80000 | 0x20)
            self.attributes("-alpha", self.alpha_val)
        except Exception:
            pass

class FullBoardOverlay(tk.Toplevel):
    def __init__(self, parent, puzzle_cv_img, x, y, w, h, alpha=0.4):
        super().__init__(parent)
        self.alpha_val = alpha
        self.overrideredirect(True)
        self.attributes("-topmost", True)
        self.attributes("-alpha", alpha)
        self.geometry(f"{w}x{h}+{x}+{y}")
        
        # Guard against zero dimensions
        w_final = max(1, w)
        h_final = max(1, h)
        
        # Resize puzzle image to board area
        resized = cv2.resize(puzzle_cv_img, (w_final, h_final), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # Create persistent photo image
        self.tk_img = ImageTk.PhotoImage(Image.fromarray(rgb))
        
        self.canvas = tk.Canvas(self, width=w_final, height=h_final, highlightthickness=0, bg="black")
        self.canvas.pack()
        self.canvas.create_image(0, 0, anchor="nw", image=self.tk_img)
        # Strong reference to prevent garbage collection
        self.canvas.image = self.tk_img 
        
        self.after(200, self._make_click_through)

    def _make_click_through(self):
        try:
            # Get the actual top-level OS handle
            frame = self.wm_frame()
            hwnd = int(frame, 16) if frame.startswith('0x') else self.winfo_id()
            
            # GWL_EXSTYLE = -20, WS_EX_LAYERED = 0x80000, WS_EX_TRANSPARENT = 0x20
            style = ctypes.windll.user32.GetWindowLongW(hwnd, -20)
            ctypes.windll.user32.SetWindowLongW(hwnd, -20, style | 0x80000 | 0x20)
            
            # Always re-apply and refresh after style changes
            self.attributes("-alpha", self.alpha_val)
        except Exception:
            pass

# ================= GUI =================
class PasteSolverGUI(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        # Window Setup
        self.title("Jigsaw Assistant Pro")
        self.geometry("500x620")
        self.resizable(False, False)
        
        # Back-end Data
        self.is_overlay_on = False
        self.full_overlay = None
        self.board_roi = None
        
        # Configure Grid
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        # Header Section
        self.header_frame = ctk.CTkFrame(self, height=80, corner_radius=0, fg_color="#1a1a1a")
        self.header_frame.grid(row=0, column=0, sticky="nsew")
        
        self.title_label = ctk.CTkLabel(self.header_frame, text="🧩 JIGSAW PIECE SOLVER", 
                                        font=("Helvetica", 22, "bold"), text_color="#3b8ed0")
        self.title_label.place(relx=0.5, rely=0.4, anchor="center")
        
        self.subtitle_label = ctk.CTkLabel(self.header_frame, text="Copy piece to clipboard & press Ctrl+V", 
                                           font=("Helvetica", 12), text_color="gray")
        self.subtitle_label.place(relx=0.5, rely=0.75, anchor="center")

        # Main Body
        self.content_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.content_frame.grid(row=1, column=0, padx=20, pady=20, sticky="nsew")
        
        # --- Settings Panel ---
        self.settings_card = ctk.CTkFrame(self.content_frame, corner_radius=12)
        self.settings_card.pack(fill="x", pady=(0, 20))
        
        # Piece Size Slider
        self.slider_label = ctk.CTkLabel(self.settings_card, text=f"Piece Reference Size: {TILE_SIZE}px", 
                                         font=("Helvetica", 13, "bold"))
        self.slider_label.pack(pady=(12, 0))
        
        self.tile_slider = ctk.CTkSlider(self.settings_card, from_=50, to=500, number_of_steps=450,
                                         command=self.update_tile_info)
        self.tile_slider.set(TILE_SIZE)
        self.tile_slider.pack(padx=20, pady=(5, 10), fill="x")

        # Overlay Duration Slider
        self.duration_var = tk.IntVar(value=10)
        self.duration_label = ctk.CTkLabel(self.settings_card, text="Overlay Duration: 10s", 
                                           font=("Helvetica", 13, "bold"))
        self.duration_label.pack(pady=(5, 0))
        
        self.duration_slider = ctk.CTkSlider(self.settings_card, from_=1, to=20, number_of_steps=19,
                                             command=self.update_duration_info)
        self.duration_slider.set(10)
        self.duration_slider.pack(padx=20, pady=(5, 10), fill="x")

        # Overlay Alpha Slider
        self.alpha_label = ctk.CTkLabel(self.settings_card, text="Full Board Transparency: 40%", 
                                        font=("Helvetica", 13, "bold"))
        self.alpha_label.pack(pady=(5, 0))
        
        self.alpha_slider = ctk.CTkSlider(self.settings_card, from_=0.1, to=1.0, number_of_steps=90,
                                          command=self.update_alpha_info)
        self.alpha_slider.set(0.4)
        self.alpha_slider.pack(padx=20, pady=(5, 15), fill="x")

        # --- Preview Card ---
        self.preview_card = ctk.CTkFrame(self.content_frame, height=160, corner_radius=12)
        self.preview_card.pack(fill="x", pady=0)
        self.preview_card.pack_propagate(False)
        
        self.preview_placeholder = ctk.CTkLabel(self.preview_card, text="Pasted Piece Preview", 
                                                font=("Helvetica", 12, "italic"), text_color="#555555")
        self.preview_placeholder.pack(expand=True)
        
        self.preview_image_lbl = ctk.CTkLabel(self.preview_card, text="")
        self.preview_image_lbl.pack_forget()

        # --- Footer Actions ---
        self.action_frame = ctk.CTkFrame(self, height=180, corner_radius=0, fg_color="#1a1a1a")
        self.action_frame.grid(row=2, column=0, sticky="nsew")
        
        self.paste_btn = ctk.CTkButton(self.action_frame, text="📋 PASTE & SEARCH", 
                                       command=self.paste_image, height=45,
                                       font=("Helvetica", 15, "bold"))
        self.paste_btn.pack(pady=(15, 8), padx=40, fill="x")
        
        self.full_overlay_btn = ctk.CTkButton(self.action_frame, text="🖼️ Show Full Overlay", 
                                              command=self.toggle_full_overlay, height=35,
                                              fg_color="#333333", hover_color="#444444",
                                              border_width=1, border_color="#555555")
        self.full_overlay_btn.pack(pady=(0, 8), padx=60, fill="x")
        
        self.board_btn = ctk.CTkButton(self.action_frame, text="🎯 Set Board Area", 
                                       command=self.set_board_area, height=35,
                                       fg_color="#333333", hover_color="#444444",
                                       border_width=1, border_color="#555555")
        self.board_btn.pack(pady=(0, 8), padx=80, fill="x")
        
        self.board_status = ctk.CTkLabel(self.action_frame, text="Board Area Not Set", 
                                         font=("Helvetica", 10), text_color="#ffae00")
        self.board_status.pack(pady=(0, 10))

        self.puzzle_img = cv2.imread(PUZZLE_IMAGE_PATH)
        if self.puzzle_img is None:
            messagebox.showerror("Error", f"Critical: Cannot find '{PUZZLE_IMAGE_PATH}'.")
            self.destroy()
            return
        else:
            h, w = self.puzzle_img.shape[:2]
            print(f"[Core] Puzzle image loaded: {PUZZLE_IMAGE_PATH} ({w}x{h} px)")
            
        # Bindings
        self.bind_all("<Control-v>", lambda e: self.paste_image())
        self.bind_all("<Control-V>", lambda e: self.paste_image())

    def update_tile_info(self, val):
        self.slider_label.configure(text=f"Piece Reference Size: {int(val)}px")

    def update_duration_info(self, val):
        self.duration_label.configure(text=f"Overlay Duration: {int(val)}s")

    def update_alpha_info(self, val):
        self.alpha_label.configure(text=f"Full Board Transparency: {int(val*100)}%")
        if self.full_overlay:
            self.full_overlay.attributes("-alpha", val)

    def set_board_area(self):
        self.withdraw()
        time.sleep(0.4)
        screen = ImageGrab.grab()
        screen_bgr = cv2.cvtColor(np.array(screen), cv2.COLOR_RGB2BGR)
        selector = ROISelector(self, screen_bgr)
        self.wait_window(selector.win)
        self.deiconify()
        
        if selector.roi:
            self.board_roi = selector.roi
            x, y, w, h = selector.roi
            self.board_status.configure(text=f"Board Set: {w}x{h} @ {x},{y}", text_color="#4CAF50")
            # If overlay is on, update it
            if self.is_overlay_on:
                self.hide_overlay()
                self.show_overlay()
        else:
            self.board_status.configure(text="Capture Cancelled", text_color="#ffae00")

    def toggle_full_overlay(self):
        if not self.board_roi:
            messagebox.showwarning("Setup Incomplete", "Please set the Board Area first!")
            return
        
        if self.is_overlay_on:
            self.hide_overlay()
        else:
            self.show_overlay()

    def show_overlay(self):
        x, y, w, h = self.board_roi
        self.full_overlay = FullBoardOverlay(self, self.puzzle_img, x, y, w, h, self.alpha_slider.get())
        self.is_overlay_on = True
        self.full_overlay_btn.configure(text="🚫 Hide Full Overlay", fg_color="#E53935", hover_color="#D32F2F")

    def hide_overlay(self):
        if self.full_overlay:
            self.full_overlay.destroy()
            self.full_overlay = None
        self.is_overlay_on = False
        self.full_overlay_btn.configure(text="🖼️ Show Full Overlay", fg_color="#333333", hover_color="#444444")

    def paste_image(self):
        try:
            cv2.destroyAllWindows()
            img = ImageGrab.grabclipboard()
            if img is None:
                messagebox.showwarning("Clipboard Empty", "Please copy a piece screenshot first!")
                return

            # Update Preview
            preview_img = img.copy()
            preview_img.thumbnail((140, 140))
            ctk_img = ctk.CTkImage(light_image=preview_img, dark_image=preview_img, size=(140, 140))
            self.preview_placeholder.pack_forget()
            self.preview_image_lbl.configure(image=ctk_img)
            self.preview_image_lbl.image = ctk_img
            self.preview_image_lbl.pack(pady=10)

            # Process
            img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            tile_size = int(self.tile_slider.get())
            processed = remove_white_background(img_cv)
            processed = crop_to_content(processed)
            processed = resize_to_target(processed, tile_size)

            # Solve
            results = find_top_matches(self.puzzle_img, processed, tile_size)
            
            # Show top matches in CV window
            puzzle_copy = self.puzzle_img.copy()
            for i, (score, x, y) in enumerate(results):
                cv2.rectangle(puzzle_copy, (x, y), (x+tile_size, y+tile_size), (0, 255, 0), 3)
                cv2.putText(puzzle_copy, f"#{i+1} ({score:.2f})", (x, y-10),
                            cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 0), 2)
            
            cv2.imshow("Detection Results", resize_for_display(puzzle_copy))

            # Overlay
            if self.board_roi:
                best_score, best_x, best_y = results[0]
                img_h, img_w = self.puzzle_img.shape[:2]
                bx, by, bw, bh = self.board_roi
                
                scale_x, scale_y = bw / img_w, bh / img_h
                screen_x = int(bx + best_x * scale_x)
                screen_y = int(by + best_y * scale_y)
                box_w = max(10, int(tile_size * scale_x))
                box_h = max(10, int(tile_size * scale_y))
                
                dur_ms = int(self.duration_slider.get()) * 1000
                ScreenOverlay(self, screen_x, screen_y, box_w, box_h, duration=dur_ms)

        except Exception as e:
            messagebox.showerror("Runtime Error", str(e))

if __name__ == "__main__":
    app = PasteSolverGUI()
    app.mainloop()
