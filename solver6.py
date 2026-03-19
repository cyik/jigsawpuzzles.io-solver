import ctypes
# ===== CRITICAL: Fix Windows DPI scaling so tkinter and ImageGrab
# both use physical pixels. Without this, the overlay lands in the
# wrong place and at the wrong size on high-DPI screens.
try:
    ctypes.windll.user32.SetProcessDPIAware()
except:
    pass

import cv2
import numpy as np
import tkinter as tk
from tkinter import messagebox
from PIL import ImageGrab, ImageTk, Image
import os
import time

# ===== SETTINGS =====
# TILE_SIZE controls ONLY the suppression radius when searching for options.
# It does NOT resize your piece — matching is always done at the piece's real size.
TILE_SIZE = 60
PUZZLE_IMAGE_PATH = "puzzle.png"
DISPLAY_MAX_W = 1000
DISPLAY_MAX_H = 700

# ================= IMAGE PROCESSING =================
def remove_white_background(image):
    """Adds an alpha channel and makes white-ish pixels transparent."""
    image_bgra = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
    b, g, r, a = cv2.split(image_bgra)
    white_mask = (b > 240) & (g > 240) & (r > 240)
    image_bgra[white_mask] = [0, 0, 0, 0]
    return image_bgra

def crop_to_content(image):
    """Crops to the bounding box of non-transparent pixels."""
    alpha = image[:, :, 3]
    coords = cv2.findNonZero(alpha)
    if coords is None:
        return image
    x, y, w, h = cv2.boundingRect(coords)
    return image[y:y+h, x:x+w]

def resize_for_display(img, max_w=DISPLAY_MAX_W, max_h=DISPLAY_MAX_H):
    h, w = img.shape[:2]
    scale = min(max_w / w, max_h / h, 1.0)
    return cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

# ================= MATCHING ENGINE =================
def find_top_matches(puzzle_img, piece_img, n_options=3):
    """
    Matches the piece at its REAL size inside the board image.
    TILE_SIZE is only used to suppress nearby duplicate results.
    """
    piece_bgr = piece_img[:, :, :3]
    mask = piece_img[:, :, 3] if piece_img.shape[2] == 4 else None
    ph, pw = piece_bgr.shape[:2]

    try:
        result = cv2.matchTemplate(puzzle_img, piece_bgr, cv2.TM_SQDIFF_NORMED, mask=mask)
    except cv2.error:
        res_sum = None
        for i in range(3):
            res_c = cv2.matchTemplate(puzzle_img[:,:,i], piece_bgr[:,:,i], cv2.TM_SQDIFF_NORMED, mask=mask)
            if res_sum is None:
                res_sum = res_c
            else:
                res_sum += res_c
        result = res_sum / 3.0

    matches = []
    result_copy = result.copy()
    suppress_r = max(TILE_SIZE // 2, pw // 2, ph // 2)

    for _ in range(n_options):
        min_val, _, min_loc, _ = cv2.minMaxLoc(result_copy)
        x, y = min_loc
        score = 1.0 - min_val
        matches.append((score, x, y, pw, ph))

        x1 = max(0, x - suppress_r)
        y1 = max(0, y - suppress_r)
        x2 = min(result.shape[1], x + suppress_r)
        y2 = min(result.shape[0], y + suppress_r)
        result_copy[y1:y2, x1:x2] = 1.0

    return matches

# ================= FULL-SCREEN ROI SELECTOR =================
class ROISelector:
    """
    Shows full-screen tkinter overlay for board selection.
    No window chrome, no titlebar => no coordinate shifts.
    """
    def __init__(self, parent, screen_img):
        self.win = tk.Toplevel(parent)
        self.win.title("")
        self.win.attributes("-fullscreen", True)
        self.win.attributes("-topmost", True)
        self.win.overrideredirect(True)

        rgb = cv2.cvtColor(screen_img, cv2.COLOR_BGR2RGB)
        self.tk_img = ImageTk.PhotoImage(Image.fromarray(rgb))

        self.canvas = tk.Canvas(self.win, cursor="cross", highlightthickness=0, bd=0)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.create_image(0, 0, anchor="nw", image=self.tk_img)

        self.canvas.create_text(
            10, 10,
            text="Drag a box over the PUZZLE BOARD  |  ENTER = confirm  |  ESC = cancel",
            fill="yellow", font=("Arial", 14, "bold"), anchor="nw"
        )

        self.rect = None
        self.start_x = self.start_y = None
        self.roi = None  # (x, y, w, h) in physical pixels

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
            outline="#00FF00", width=3
        )

    def on_drag(self, event):
        self.canvas.coords(self.rect, self.start_x, self.start_y, event.x, event.y)

    def on_release(self, event):
        x1 = min(self.start_x, event.x)
        y1 = min(self.start_y, event.y)
        x2 = max(self.start_x, event.x)
        y2 = max(self.start_y, event.y)
        self.roi = (x1, y1, x2 - x1, y2 - y1)
        self.canvas.delete("lbl")
        self.canvas.create_text(
            x1, max(y1 - 8, 20),
            text=f"{x2-x1} x {y2-y1} px  —  press ENTER to confirm",
            fill="#00FF00", font=("Arial", 12, "bold"), anchor="sw", tags="lbl"
        )

    def confirm(self, event=None):
        if self.roi and self.roi[2] > 0 and self.roi[3] > 0:
            self.win.destroy()

    def cancel(self):
        self.roi = None
        self.win.destroy()

# ================= SCREEN OVERLAY =================
class ScreenOverlay(tk.Toplevel):
    """
    Borderless transparent window that marks exactly where to place the piece.
    Width/height come from the actual matched piece size in pixels.
    """
    def __init__(self, parent, screen_x, screen_y, pw, ph, duration=7000):
        super().__init__(parent)
        self.overrideredirect(True)
        self.attributes("-topmost", True)
        self.attributes("-alpha", 0.85)
        self.config(bg="white")
        self.attributes("-transparentcolor", "white")

        self.geometry(f"{pw}x{ph}+{screen_x}+{screen_y}")

        c = tk.Canvas(self, width=pw, height=ph, bg="white", highlightthickness=0)
        c.pack()
        c.create_rectangle(4, 4, pw-4, ph-4, outline="#00FF00", width=6)
        if pw > 50 and ph > 20:
            c.create_text(pw//2, ph//2, text="HERE", fill="#00FF00",
                          font=("Arial", min(12, ph // 4), "bold"))

        self.after(duration, self.destroy)

# ================= MAIN GUI =================
class SolverPlusGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Puzzle Solver Pro")
        self.root.geometry("420x280")
        self.root.resizable(False, False)

        self.board_roi = None   # (screen_x, screen_y, screen_w, screen_h) — where the puzzle lives on screen
        self.puzzle_img = None  # The reference image from puzzle.png — used for matching

        tk.Label(root, text="🧩 Puzzle Solver", font=("Arial", 16, "bold"), pady=12).pack()

        self.status = tk.Label(root, text="Step 1: Set Board Area on screen", fg="#555", font=("Arial", 10))
        self.status.pack()

        btn_frame = tk.Frame(root)
        btn_frame.pack(pady=14, fill=tk.X, padx=40)

        tk.Button(btn_frame, text="🎯  Set Board Area",
                  command=self.calibrate, bg="#bbdefb",
                  font=("Arial", 12), height=2).pack(fill=tk.X, pady=4)

        tk.Button(btn_frame, text="📋  Paste Piece  (Ctrl+V)",
                  command=self.paste_image, bg="#c8e6c9",
                  font=("Arial", 12), height=2).pack(fill=tk.X, pady=4)

        tk.Label(root, text="Set board area once. Then paste pieces any time.",
                 font=("Arial", 9), fg="#999").pack()

        root.bind_all("<Control-v>", self.paste_image)

        # Always load puzzle.png as the reference image for matching
        if os.path.exists(PUZZLE_IMAGE_PATH):
            self.puzzle_img = cv2.imread(PUZZLE_IMAGE_PATH)
            img_h, img_w = self.puzzle_img.shape[:2]
            self.status.config(text=f"puzzle.png loaded ({img_w}×{img_h}) — now set board area", fg="green")
        else:
            self.status.config(text=f"ERROR: {PUZZLE_IMAGE_PATH} not found!", fg="red")

    # ---- CALIBRATE ----
    def calibrate(self):
        """Records WHERE the puzzle board is on screen. Does NOT touch puzzle.png."""
        self.root.withdraw()
        time.sleep(0.4)

        # Screenshot just to give the user a visible background to draw on
        screen = ImageGrab.grab()
        screen_bgr = cv2.cvtColor(np.array(screen), cv2.COLOR_RGB2BGR)

        selector = ROISelector(self.root, screen_bgr)
        self.root.wait_window(selector.win)

        roi = selector.roi
        self.root.deiconify()

        if roi:
            self.board_roi = roi  # Just save the screen location — puzzle.png stays untouched
            x, y, w, h = roi
            self.status.config(
                text=f"Board area set: {w}×{h} px at screen ({x},{y})", fg="green"
            )
            print(f"Board area on screen: {w}x{h} at ({x},{y})")
        else:
            self.status.config(text="Calibration cancelled", fg="orange")

    # ---- PASTE & MATCH ----
    def paste_image(self, event=None):
        if self.puzzle_img is None:
            messagebox.showwarning("No reference image",
                                   f"{PUZZLE_IMAGE_PATH} not found. Put your complete puzzle image there.")
            return

        try:
            img_pil = ImageGrab.grabclipboard()
            if img_pil is None:
                messagebox.showwarning("No image", "Copy a piece screenshot first.")
                return

            # Convert to BGR at natural size — no resize
            img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
            processed = remove_white_background(img_cv)
            processed = crop_to_content(processed)

            # Match against puzzle.png (the reference image)
            results = find_top_matches(self.puzzle_img, processed)
            score, match_x, match_y, piece_w, piece_h = results[0]

            img_h, img_w = self.puzzle_img.shape[:2]
            print(f"piece={piece_w}×{piece_h}  match=({match_x},{match_y}) in {img_w}×{img_h} reference  score={score:.3f}")

            # ---- OpenCV preview on the reference image ----
            preview = self.puzzle_img.copy()
            cv2.rectangle(preview, (match_x, match_y),
                          (match_x + piece_w, match_y + piece_h), (0, 255, 0), 3)
            cv2.putText(preview, f"{score:.1%}", (match_x, max(match_y - 8, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow("Detected Position (reference)", resize_for_display(preview))

            # ---- Screen overlay ----
            if self.board_roi:
                bx, by, bw, bh = self.board_roi

                # Scale match coords from puzzle.png image space → screen board space
                # because puzzle.png and the physical board on screen can be different sizes
                scale_x = bw / img_w
                scale_y = bh / img_h

                screen_x = int(bx + match_x * scale_x)
                screen_y = int(by  + match_y * scale_y)
                overlay_w = max(10, int(piece_w * scale_x))
                overlay_h = max(10, int(piece_h * scale_y))

                print(f"Overlay at screen ({screen_x},{screen_y}) size {overlay_w}×{overlay_h}")
                ScreenOverlay(self.root, screen_x, screen_y, overlay_w, overlay_h)
                self.status.config(text=f"Match: {score:.1%}  at image ({match_x},{match_y})", fg="green")
            else:
                self.status.config(text=f"Match: {score:.1%} — set board area for overlay", fg="orange")

        except Exception as e:
            messagebox.showerror("Error", str(e))
            raise

if __name__ == "__main__":
    root = tk.Tk()
    app = SolverPlusGUI(root)
    root.mainloop()
