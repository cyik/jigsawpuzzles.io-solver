import cv2
import numpy as np
import tkinter as tk
from tkinter import messagebox
from PIL import ImageGrab, Image, ImageTk

# ===== SETTINGS =====
TILE_SIZE = 115
N_OPTIONS = 3
PUZZLE_IMAGE_PATH = "puzzle.png"  # your puzzle board image
DISPLAY_MAX_W = 1200  # max window width for preview
DISPLAY_MAX_H = 800   # max window height for preview

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
#    return background#

def find_top_matches(puzzle_img, piece_img, n_options=N_OPTIONS):
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
        # suppress nearby so the same area isn’t picked again
        x1 = max(0, x - TILE_SIZE//2)
        y1 = max(0, y - TILE_SIZE//2)
        x2 = min(result.shape[1], x + TILE_SIZE//2)
        y2 = min(result.shape[0], y + TILE_SIZE//2)
        result_copy[y1:y2, x1:x2] = -1
    return matches

# ================= GUI =================
class PasteSolverGUI:
    def __init__(self, root):
        self.root = root
        root.title("Puzzle Solver (Paste Screenshot)")
        self.label = tk.Label(root, text="Paste your puzzle piece screenshot (Ctrl+V)")
        self.label.pack(pady=5)

        self.canvas = tk.Canvas(root, width=TILE_SIZE*2, height=TILE_SIZE*2)
        self.canvas.pack(pady=10)

        # Bind Ctrl+V for pasting
        self.root.bind_all("<Control-v>", self.paste_image)

        # Load puzzle image
        self.puzzle_img = cv2.imread(PUZZLE_IMAGE_PATH)
        if self.puzzle_img is None:
            messagebox.showerror("Error", f"Cannot load {PUZZLE_IMAGE_PATH}")
            root.destroy()
            return

        self.tk_image = None

    def paste_image(self, event=None):
        try:
            # ===== CLOSE PREVIOUS WINDOWS =====
            cv2.destroyAllWindows()

            # Grab image from clipboard
            img = ImageGrab.grabclipboard()
            if img is None:
                messagebox.showwarning("Warning", "No image in clipboard!")
                return

            # Convert PIL -> OpenCV
            img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

            # ===== PROCESS PIECE =====
            processed = remove_white_background(img_cv)
            processed = crop_to_content(processed)
            processed = resize_to_target(processed, TILE_SIZE)

            # ===== RUN SOLVER =====
            results = find_top_matches(self.puzzle_img, processed)

            # ===== DRAW RESULTS =====
            puzzle_copy = self.puzzle_img.copy()
            for i, (score, x, y) in enumerate(results):
                cv2.rectangle(puzzle_copy, (x, y), (x+TILE_SIZE, y+TILE_SIZE), (0,255,0), 2)
                cv2.putText(puzzle_copy, f"{i+1}:{score:.4f}", (x, y-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
                overlay = overlay_transparent(self.puzzle_img.copy(), processed, x, y)
                #cv2.imshow(f"Option {i+1} score={score:.4f}", resize_for_display(overlay))
                print(f"[Option {i+1}] score={score:.5f} @ ({x},{y})")

            cv2.imshow("Top Matches", resize_for_display(puzzle_copy))

        except Exception as e:
            messagebox.showerror("Error", str(e))

# ================= RUN =================
if __name__ == "__main__":
    root = tk.Tk()
    gui = PasteSolverGUI(root)
    root.mainloop()
