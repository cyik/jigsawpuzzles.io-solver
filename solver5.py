import cv2
import numpy as np
import tkinter as tk
from tkinter import messagebox
from PIL import ImageGrab

# ===== SETTINGS =====
TILE_SIZE = 60
N_OPTIONS = 3
PUZZLE_IMAGE_PATH = "puzzle.png"
DISPLAY_MAX_W = 1200
DISPLAY_MAX_H = 800

# ================= IMAGE PROCESSING =================
def remove_white_background(image):
    """
    Adds an alpha channel and sets white pixels to transparent.
    This helps in matching the piece's non-background content.
    """
    image_bgra = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
    b, g, r, a = cv2.split(image_bgra)
    # Define "white" as anything very light. 
    # For digital images, this is usually 255 but we allow a small margin.
    white_mask = (b > 235) & (g > 235) & (r > 235)
    image_bgra[white_mask] = [0, 0, 0, 0]
    return image_bgra

def crop_to_content(image):
    """
    Crops the piece image to the actual puzzle piece (non-transparent area).
    """
    alpha = image[:, :, 3]
    coords = cv2.findNonZero(alpha)
    if coords is None:
        return image
    x, y, w, h = cv2.boundingRect(coords)
    return image[y:y+h, x:x+w]

def resize_to_target(image, size):
    """
    Resizes the piece to the TILE_SIZE used for matching.
    """
    return cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)

def resize_for_display(img, max_w=DISPLAY_MAX_W, max_h=DISPLAY_MAX_H):
    h, w = img.shape[:2]
    scale = min(max_w / w, max_h / h, 1.0)
    return cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

# ================= ENHANCED MATCHING =================
def find_top_matches(puzzle_img, piece_img, n_options=N_OPTIONS):
    """
    Finds the top N matches for the piece in the puzzle board.
    Uses multi-channel SQDIFF_NORMED to respect color and avoid matching
    textured pieces to solid-color areas (like solid pink).
    """
    # piece_img is BGRA, puzzle_img is BGR
    piece_bgr = piece_img[:, :, :3]
    mask = piece_img[:, :, 3] if piece_img.shape[2] == 4 else None

    # THE FIX: 
    # 1. DO NOT convert to grayscale. Color information is crucial for digital puzzles.
    # 2. Use TM_SQDIFF_NORMED instead of TM_CCOEFF_NORMED.
    #    Correlation (CCOEFF) can produce false positives in solid areas by amplifying noise.
    #    Squared Difference (SQDIFF) measures the literal color distance, making it
    #    impossible for a non-pink piece to match a solid pink area with a high score.
    
    try:
        # SQDIFF_NORMED: 0.0 = perfect match, 1.0 = total mismatch
        # Many versions of OpenCV support mask + multi-channel here.
        result = cv2.matchTemplate(puzzle_img, piece_bgr, cv2.TM_SQDIFF_NORMED, mask=mask)
    except cv2.error:
        # Fallback: Process each channel separately if multi-channel mask isn't supported.
        res_sum = None
        for i in range(3):
            res_c = cv2.matchTemplate(puzzle_img[:,:,i], piece_bgr[:,:,i], cv2.TM_SQDIFF_NORMED, mask=mask)
            if res_sum is None: res_sum = res_c
            else: res_sum += res_c
        result = res_sum / 3.0

    matches = []
    result_copy = result.copy()
    
    for i in range(n_options):
        # We look for the MINIMUM value for SQDIFF
        min_val, _, min_loc, _ = cv2.minMaxLoc(result_copy)
        x, y = min_loc
        
        # Convert to a human-readable score where 1.0 is perfect and 0.0 is terrible
        score = 1.0 - min_val
        matches.append((score, x, y))
        
        # Suppress the area around the match so we find distinct options
        x1 = max(0, x - TILE_SIZE//2)
        y1 = max(0, y - TILE_SIZE//2)
        x2 = min(result.shape[1], x + TILE_SIZE//2)
        y2 = min(result.shape[0], y + TILE_SIZE//2)
        result_copy[y1:y2, x1:x2] = 1.0  # Set to max mismatch for next iteration
        
    return matches

# ================= GUI =================
class PasteSolverGUI:
    def __init__(self, root):
        self.root = root
        root.title("Puzzle Solver (Color Aware Mode)")
        root.geometry("400x150")
        
        self.label = tk.Label(root, text="Step 1: Copy a piece screenshot (Ctrl+C)\nStep 2: Press Ctrl+V here",
                             font=("Arial", 12), pady=20)
        self.label.pack()

        # Bind Ctrl+V
        self.root.bind_all("<Control-v>", self.paste_image)

        # Load board
        self.puzzle_img = cv2.imread(PUZZLE_IMAGE_PATH)
        if self.puzzle_img is None:
            messagebox.showerror("Error", f"Could not find {PUZZLE_IMAGE_PATH}.\nMake sure it's in the same folder!")
            root.destroy()
            return
            
        print(f"Board loaded: {self.puzzle_img.shape[1]}x{self.puzzle_img.shape[0]}")

    def paste_image(self, event=None):
        try:
            # Clear old windows
            cv2.destroyAllWindows()

            # Grab from clipboard
            img_pil = ImageGrab.grabclipboard()
            if img_pil is None:
                messagebox.showwarning("Clipboard Empty", "No image found in clipboard. Please copy a puzzle piece first!")
                return

            # Prepare image
            img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
            
            # Processing pipeline
            processed = remove_white_background(img_cv)
            processed = crop_to_content(processed)
            processed = resize_to_target(processed, TILE_SIZE)

            # Find matches
            results = find_top_matches(self.puzzle_img, processed)

            # Draw
            board_display = self.puzzle_img.copy()
            for i, (score, x, y) in enumerate(results):
                color = (0, 255, 0) if i == 0 else (255, 100, 0) # Green for #1, Blue-ish for others
                cv2.rectangle(board_display, (x, y), (x+TILE_SIZE, y+TILE_SIZE), color, 3)
                label = f"#{i+1} ({score:.2%})"
                cv2.putText(board_display, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                print(f"[Match {i+1}] Score: {score:.4f} at ({x}, {y})")

            # Show results
            cv2.imshow("Puzzle Matching Results", resize_for_display(board_display))

        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")

# ================= RUN =================
if __name__ == "__main__":
    root = tk.Tk()
    app = PasteSolverGUI(root)
    root.mainloop()