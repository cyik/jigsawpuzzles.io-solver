"""
solver7.py - Maximum-accuracy puzzle piece locator

Matching pipeline:
  1. SIFT feature matching     — scale-invariant; finds structural keypoints
                                  and matches them regardless of zoom level.
  2. Multi-scale template ensemble — SQDIFF + CCOEFF on BGR + LAB, tried at
                                  9 different scales. Fallback when SIFT has
                                  too few keypoints (e.g. solid-colour pieces).
  3. Histogram verification    — independently scores every candidate by
                                  comparing the piece's colour distribution
                                  against the matched region.  A wrong match
                                  will almost always fail this check.
  4. Consensus ranking         — all candidates from all methods are combined,
                                  deduplicated, and the highest overall score
                                  wins.
"""

import ctypes
try:
    ctypes.windll.user32.SetProcessDPIAware()
except Exception:
    pass

import cv2
import numpy as np
import tkinter as tk
from tkinter import messagebox
from PIL import ImageGrab, ImageTk, Image
import os
import time

# ===== SETTINGS =====
TILE_SIZE = 60          # Used only for overlap-suppression radius
PUZZLE_IMAGE_PATH = "puzzle.png"
N_OPTIONS = 3
DISPLAY_MAX_W = 1000
DISPLAY_MAX_H = 700

# ============================================================
# PRE-PROCESSING
# ============================================================
def remove_white_background(image):
    """Makes white / near-white pixels transparent (alpha = 0)."""
    bgra = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
    b, g, r, a = cv2.split(bgra)
    mask = (b > 240) & (g > 240) & (r > 240)
    bgra[mask] = [0, 0, 0, 0]
    return bgra

def crop_to_content(image):
    """Crops to the smallest bounding box that contains opaque pixels."""
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

# ============================================================
# METHOD 1 — SIFT FEATURE MATCHING
# ============================================================
def sift_find(puzzle_img, piece_bgr):
    """
    SIFT-based keypoint matching.

    Why SIFT beats template matching for this task:
      • Completely scale-invariant  — handles zoom differences between the
        piece screenshot and puzzle.png without trying dozens of scales.
      • Brightness-invariant        — descriptors encode local *gradients*,
        not raw pixel values, so lighting differences don't hurt.
      • Structurally specific       — matches corners/edges, not large flat
        colour regions, so it never gets confused by solid-colour areas.

    Returns: list of (score, x, y, matched_w, matched_h)  or  []
    """
    try:
        sift = cv2.SIFT_create(
            nfeatures=0,
            contrastThreshold=0.02,   # low threshold → finds features in subtle art
            edgeThreshold=10,
            sigma=1.6
        )
    except AttributeError:
        print("SIFT unavailable — upgrade opencv-python to 4.4+")
        return []

    # CLAHE enhances local contrast so SIFT finds more keypoints in
    # areas that would otherwise look flat.
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    puzzle_gray = clahe.apply(cv2.cvtColor(puzzle_img, cv2.COLOR_BGR2GRAY))
    piece_gray  = clahe.apply(cv2.cvtColor(piece_bgr,  cv2.COLOR_BGR2GRAY))

    kp_puz, des_puz = sift.detectAndCompute(puzzle_gray, None)
    kp_pc,  des_pc  = sift.detectAndCompute(piece_gray,  None)

    if des_pc is None or des_puz is None or len(des_pc) < 5 or len(des_puz) < 5:
        print(f"SIFT: too few keypoints (piece={len(des_pc) if des_pc is not None else 0})")
        return []

    # FLANN matcher — fast approximate nearest-neighbour search
    FLANN_KDTREE = 1
    flann = cv2.FlannBasedMatcher(
        dict(algorithm=FLANN_KDTREE, trees=5),
        dict(checks=100)
    )
    raw_matches = flann.knnMatch(des_pc, des_puz, k=2)

    # Lowe's ratio test — keep only unambiguous matches
    good = [m for pair in raw_matches
            if len(pair) == 2
            for m, n in [pair]
            if m.distance < 0.73 * n.distance]

    print(f"SIFT: {len(good)} good matches out of {len(raw_matches)}")

    if len(good) < 8:
        print("SIFT: not enough good matches — will use template fallback")
        return []

    src_pts = np.float32([kp_pc [m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp_puz[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    M, inlier_mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    if M is None or inlier_mask is None:
        print("SIFT: homography failed")
        return []

    inlier_ratio = float(np.sum(inlier_mask)) / len(inlier_mask)
    print(f"SIFT: inlier ratio = {inlier_ratio:.2f}")

    if inlier_ratio < 0.25:
        print("SIFT: homography unreliable (too many outliers)")
        return []

    # Project the four corners of the piece through the homography
    ph, pw = piece_bgr.shape[:2]
    corners = np.float32([[0, 0], [pw, 0], [pw, ph], [0, ph]]).reshape(-1, 1, 2)
    trans   = cv2.perspectiveTransform(corners, M)

    xs = trans[:, 0, 0]
    ys = trans[:, 0, 1]
    x  = int(np.min(xs));  y  = int(np.min(ys))
    rw = int(np.max(xs) - np.min(xs))
    rh = int(np.max(ys) - np.min(ys))

    # Clamp to image bounds
    img_h, img_w = puzzle_img.shape[:2]
    x  = max(0, min(x,  img_w - 2))
    y  = max(0, min(y,  img_h - 2))
    rw = max(8, min(rw, img_w - x))
    rh = max(8, min(rh, img_h - y))

    # Score: inlier_ratio * log-scaled match count (more matches = more confident)
    score = inlier_ratio * min(1.0, len(good) / 20.0)
    return [(score, x, y, rw, rh)]


# ============================================================
# METHOD 2 — MULTI-SCALE TEMPLATE MATCHING ENSEMBLE
# ============================================================
def template_find(puzzle_img, piece_bgr, mask, n_options):
    """
    Tries the piece at 9 different scales.
    At each scale, runs 4 matching methods:
      • SQDIFF_NORMED  on BGR  — exact pixel colour distance
      • CCOEFF_NORMED  on BGR  — structural/gradient similarity
      • SQDIFF_NORMED  on LAB  — perceptually uniform colour distance
      • CCOEFF_NORMED  on LAB  — perceptual + structural
    Results are weighted and combined into a single score map per scale.
    The scale that produces the best peak is used for final extraction.
    """
    img_h, img_w = puzzle_img.shape[:2]
    ph, pw = piece_bgr.shape[:2]
    puzzle_lab = cv2.cvtColor(puzzle_img, cv2.COLOR_BGR2LAB)

    scales = [0.80, 0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15, 1.20]

    best_map  = None
    best_peak = -1.0
    best_pw   = pw
    best_ph   = ph

    for scale in scales:
        nw = max(4, int(pw * scale))
        nh = max(4, int(ph * scale))
        if nw >= img_w or nh >= img_h:
            continue

        interp   = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
        sp       = cv2.resize(piece_bgr, (nw, nh), interpolation=interp)
        sp_lab   = cv2.cvtColor(sp, cv2.COLOR_BGR2LAB)

        sm = None
        if mask is not None:
            sm = cv2.resize(mask, (nw, nh), interpolation=interp)
            _, sm = cv2.threshold(sm, 127, 255, cv2.THRESH_BINARY)

        # (method_flag, reference_image, template, invert_result, weight)
        ops = [
            (cv2.TM_SQDIFF_NORMED,  puzzle_img, sp,     True,  0.35),
            (cv2.TM_CCOEFF_NORMED,  puzzle_img, sp,     False, 0.20),
            (cv2.TM_SQDIFF_NORMED,  puzzle_lab, sp_lab, True,  0.30),
            (cv2.TM_CCOEFF_NORMED,  puzzle_lab, sp_lab, False, 0.15),
        ]

        combined     = None
        total_weight = 0.0

        for method, ref, tmpl, invert, w in ops:
            try:
                r = cv2.matchTemplate(ref, tmpl, method, mask=sm)
                r = (1.0 - r) if invert else np.clip(r, 0.0, 1.0)
                combined     = r * w if combined is None else combined + r * w
                total_weight += w
            except cv2.error:
                pass

        if combined is None or total_weight == 0:
            continue

        combined /= total_weight   # normalise so weights always sum to 1

        _, peak, _, _ = cv2.minMaxLoc(combined)
        if peak > best_peak:
            best_peak = peak
            best_map  = combined.copy()
            best_pw   = nw
            best_ph   = nh

    if best_map is None:
        return []

    print(f"Template: best peak={best_peak:.4f}  piece size used={best_pw}×{best_ph}")

    matches    = []
    result_cp  = best_map.copy()
    suppress_r = max(TILE_SIZE // 2, best_pw // 2, best_ph // 2)

    for _ in range(n_options):
        _, mv, _, ml = cv2.minMaxLoc(result_cp)
        x, y = ml
        matches.append((float(mv), x, y, best_pw, best_ph))

        x1 = max(0, x - suppress_r);  y1 = max(0, y - suppress_r)
        x2 = min(result_cp.shape[1], x + suppress_r)
        y2 = min(result_cp.shape[0], y + suppress_r)
        result_cp[y1:y2, x1:x2] = 0.0

    return matches


# ============================================================
# HISTOGRAM VERIFICATION
# ============================================================
def histogram_score(puzzle_img, piece_bgr, x, y, pw, ph):
    """
    Compares the HSV colour histogram of the piece with the puzzle region
    at (x, y, pw, ph).

    This is completely independent of how the match was found — it simply
    asks 'does this region contain the same colours as the piece?'

    Returns a score in [0, 1]:  1.0 = perfect colour match.
    """
    img_h, img_w = puzzle_img.shape[:2]
    x  = max(0, min(x,  img_w - 1))
    y  = max(0, min(y,  img_h - 1))
    pw = max(1, min(pw, img_w - x))
    ph = max(1, min(ph, img_h - y))

    region = puzzle_img[y:y+ph, x:x+pw]
    if region.size == 0:
        return 0.0

    piece_rs = cv2.resize(piece_bgr, (region.shape[1], region.shape[0]))

    piece_hsv  = cv2.cvtColor(piece_rs, cv2.COLOR_BGR2HSV)
    region_hsv = cv2.cvtColor(region,   cv2.COLOR_BGR2HSV)

    # Compare each channel; hue gets more bins because it carries the most info
    configs = [(0, 36, [0, 180]),   # hue
               (1, 16, [0, 256]),   # saturation
               (2, 16, [0, 256])]   # value

    total_corr = 0.0
    for ch, bins, rng in configs:
        hp = cv2.calcHist([piece_hsv],  [ch], None, [bins], rng)
        hr = cv2.calcHist([region_hsv], [ch], None, [bins], rng)
        cv2.normalize(hp, hp)
        cv2.normalize(hr, hr)
        corr = cv2.compareHist(hp, hr, cv2.HISTCMP_CORREL)
        total_corr += max(0.0, corr)   # clip negatives: anti-correlations = 0

    return total_corr / len(configs)


# ============================================================
# MASTER MATCHING FUNCTION
# ============================================================
def find_top_matches(puzzle_img, piece_img, n_options=N_OPTIONS):
    """
    Runs all methods, combines their candidates using histogram verification,
    deduplicates overlapping results, and returns the top n_options matches.

    Each returned tuple: (final_score, x, y, piece_w, piece_h)
    """
    piece_bgr = piece_img[:, :, :3]
    mask      = piece_img[:, :, 3] if piece_img.shape[2] == 4 else None

    sift_candidates     = sift_find(puzzle_img, piece_bgr)
    template_candidates = template_find(puzzle_img, piece_bgr, mask, n_options * 2)

    # Label for logging
    labeled = [(s, x, y, pw, ph, "SIFT")     for s, x, y, pw, ph in sift_candidates] + \
              [(s, x, y, pw, ph, "TEMPLATE") for s, x, y, pw, ph in template_candidates]

    if not labeled:
        raise RuntimeError("All matching methods failed — no candidates found.")

    # Score every candidate: 60 % match score + 40 % histogram agreement
    scored = []
    for raw_score, x, y, pw, ph, method in labeled:
        h = histogram_score(puzzle_img, piece_bgr, x, y, pw, ph)
        final = 0.60 * raw_score + 0.40 * h
        scored.append((final, x, y, pw, ph, method, h))
        print(f"  [{method}] raw={raw_score:.3f}  hist={h:.3f}  final={final:.3f}  @ ({x},{y})")

    scored.sort(key=lambda c: c[0], reverse=True)

    # Pick top N without significant overlap
    selected = []
    for final, x, y, pw, ph, method, h in scored:
        overlaps = any(
            abs(x - sx) < (pw + spw) // 2 and abs(y - sy) < (ph + sph) // 2
            for _, sx, sy, spw, sph in selected
        )
        if not overlaps:
            selected.append((final, x, y, pw, ph))
            print(f"  → SELECTED via {method}: {final:.3f} @ ({x},{y})")
            if len(selected) >= n_options:
                break

    return selected if selected else [(0.0, 0, 0, 60, 60)]


# ============================================================
# FULL-SCREEN ROI SELECTOR  (no window chrome = no offset)
# ============================================================
class ROISelector:
    def __init__(self, parent, screen_img):
        self.win = tk.Toplevel(parent)
        self.win.title("")
        self.win.attributes("-fullscreen", True)
        self.win.attributes("-topmost", True)
        self.win.overrideredirect(True)

        rgb = cv2.cvtColor(screen_img, cv2.COLOR_BGR2RGB)
        self.tk_img = ImageTk.PhotoImage(Image.fromarray(rgb))

        self.canvas = tk.Canvas(self.win, cursor="cross",
                                highlightthickness=0, bd=0)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.create_image(0, 0, anchor="nw", image=self.tk_img)
        self.canvas.create_text(
            10, 10,
            text="Drag a box around the PUZZLE BOARD  |  ENTER = confirm  |  ESC = cancel",
            fill="yellow", font=("Arial", 14, "bold"), anchor="nw"
        )

        self.rect = self.start_x = self.start_y = None
        self.roi  = None

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
            event.x, event.y, event.x, event.y,
            outline="#00FF00", width=3
        )

    def on_drag(self, event):
        self.canvas.coords(self.rect, self.start_x, self.start_y,
                           event.x, event.y)

    def on_release(self, event):
        x1 = min(self.start_x, event.x);  y1 = min(self.start_y, event.y)
        x2 = max(self.start_x, event.x);  y2 = max(self.start_y, event.y)
        self.roi = (x1, y1, x2 - x1, y2 - y1)
        self.canvas.delete("lbl")
        self.canvas.create_text(
            x1, max(y1 - 8, 20),
            text=f"{x2-x1}×{y2-y1} px  —  press ENTER to confirm",
            fill="#00FF00", font=("Arial", 12, "bold"), anchor="sw", tags="lbl"
        )

    def confirm(self, event=None):
        if self.roi and self.roi[2] > 0 and self.roi[3] > 0:
            self.win.destroy()

    def cancel(self):
        self.roi = None
        self.win.destroy()


# ============================================================
# SCREEN OVERLAY
# ============================================================
class ScreenOverlay(tk.Toplevel):
    """Borderless transparent window drawn at the exact board location."""

    def __init__(self, parent, screen_x, screen_y, pw, ph, duration=7000):
        super().__init__(parent)
        self.overrideredirect(True)
        self.attributes("-topmost", True)
        self.attributes("-alpha", 0.85)
        self.config(bg="white")
        self.attributes("-transparentcolor", "white")
        self.geometry(f"{pw}x{ph}+{screen_x}+{screen_y}")

        c = tk.Canvas(self, width=pw, height=ph,
                      bg="white", highlightthickness=0)
        c.pack()
        c.create_rectangle(4, 4, pw-4, ph-4, outline="#00FF00", width=6)
        if pw > 40 and ph > 20:
            c.create_text(pw//2, ph//2, text="HERE",
                          fill="#00FF00",
                          font=("Arial", max(9, min(14, ph // 4)), "bold"))

        self.after(duration, self.destroy)


# ============================================================
# MAIN GUI
# ============================================================
class SolverGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Puzzle Solver 7  —  Max Accuracy")
        self.root.geometry("440x320")
        self.root.resizable(False, False)

        self.board_roi  = None   # (screen_x, screen_y, screen_w, screen_h)
        self.puzzle_img = None   # Reference image from puzzle.png

        # Title
        tk.Label(root, text="🧩 Puzzle Solver 7",
                 font=("Arial", 16, "bold"), pady=10).pack()

        self.status = tk.Label(root, text="Loading…", fg="#555",
                               font=("Arial", 10), wraplength=400)
        self.status.pack()

        # Confidence bar frame
        bar_frame = tk.Frame(root)
        bar_frame.pack(pady=4)
        tk.Label(bar_frame, text="Confidence:", font=("Arial", 9)).pack(side=tk.LEFT)
        self.conf_bar = tk.Canvas(bar_frame, width=200, height=14,
                                  bg="#ddd", highlightthickness=1,
                                  highlightbackground="#aaa")
        self.conf_bar.pack(side=tk.LEFT, padx=6)
        self.conf_fill = self.conf_bar.create_rectangle(
            0, 0, 0, 14, fill="#4CAF50", width=0
        )

        # Buttons
        btn = tk.Frame(root)
        btn.pack(pady=10, fill=tk.X, padx=40)
        tk.Button(btn, text="🎯  Set Board Area",
                  command=self.calibrate, bg="#bbdefb",
                  font=("Arial", 12), height=2).pack(fill=tk.X, pady=4)
        tk.Button(btn, text="📋  Paste Piece  (Ctrl+V)",
                  command=self.paste_image, bg="#c8e6c9",
                  font=("Arial", 12), height=2).pack(fill=tk.X, pady=4)

        tk.Label(root, text="Set board area once. Then paste pieces any time.",
                 font=("Arial", 9), fg="#999").pack()

        root.bind_all("<Control-v>", self.paste_image)

        # Load reference puzzle
        if os.path.exists(PUZZLE_IMAGE_PATH):
            self.puzzle_img = cv2.imread(PUZZLE_IMAGE_PATH)
            h, w = self.puzzle_img.shape[:2]
            self.status.config(
                text=f"puzzle.png loaded ({w}×{h}) — set board area, then paste a piece",
                fg="green"
            )
        else:
            self.status.config(
                text=f"❌  {PUZZLE_IMAGE_PATH} not found! Put the complete puzzle image there.",
                fg="red"
            )

    # ---- CALIBRATE ----
    def calibrate(self):
        """Records the on-screen region of the puzzle board. Does NOT touch puzzle.png."""
        self.root.withdraw()
        time.sleep(0.4)

        screen     = ImageGrab.grab()
        screen_bgr = cv2.cvtColor(np.array(screen), cv2.COLOR_RGB2BGR)

        selector = ROISelector(self.root, screen_bgr)
        self.root.wait_window(selector.win)
        roi = selector.roi
        self.root.deiconify()

        if roi:
            self.board_roi = roi
            x, y, w, h = roi
            self.status.config(
                text=f"Board synced: {w}×{h} px at screen ({x},{y})", fg="green"
            )
            print(f"Board area: {w}×{h} at ({x},{y})")
        else:
            self.status.config(text="Calibration cancelled", fg="orange")

    # ---- UPDATE CONFIDENCE BAR ----
    def _set_confidence(self, score):
        fill_w = int(score * 200)
        color  = "#4CAF50" if score > 0.7 else "#FF9800" if score > 0.45 else "#F44336"
        self.conf_bar.coords(self.conf_fill, 0, 0, fill_w, 14)
        self.conf_bar.itemconfig(self.conf_fill, fill=color)

    # ---- PASTE & MATCH ----
    def paste_image(self, event=None):
        if self.puzzle_img is None:
            messagebox.showwarning(
                "No reference",
                f"{PUZZLE_IMAGE_PATH} not found.\n"
                "Put your complete puzzle image there and restart."
            )
            return

        try:
            img_pil = ImageGrab.grabclipboard()
            if img_pil is None:
                messagebox.showwarning("Clipboard empty",
                                       "Copy a puzzle piece screenshot first.")
                return

            self.status.config(text="Matching… (SIFT + template + histogram)", fg="blue")
            self.root.update()
            cv2.destroyAllWindows()

            img_cv    = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
            processed = remove_white_background(img_cv)
            processed = crop_to_content(processed)

            results = find_top_matches(self.puzzle_img, processed)
            top_score, match_x, match_y, piece_w, piece_h = results[0]

            img_h, img_w = self.puzzle_img.shape[:2]
            print(f"\n=== TOP RESULT: score={top_score:.3f} @ ({match_x},{match_y}) "
                  f"size={piece_w}×{piece_h} ===\n")

            # ---- OpenCV preview ----
            preview = self.puzzle_img.copy()
            for i, (sc, mx, my, pw, ph) in enumerate(results):
                color = (0, 220, 0) if i == 0 else (255, 140, 0)
                thick = 3 if i == 0 else 2
                cv2.rectangle(preview, (mx, my), (mx+pw, my+ph), color, thick)
                cv2.putText(preview, f"#{i+1} {sc:.0%}",
                            (mx, max(my-8, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.imshow("Best Matches (reference image)", resize_for_display(preview))

            # ---- Confidence bar ----
            self._set_confidence(top_score)

            # ---- Screen overlay ----
            if self.board_roi:
                bx, by, bw, bh = self.board_roi
                sx = bw / img_w   # scale from puzzle.png → screen
                sy = bh / img_h

                scr_x = int(bx + match_x * sx)
                scr_y = int(by + match_y * sy)
                ov_w  = max(10, int(piece_w * sx))
                ov_h  = max(10, int(piece_h * sy))

                ScreenOverlay(self.root, scr_x, scr_y, ov_w, ov_h)
                self.status.config(
                    text=f"Match: {top_score:.0%}  —  board pos ({match_x},{match_y})",
                    fg="green" if top_score > 0.65 else "orange"
                )
            else:
                self.status.config(
                    text=f"Match: {top_score:.0%}  —  set board area for screen overlay",
                    fg="orange"
                )

        except Exception as e:
            messagebox.showerror("Error", str(e))
            self.status.config(text=f"Error: {e}", fg="red")
            raise


# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == "__main__":
    root = tk.Tk()
    app  = SolverGUI(root)
    root.mainloop()
