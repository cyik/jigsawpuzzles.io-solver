import cv2
import numpy as np
import sys

# ---------- utils ----------

def resize_to_screen(img, max_w=1920, max_h=1080):
    h, w = img.shape[:2]
    scale = min(max_w / w, max_h / h, 1.0)
    return cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

def overlay_transparent(background, overlay, x, y):
    bh, bw = background.shape[:2]
    h, w = overlay.shape[:2]
    if x >= bw or y >= bh:
        return background
    if x + w > bw:
        w = bw - x
    if y + h > bh:
        h = bh - y
    if w <= 0 or h <= 0:
        return background
    overlay_crop = overlay[0:h, 0:w]
    if overlay.shape[2] < 4:
        overlay_bgr = overlay_crop
        mask = 255 * np.ones((h, w), dtype=np.uint8)
    else:
        overlay_bgr = overlay_crop[:, :, :3]
        mask = overlay_crop[:, :, 3]
    background_crop = background[y:y+h, x:x+w]
    mask_f = mask.astype(float) / 255.0
    mask_inv_f = 1.0 - mask_f
    for c in range(3):
        background_crop[:, :, c] = (
            mask_inv_f * background_crop[:, :, c] +
            mask_f * overlay_bgr[:, :, c]
        )
    background[y:y+h, x:x+w] = background_crop
    return background

# ---------- metrics ----------

def hist_score(patch_bgr, piece_bgr, mask=None):
    patch_hsv = cv2.cvtColor(patch_bgr.astype(np.uint8), cv2.COLOR_BGR2HSV)
    piece_hsv = cv2.cvtColor(piece_bgr.astype(np.uint8), cv2.COLOR_BGR2HSV)
    score_total = 0
    for ch in range(3):
        h_patch = cv2.calcHist([patch_hsv],[ch],None,[32],[0,256])
        h_piece = cv2.calcHist([piece_hsv],[ch],None,[32],[0,256])
        cv2.normalize(h_patch,h_patch,0,1,cv2.NORM_MINMAX)
        cv2.normalize(h_piece,h_piece,0,1,cv2.NORM_MINMAX)
        score_total += cv2.compareHist(h_patch,h_piece,cv2.HISTCMP_CORREL)
    return score_total / 3.0  # range [-1,1]

def edge_overlap_score(patch_gray, piece_gray, mask_uint8):
    p_edge = cv2.Canny(patch_gray, 80, 200)
    t_edge = cv2.Canny(piece_gray, 80, 200)
    mask_bool = (mask_uint8 > 0)
    t_edge_masked = (t_edge > 0) & mask_bool
    denom = t_edge_masked.sum()
    if denom == 0:
        return 0.0
    overlap = np.logical_and(t_edge_masked, p_edge > 0).sum()
    return float(overlap) / float(denom)

def orb_match_score(patch_gray, piece_gray):
    orb = cv2.ORB_create(300)
    kp1, des1 = orb.detectAndCompute(piece_gray, None)
    kp2, des2 = orb.detectAndCompute(patch_gray, None)
    if des1 is None or des2 is None:
        return 0.0
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    try:
        matches = bf.match(des1, des2)
    except:
        return 0.0
    good = [m for m in matches if m.distance < 60]
    denom = min(len(kp1), len(kp2))
    return float(len(good)) / float(denom) if denom > 0 else 0.0

def contour_score(patch_gray, piece_mask):
    if piece_mask is None:
        return 0.0
    contours, _ = cv2.findContours(piece_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0
    piece_contour = max(contours, key=cv2.contourArea)
    patch_edge = cv2.Canny(patch_gray, 80, 200)
    patch_contours, _ = cv2.findContours(patch_edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not patch_contours:
        return 0.0
    best = 1e9
    for pc in patch_contours:
        s = cv2.matchShapes(piece_contour, pc, cv2.CONTOURS_MATCH_I1, 0.0)
        best = min(best, s)
    return 1.0 / (1.0 + best)

# ---------- main search ----------

def find_candidates_color_first(puzzle_img, piece_img, grid_w=39, grid_h=26, n_options=3):
    p_h, p_w = puzzle_img.shape[:2]
    target_w = max(4, int(round(p_w / grid_w)))
    target_h = max(4, int(round(p_h / grid_h)))
    piece_resized = cv2.resize(piece_img, (target_w, target_h), interpolation=cv2.INTER_AREA)

    piece_bgr = piece_resized[:, :, :3]
    piece_gray = cv2.cvtColor(piece_bgr, cv2.COLOR_BGR2GRAY)
    piece_mask = piece_resized[:, :, 3] if piece_resized.shape[2] == 4 else None

    puzzle_gray = cv2.cvtColor(puzzle_img, cv2.COLOR_BGR2GRAY)

    candidates = []
    step_x = target_w // 2
    step_y = target_h // 2

    for y in range(0, p_h-target_h, step_y):
        for x in range(0, p_w-target_w, step_x):
            patch_bgr = puzzle_img[y:y+target_h, x:x+target_w]
            patch_gray = puzzle_gray[y:y+target_h, x:x+target_w]

            hscore = hist_score(patch_bgr, piece_bgr)
            escore = edge_overlap_score(patch_gray, piece_gray, piece_mask if piece_mask is not None else np.ones_like(piece_gray))
            oscore = orb_match_score(patch_gray, piece_gray)
            cscore = contour_score(patch_gray, piece_mask)

            # Color-first scoring
            final = 0.8*hscore + 0.15*escore + 0.05*oscore  # contour ignored for simplicity
            candidates.append((final, x, y, hscore, escore, oscore, cscore))

    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[:n_options], piece_resized

# ---------- run ----------

if __name__ == "__main__":
    args = sys.argv[1:]
    puzzle_path = args[0] if len(args) > 0 else "puzzle.png"
    piece_path = args[1] if len(args) > 1 else "piece.png"
    n_options = int(args[2]) if len(args) > 2 else 3

    puzzle_img = cv2.imread(puzzle_path)
    piece_img = cv2.imread(piece_path, cv2.IMREAD_UNCHANGED)

    if puzzle_img is None or piece_img is None:
        raise SystemExit("Could not load images!")

    results, piece_resized = find_candidates_color_first(puzzle_img, piece_img, n_options=n_options)

    puzzle_marked = puzzle_img.copy()
    for i, (score, x, y, hs, es, os, cs) in enumerate(results):
        w, h = piece_resized.shape[1], piece_resized.shape[0]
        cv2.rectangle(puzzle_marked, (x,y), (x+w,y+h), (0,255,0), 2)
        cv2.putText(puzzle_marked, f"{i+1}:{score:.3f}", (x, y-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
        overlayed = overlay_transparent(puzzle_img.copy(), piece_resized, x, y)
        cv2.imshow(f"Option {i+1} score={score:.3f}", resize_to_screen(overlayed))
        cv2.imwrite(f"option_{i+1}.png", overlayed)
        print(f"[Option {i+1}] score={score:.3f} (hist={hs:.2f}, edge={es:.2f}, orb={os:.2f}, contour={cs:.2f}) @ ({x},{y})")

    cv2.imshow("Candidates", resize_to_screen(puzzle_marked))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
