import cv2
import numpy as np
import sys
from math import floor

# ---------- utils ----------

def resize_to_screen(img, max_w=1920, max_h=1080):
    h, w = img.shape[:2]
    scale = min(max_w / w, max_h / h, 1.0)
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

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

# ---------- matching building blocks ----------

def masked_norm_corr_single(A, B, mask_bool):
    n = mask_bool.sum()
    if n == 0:
        return 0.0
    a = A[mask_bool].astype(np.float64)
    b = B[mask_bool].astype(np.float64)
    a_mean = a.mean()
    b_mean = b.mean()
    a_cent = a - a_mean
    b_cent = b - b_mean
    num = np.sum(a_cent * b_cent)
    denom = np.sqrt(np.sum(a_cent * a_cent) * np.sum(b_cent * b_cent))
    if denom == 0:
        return 0.0
    return float(num / denom)

def color_masked_score(patch_bgr, piece_bgr, mask_uint8):
    mask_bool = (mask_uint8 > 0)
    scores = []
    for c in range(3):
        s = masked_norm_corr_single(patch_bgr[:, :, c], piece_bgr[:, :, c], mask_bool)
        scores.append(s)
    scores = [(s + 1.0) / 2.0 for s in scores]  # map [-1,1] -> [0,1]
    return sum(scores) / 3.0

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
    orb = cv2.ORB_create(500)
    kp1, des1 = orb.detectAndCompute(piece_gray, None)
    kp2, des2 = orb.detectAndCompute(patch_gray, None)
    if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
        return 0.0
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    try:
        matches = bf.match(des1, des2)
    except Exception:
        return 0.0
    if not matches:
        return 0.0
    matches = sorted(matches, key=lambda x: x.distance)
    good = [m for m in matches if m.distance < 60]
    denom = min(len(kp1), len(kp2))
    return float(len(good)) / float(denom) if denom > 0 else 0.0

# ---------- grid-aware multi-candidate search (color-focused) ----------

def find_top_candidates_color(puzzle_img, piece_img, grid_w=39, grid_h=26, n_options=3, top_k_orb=10):
    p_h, p_w = puzzle_img.shape[:2]
    target_w = max(4, int(round(p_w / grid_w)))
    target_h = max(4, int(round(p_h / grid_h)))
    print(f"[grid] target size = {target_w} x {target_h} (W x H)")

    piece_resized = cv2.resize(piece_img, (target_w, target_h), interpolation=cv2.INTER_AREA)
    piece_bgr = piece_resized[:, :, :3].astype(np.float32)
    piece_gray = cv2.cvtColor(piece_bgr.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    mask = piece_resized[:, :, 3] if piece_resized.shape[2] == 4 else np.ones((target_h, target_w), dtype=np.uint8) * 255

    puzzle_gray = cv2.cvtColor(puzzle_img, cv2.COLOR_BGR2GRAY)
    puzzle_bgr = puzzle_img.astype(np.float32)

    candidates = []
    cell_w_f = p_w / grid_w
    cell_h_f = p_h / grid_h

    # evaluate grid centers
    for gy in range(grid_h):
        for gx in range(grid_w):
            center_x = int(round(gx * cell_w_f + cell_w_f / 2.0))
            center_y = int(round(gy * cell_h_f + cell_h_f / 2.0))
            tlx = max(0, min(center_x - target_w // 2, p_w - target_w))
            tly = max(0, min(center_y - target_h // 2, p_h - target_h))
            patch = puzzle_img[tly:tly + target_h, tlx:tlx + target_w]
            patch_bgr = patch.astype(np.float32)
            patch_gray = puzzle_gray[tly:tly + target_h, tlx:tlx + target_w]

            # Color-focused scoring
            color_score = color_masked_score(patch_bgr, piece_bgr, mask)
            edge_score = edge_overlap_score(patch_gray, piece_gray, mask)
            quick_combined = 0.85 * color_score + 0.15 * edge_score  # shift more weight to color
            candidates.append((quick_combined, tlx, tly, color_score, edge_score))

    candidates.sort(key=lambda x: x[0], reverse=True)

    # ORB scoring for top few
    top_k = min(max(n_options, top_k_orb), len(candidates))
    final_list = []
    for i in range(top_k):
        quick, tx, ty, cscore, escore = candidates[i]
        patch_gray = puzzle_gray[ty:ty + target_h, tx:tx + target_w]
        orb_score = orb_match_score(patch_gray, piece_gray)
        final_score = 0.7 * cscore + 0.2 * escore + 0.1 * orb_score  # color dominates
        final_list.append((final_score, tx, ty, cscore, escore, orb_score))

    final_list.sort(key=lambda x: x[0], reverse=True)
    selected = final_list[:n_options]

    # local refinement
    results = []
    refine_radius = 4
    for final_score, tx, ty, cscore, escore, orb_score in selected:
        best_local = (tx, ty, final_score)
        for dy in range(-refine_radius, refine_radius + 1):
            for dx in range(-refine_radius, refine_radius + 1):
                rx = tx + dx
                ry = ty + dy
                if rx < 0 or ry < 0 or rx + target_w > p_w or ry + target_h > p_h:
                    continue
                patch_bgr = puzzle_img[ry:ry + target_h, rx:rx + target_w].astype(np.float32)
                patch_gray = puzzle_gray[ry:ry + target_h, rx:rx + target_w]
                c = color_masked_score(patch_bgr, piece_bgr, mask)
                e = edge_overlap_score(patch_gray, piece_gray, mask)
                s = 0.7 * c + 0.2 * e
                if s > best_local[2]:
                    best_local = (rx, ry, s)
        rx, ry, refined_score = best_local
        patch_gray_ref = puzzle_gray[ry:ry + target_h, rx:rx + target_w]
        refined_orb = orb_match_score(patch_gray_ref, piece_gray)
        c_final = color_masked_score(puzzle_img[ry:ry + target_h, rx:rx + target_w].astype(np.float32), piece_bgr, mask)
        e_final = edge_overlap_score(patch_gray_ref, piece_gray, mask)
        refined_final = 0.7 * c_final + 0.2 * e_final + 0.1 * refined_orb

        results.append({
            "x": rx, "y": ry,
            "score": refined_final,
            "color_score": c_final,
            "edge_score": e_final,
            "orb_score": refined_orb
        })

    results.sort(key=lambda r: r["score"], reverse=True)
    return results, piece_resized

# ---------- main ----------

if __name__ == "__main__":
    args = sys.argv[1:]
    puzzle_path = args[0] if len(args) >= 1 else "puzzle.png"
    piece_path = args[1] if len(args) >= 2 else "piece.png"
    try:
        n_options = int(args[2]) if len(args) >= 3 else 3
    except Exception:
        n_options = 3

    print(f"Loading '{puzzle_path}' and '{piece_path}', top {n_options} options...")

    puzzle_img = cv2.imread(puzzle_path)
    piece_img = cv2.imread(piece_path, cv2.IMREAD_UNCHANGED)

    if puzzle_img is None or piece_img is None:
        raise SystemExit("Could not load images!")

    results, piece_resized = find_top_candidates_color(puzzle_img, piece_img, grid_w=39, grid_h=26, n_options=n_options, top_k_orb=12)

    # display candidates
    palette = [(0,255,0), (255,0,0), (0,0,255), (0,255,255), (255,0,255)]
    puzzle_marked = puzzle_img.copy()
    for i, res in enumerate(results):
        col = palette[i % len(palette)]
        x, y = res["x"], res["y"]
        w, h = piece_resized.shape[1], piece_resized.shape[0]
        cv2.rectangle(puzzle_marked, (x,y), (x+w, y+h), col, 2)
        cv2.putText(puzzle_marked, f"{i+1}:{res['score']:.3f}", (x, y-6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
        overlay_img = overlay_transparent(puzzle_img.copy(), piece_resized, x, y)
        cv2.imshow(f"Option {i+1} score={res['score']:.3f}", resize_to_screen(overlay_img))
        cv2.imwrite(f"option_{i+1}_overlay.png", overlay_img)
        print(f"[Option {i+1}] score={res['score']:.4f}, color={res['color_score']:.3f}, edge={res['edge_score']:.3f}, orb={res['orb_score']:.3f} @ ({x},{y})")

    cv2.imshow("Candidates (numbered)", resize_to_screen(puzzle_marked))
    print("Press any key to close.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
