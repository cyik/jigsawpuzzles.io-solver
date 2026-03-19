import cv2
import numpy as np
import sys

TILE_SIZE = 130
N_OPTIONS = 3

def resize_to_screen(img, max_w=1920, max_h=1080):
    h, w = img.shape[:2]
    scale = min(max_w / w, max_h / h, 1.0)
    return cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

def crop_to_alpha(img):
    if img.shape[2] != 4:
        return img
    alpha = img[:, :, 3]
    coords = cv2.findNonZero(alpha)
    if coords is None:
        return img
    x, y, w, h = cv2.boundingRect(coords)
    return img[y:y+h, x:x+w]

def overlay_transparent(background, overlay, x, y):
    h, w = overlay.shape[:2]
    overlay_rgb = overlay[:, :, :3]
    mask = overlay[:, :, 3] / 255.0 if overlay.shape[2] == 4 else np.ones((h, w))

    bg_crop = background[y:y+h, x:x+w]

    for c in range(3):
        bg_crop[:, :, c] = (
            (1 - mask) * bg_crop[:, :, c] +
            mask * overlay_rgb[:, :, c]
        )

    background[y:y+h, x:x+w] = bg_crop
    return background

def find_top_matches(puzzle_img, piece_img, n_options=3):

    # ---- Normalize piece size ----
    piece_img = crop_to_alpha(piece_img)
    piece_img = cv2.resize(
        piece_img,
        (TILE_SIZE, TILE_SIZE),
        interpolation=cv2.INTER_AREA
    )

    piece_gray = cv2.cvtColor(piece_img[:, :, :3], cv2.COLOR_BGR2GRAY)
    puzzle_gray = cv2.cvtColor(puzzle_img, cv2.COLOR_BGR2GRAY)

    mask = piece_img[:, :, 3] if piece_img.shape[2] == 4 else None

    result = cv2.matchTemplate(
        puzzle_gray,
        piece_gray,
        cv2.TM_CCOEFF_NORMED,
        mask=mask
    )

    matches = []
    result_copy = result.copy()

    for _ in range(n_options):
        _, max_val, _, max_loc = cv2.minMaxLoc(result_copy)

        x, y = max_loc
        matches.append((max_val, x, y))

        # Suppress nearby region so we don't pick the same spot again
        x1 = max(0, x - TILE_SIZE//2)
        y1 = max(0, y - TILE_SIZE//2)
        x2 = min(result.shape[1], x + TILE_SIZE//2)
        y2 = min(result.shape[0], y + TILE_SIZE//2)

        result_copy[y1:y2, x1:x2] = -1  # suppress area

    return matches, piece_img


if __name__ == "__main__":

    args = sys.argv[1:]
    puzzle_path = args[0] if len(args) > 0 else "puzzle.png"
    piece_path = args[1] if len(args) > 1 else "piece.png"

    puzzle_img = cv2.imread(puzzle_path)
    piece_img = cv2.imread(piece_path, cv2.IMREAD_UNCHANGED)

    if puzzle_img is None or piece_img is None:
        raise SystemExit("Could not load images!")

    print("Puzzle size:", puzzle_img.shape)
    print("Original piece size:", piece_img.shape)

    results, resized_piece = find_top_matches(puzzle_img, piece_img, N_OPTIONS)

    puzzle_marked = puzzle_img.copy()

    for i, (score, x, y) in enumerate(results):
        cv2.rectangle(
            puzzle_marked,
            (x, y),
            (x + TILE_SIZE, y + TILE_SIZE),
            (0, 255, 0),
            2
        )

        cv2.putText(
            puzzle_marked,
            f"{i+1}:{score:.4f}",
            (x, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

        overlay = overlay_transparent(puzzle_img.copy(), resized_piece, x, y)
        cv2.imshow(f"Option {i+1} score={score:.4f}", resize_to_screen(overlay))

        print(f"[Option {i+1}] score={score:.5f} @ ({x},{y})")

    cv2.imshow("Top Matches", resize_to_screen(puzzle_marked))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
