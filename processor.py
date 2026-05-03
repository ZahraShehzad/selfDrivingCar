import cv2
import numpy as np
import multiprocessing as mp
from ultralytics import YOLO

from config import (
    YOLO_MODEL, YOLO_CONF, YOLO_IMG_SIZE,
    MIN_BOX_H_RATIO, MAX_BOX_AREA_RATIO,
    CLASS_MIN_CONF, CLASSES_WE_CARE_ABOUT,
)

# ── State memory (main process only) ────────────────────────
# These are mutated by lane-detection functions each frame.
# The YOLO worker process has its own copy — no sharing needed.
_last_lane_mask  = None   # last successfully drawn lane fill mask
_last_ext_lines  = []     # last known [ext_neg, ext_pos] extrapolated line endpoints
_prev_centers    = None   # k-means cluster centres from the previous run (warm-start)



#  YOLO WORKER — runs in a separate process, owns its own GIL
def yolo_process_worker(frame_queue: mp.Queue, result_queue: mp.Queue, ready_event: mp.Event):
    model = YOLO(YOLO_MODEL)
    model.to("cuda")
    print("[YOLO process] ready")
    ready_event.set()

    while True:

        # 1 Wait for at least one frame
        item = frame_queue.get()
        if item is None:
            break

        # 2 DRAIN QUEUE → keep ONLY latest frame
        while True:
            try:
                latest = frame_queue.get_nowait()
                if latest is None:
                    item = None
                    break
                item = latest
            except Exception:
                break

        if item is None:
            break

        frame_id, frame, ext_neg, ext_pos = item

        # 3 RUN YOLO
        dets = _run_yolo(model, frame, ext_neg, ext_pos)

        # 4 OVERWRITE result queue (keep only latest output)
        try:
            while True:
                result_queue.get_nowait()
        except Exception:
            pass

        result_queue.put(dets)



def _run_yolo(model, img, ext_neg, ext_pos):
    fh, fw = img.shape[:2]

    # ─────────────────────────────────────────────
    # STEP 1: WARP THE LANE TRAPEZOID
    # Only pixels inside the lane trapezoid survive the warp
    # Cars parked outside the lane are NOT in the warped image at all
    # ─────────────────────────────────────────────
    OUTPUT_W, OUTPUT_H = 416, 416

    # SAFE CHECK (prevents crash / bad warp)
    if ext_neg is None or ext_pos is None:
        return []

    src_pts = np.array([
        ext_neg[0],
        ext_neg[1],
        ext_pos[1],
        ext_pos[0]
    ], dtype=np.float32)
    dst_pts = np.float32([
        [0, OUTPUT_H],
        [0, 0],
        [OUTPUT_W, 0],
        [OUTPUT_W, OUTPUT_H],
    ])

    try:
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        M_inv = cv2.invert(M)[1]
        warped = cv2.warpPerspective(img, M, (OUTPUT_W, OUTPUT_H))
    except Exception:
        return []

    if warped.shape[0] < 10 or warped.shape[1] < 10:
        return []

    # ─────────────────────────────────────────────
    # STEP 2: RUN YOLO ON WARPED LANE IMAGE
    # ─────────────────────────────────────────────
    results = model(warped, imgsz=YOLO_IMG_SIZE, conf=0.20, verbose=False)

    frame_area = fh * fw
    dets_orig = []

    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            if cls_id not in CLASSES_WE_CARE_ABOUT:
                continue

            conf_val = float(box.conf[0])
            if conf_val < CLASS_MIN_CONF.get(cls_id, YOLO_CONF):
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # ── Filter degenerate boxes in warped space ──
            box_w_ratio = (x2 - x1) / OUTPUT_W
            box_h_ratio = (y2 - y1) / OUTPUT_H
            if box_w_ratio > 0.60 or box_h_ratio > 0.80:
                continue
            if box_h_ratio < MIN_BOX_H_RATIO:
                continue

            # ── Map corners back to original frame via inverse warp ──
            corners_warped = np.float32([
                [x1, y1], [x2, y1],
                [x2, y2], [x1, y2]
            ]).reshape(-1, 1, 2)

            corners_orig = cv2.perspectiveTransform(corners_warped, M_inv)
            corners_orig = corners_orig.reshape(-1, 2).astype(int)

            ox1 = int(corners_orig[:, 0].min())
            oy1 = int(corners_orig[:, 1].min())
            ox2 = int(corners_orig[:, 0].max())
            oy2 = int(corners_orig[:, 1].max())

            # ── Filter giant/garbage inverse-warp results ──
            mapped_area = (ox2 - ox1) * (oy2 - oy1)
            if mapped_area > frame_area * MAX_BOX_AREA_RATIO:
                continue
            if oy1 > fh * 0.80:
                continue

            # ── Clamp to frame ──
            ox1 = max(0, ox1);
            oy1 = max(0, oy1)
            ox2 = min(fw - 1, ox2);
            oy2 = min(fh - 1, oy2)

            label, color = CLASSES_WE_CARE_ABOUT[cls_id]
            dets_orig.append((label, conf_val, ox1, oy1, ox2, oy2, color))

    return dets_orig


# ── Lane Detection ───────────────────────────────────────────
"""Mask everything outside the lane trapezoid ROI."""
def region_of_interest(img):
    r, c = img.shape[:2]
    verts = np.array([[
        (0, r), (2 * c // 5, r // 2), (3 * c // 5, r // 2), (c, r)
    ]], dtype=np.int32)
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, verts, 255)
    return cv2.bitwise_and(img, mask)

"""Morphological close to bridge small gaps in the road mask."""
def fill_cracks(mask, kernel_size=9):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)



"""
    HSV colour filter for yellow and white lane markings.
    Returns (mask, canny_edges, has_lane).
    has_lane is True when enough edge pixels exist in the lower third.
    """

def extract_lane_color_mask(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    low_y2 = np.array([10, 20, 190], np.uint8);
    high_y2 = np.array([25, 60, 220], np.uint8)
    low_w = np.array([140, 0, 220], np.uint8);
    high_w = np.array([160, 6, 250], np.uint8)
    mask = cv2.bitwise_or(cv2.inRange(hsv, low_y2, high_y2), cv2.inRange(hsv, low_w, high_w))
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    edges = cv2.Canny(mask, 100, 200)
    h = image.shape[0]
    return mask, edges, np.count_nonzero(edges[h * 2 // 3:, :]) > 200

"""
    Segment the frame into k colour clusters for road/non-road separation.
    Runs on a 0.25x downscale for speed; warm-starts from previous centres.
    """

def kmeans_segmentation(img, k=4):
    global _prev_centers
    # ── Change fx=0.5 to fx=0.25 ──────────────────────────
    small = cv2.resize(img, (0, 0), fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
    blur = cv2.GaussianBlur(small, (9, 9), 0)  # smaller kernel too (was 19,19)
    Z = blur.reshape(-1, 3).astype(np.float32)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 5, 2.0)  # fewer iters (was 10)
    if _prev_centers is not None:
        diffs = Z[:, None, :] - _prev_centers[None, :, :]
        li = np.argmin(np.linalg.norm(diffs, axis=2), axis=1).astype(np.int32)
        _, labels, centers = cv2.kmeans(Z, k, li, crit, 1, cv2.KMEANS_USE_INITIAL_LABELS, _prev_centers)
    else:
        _, labels, centers = cv2.kmeans(Z, k, None, crit, 3, cv2.KMEANS_PP_CENTERS)  # attempts 3 (was 10)
    _prev_centers = centers
    grays = [1, 64, 128, 255]
    gray_small = np.array([grays[l] for l in labels.flatten()], np.uint8).reshape(blur.shape[:2])
    return cv2.resize(gray_small, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

"""Return the k-means label value that covers the most of the ROI (the road cluster)."""

def best_label(mask_img):
    roi = region_of_interest(mask_img)

    scores = {}
    for v in [1, 64, 128, 255]:
        scores[v] = np.count_nonzero(roi == v)

    return max(scores, key=scores.get)


"""Keep only the largest connected component of the road cluster."""
def create_road_mask(mask_img, label):
    bin_img = (mask_img == label).astype(np.uint8)
    num, labels = cv2.connectedComponents(bin_img, connectivity=8)
    if num <= 1: return np.zeros_like(bin_img)
    best = max(range(1, num), key=lambda l: np.sum(labels == l))
    return (labels == best).astype(np.uint8) * 255

"""Extrapolate a line segment to span from the horizon to the bottom of the frame."""
def extend_line(line, img_h):
    x1, y1, x2, y2 = line
    if x2 == x1: return None
    m = (y2 - y1) / (x2 - x1);
    b = y1 - m * x1
    return (int((img_h - b) / m), img_h), (int(((img_h // 2) - b) / m), img_h // 2)


"""
    Find the vanishing point of the two lane lines.
    Returns (alignment_string, vp_x, frame_centre_x).
    Alignment is Centered / LaneLeft / LaneRight / Parallel.
    """
def analyze_alignment(img_w, neg_ext, pos_ext, thresh=0.05):
    (x1, y1), (x2, y2) = neg_ext;
    (x3, y3), (x4, y4) = pos_ext
    A1, B1, C1 = y2 - y1, x1 - x2, (y2 - y1) * x1 + (x1 - x2) * y1
    A2, B2, C2 = y4 - y3, x3 - x4, (y4 - y3) * x3 + (x3 - x4) * y3
    D = A1 * B2 - A2 * B1
    if D == 0: return "Parallel", None, img_w // 2
    vp_x = int((B2 * C1 - B1 * C2) / D);
    ctr = img_w // 2;
    diff = vp_x - ctr
    if abs(diff) <= img_w * thresh: return "Centered", vp_x, ctr
    return ("LaneLeft", vp_x, ctr) if diff < 0 else ("LaneRight", vp_x, ctr)


"""Mirror a lane line across the frame centre to synthesise the opposite lane."""
def _mirror_line(line, img_w):
    x1, y1, x2, y2 = line;
    cx = img_w // 2
    mx1, mx2 = 2 * cx - x1, 2 * cx - x2
    return None if mx2 == mx1 else (mx1, y1, mx2, y2)


"""Draw the last known lane mask and lines onto the frame when detection fails."""
def _ghost_overlay(img, color=(0, 255, 255)):
    out = img.copy()
    if _last_lane_mask is not None:
        ov = out.copy();
        ov[_last_lane_mask > 0] = color
        cv2.addWeighted(ov, 0.3, out, 0.7, 0, out)
    ext_snapshot = _last_ext_lines.copy()

    for idx, ext in enumerate(ext_snapshot):
        cv2.line(out, ext[0], ext[1], [(0, 0, 255), (0, 255, 0)][idx], 4)
    return out


"""Select the strongest positive-slope and negative-slope Hough lines,
    extrapolate them, fill the lane polygon, and return
    (annotated_frame, lane_mask, alignment_string).
    Falls back to ghost overlay when no valid lines are found."""
def detect_and_draw_two_strongest_lines(image, edge_img):
    global _last_lane_mask, _last_ext_lines
    h, w = image.shape[:2]
    if _last_lane_mask is None:
        _last_lane_mask = np.zeros((h, w), np.uint8);
        _last_ext_lines = []

    lines = cv2.HoughLinesP(edge_img, 1, np.pi / 180, 50, minLineLength=30, maxLineGap=20)
    if lines is None:
        return _ghost_overlay(image), _last_lane_mask, "Ghost"

    pos, neg = [], []
    for x1, y1, x2, y2 in lines.reshape(-1, 4):
        dx, dy = x2 - x1, y2 - y1
        if dx == 0: continue
        sl = dy / dx;
        ang = abs(np.degrees(np.arctan(sl)));
        ln = np.hypot(dx, dy)
        if ln < 40 or ang < 10 or ang > 80: continue
        if (y1 < 0.4 * h and y2 < 0.4 * h) or (y1 > 0.95 * h and y2 > 0.95 * h): continue
        (neg if sl < 0 else pos).append((ln, (x1, y1, x2, y2)))

    if not neg and not pos:
        return _ghost_overlay(image), _last_lane_mask, "Ghost"
    if neg and not pos:
        _, bn = max(neg, key=lambda t: t[0]);
        p = _mirror_line(bn, w)
        if p is None: return _ghost_overlay(image), _last_lane_mask, "Ghost"
        pos = [(0, p)]
    if pos and not neg:
        _, bp = max(pos, key=lambda t: t[0]);
        n = _mirror_line(bp, w)
        if n is None: return _ghost_overlay(image), _last_lane_mask, "Ghost"
        neg = [(0, n)]

    _, nline = max(neg, key=lambda t: t[0]);
    _, pline = max(pos, key=lambda t: t[0])
    ext_neg = extend_line(nline, h);
    ext_pos = extend_line(pline, h)
    if not ext_neg or not ext_pos:
        return _ghost_overlay(image), _last_lane_mask, "Ghost"

    # Update shared ext_lines — main process only, no lock needed
    _last_ext_lines = [ext_neg, ext_pos]

    out = image.copy();
    mask = np.zeros((h, w), np.uint8)
    for idx, ext in enumerate([ext_neg, ext_pos]):
        cv2.line(out, ext[0], ext[1], [(0, 0, 255), (0, 255, 0)][idx], 4)
    pts = np.array([ext_neg[0], ext_neg[1], ext_pos[1], ext_pos[0]], np.int32)
    cv2.fillPoly(mask, [pts], 255)
    ov = out.copy();
    cv2.fillPoly(ov, [pts], (0, 255, 255))
    cv2.addWeighted(ov, 0.3, out, 0.7, 0, out)
    align, _, _ = analyze_alignment(w, ext_neg, ext_pos)
    _last_lane_mask = mask.copy()
    return out, mask, align

"""Full k-means fallback pipeline: select road cluster → mask → Canny → Hough."""
def lane_lines_detection(proc_img, og_img):
    bestlabel = best_label(proc_img)
    road_mask = create_road_mask(proc_img, bestlabel)
    edges = cv2.Canny(fill_cracks(road_mask), 50, 150)
    return detect_and_draw_two_strongest_lines(og_img, edges)

"""Perspective-warp the lane trapezoid to a top-down rectangle."""
def warp_lane_area(image, ext_neg, ext_pos, output_size=(320, 480)):
    src_pts = np.float32([ext_neg[0], ext_neg[1], ext_pos[1], ext_pos[0]])
    dst_w, dst_h = output_size
    dst_pts = np.float32([[0, dst_h], [0, 0], [dst_w, 0], [dst_w, dst_h]])
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    return cv2.warpPerspective(image, M, (dst_w, dst_h)), M

"""Draw labelled bounding boxes (remapped from warped space) onto the frame."""
def draw_yolo_boxes(frame, dets_orig):
    fh, fw = frame.shape[:2]
    for (label, conf_val, x1, y1, x2, y2, color) in dets_orig:
        x1, y1 = max(0, x1), max(0, y1);
        x2, y2 = min(fw - 1, x2), min(fh - 1, y2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        text = f"{label} {conf_val:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(frame, text, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
    return frame

"""Map (lane alignment, object presence) → (move/stop decision, steer direction)."""
def decisionControl(alignment, object_found):
    if alignment == "Ghost":        return "Stop", "No Lane Detected"
    if object_found:
        return "Stop", "Forward Blocked" if alignment == "Centered" else "Realign First"
    if alignment == "Centered":     return "Move", "Forward"
    if alignment == "LaneLeft":     return "Move", "Steer Left"
    if alignment == "LaneRight":    return "Move", "Steer Right"
    return "Stop", "Unknown"


"""Render the semi-transparent HUD bar with ACTION / LANE / FPS / STEER / OBJECT."""
def draw_hud(frame, align, obj, move_dec, steer_dec, frame_id, fps, det_labels, object_detection_state,
             active_object_labels):
    h, w = frame.shape[:2]
    ov = frame.copy();
    cv2.rectangle(ov, (0, 0), (w, 80), (0, 0, 0), -1)
    cv2.addWeighted(ov, 0.55, frame, 0.45, 0, frame)
    font = cv2.FONT_HERSHEY_SIMPLEX
    GREEN = (0, 255, 80);
    RED = (0, 60, 255);
    ORANGE = (0, 165, 255);
    GREY = (180, 180, 180)
    ac = GREEN if move_dec == "Move" else RED
    sc = RED if move_dec == "Stop" else GREEN
    lc = GREEN if align == "Centered" else (0, 200, 255) if align == "Ghost" else ORANGE
    oc = RED if active_object_labels else GREEN

    col1, col2, col3 = 10, w // 3, max(w - 160, w // 2 + 20)
    cv2.putText(frame, f"ACTION : {move_dec}", (col1, 30), font, 0.68, ac, 2)
    cv2.putText(frame, f"LANE   : {align}", (col2, 30), font, 0.68, lc, 2)
    fps_text = f"FPS : {fps:.1f}"
    (fps_tw, _), _ = cv2.getTextSize(fps_text, font, 0.68, 2)
    fps_x = min(col3, w - fps_tw - 10)  # never clips outside frame
    cv2.putText(frame, fps_text, (fps_x, 30), font, 0.68, GREY, 2)
    cv2.putText(frame, f"STEER  : {steer_dec}", (col1, 62), font, 0.68, sc, 2)
    if active_object_labels:
        obj_text = "OBJECT : YES " + ", ".join(sorted(active_object_labels))
    else:
        obj_text = "OBJECT : Clear"
    cv2.putText(frame, obj_text, (col2, 62), font, 0.68, oc, 2)
    return frame