import cv2
import time
import multiprocessing as mp

from config import GHOST_FORGIVENESS, KMEANS_COOLDOWN_FRAMES
from processor import (
    yolo_process_worker,
    extract_lane_color_mask,
    detect_and_draw_two_strongest_lines,
    kmeans_segmentation,
    lane_lines_detection,
    warp_lane_area,
    draw_yolo_boxes,
    decisionControl,
    draw_hud,
)
import processor   # needed to read _last_ext_lines after lane functions update it



_ghost_frame_count = 0
_last_good_align   = "Centered"
_kmeans_cooldown   = 0
_last_seg_result   = None

SHOW_INTERMEDIATES = False


#  MAIN
def main():
    global _ghost_frame_count, _last_good_align, _kmeans_cooldown, _last_seg_result

    # Object state machine settings
    object_detection_state = "MOVE"
    HOLD_FRAMES_TOTAL      = 20
    VERIFY_FRAMES_TOTAL    = 8
    hold_frames            = 0
    verify_frames          = 0
    active_object_labels   = []
    YOLO_SUBMIT_EVERY      = 2   # default; updated per state each frame

    #  Video source
    source = r"C:\Users\zaras\PycharmProjects\project\2.mp4"
    # source = "http://192.168.10.2:8080/video"   # IP camera (uncomment to use)

    cap = cv2.VideoCapture(source)
    # cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)         # reduce latency for live streams
    if not cap.isOpened():
        print(f" Cannot open: {source}")
        return

    # Launch YOLO worker and wait for model to load 
    frame_queue  = mp.Queue(maxsize=1)
    result_queue = mp.Queue(maxsize=1)
    ready_event  = mp.Event()
    yolo_proc    = mp.Process(
        target=yolo_process_worker,
        args=(frame_queue, result_queue, ready_event),
        daemon=True,
    )
    yolo_proc.start()
    print(" Waiting for YOLO model to load...")
    ready_event.wait()   # blocks here; video hasn't started yet
    print(" YOLO ready — starting video now")

    # Counters & metrics
    frame_id     = 0
    fps_display  = 0.0
    fps_counter  = 0
    t_start      = time.time()
    screenshot_n = 0
    _latest_dets = []   # always holds the most recent YOLO result

    # Pre-initialise ext_lines so YOLO can start from frame 1 
    # Peek at the first frame to derive a sensible default trapezoid,
    # then rewind so the main loop processes every frame normally.
    _ret, _peek = cap.read()
    if _ret:
        _ph = _peek.shape[0] // 2
        _pw = _peek.shape[1] // 2
        processor._last_ext_lines = [
            ((_pw * 2 // 5, _ph), (int(_pw * 0.45), _ph // 2)),   # ext_neg
            ((_pw * 3 // 5, _ph), (int(_pw * 0.55), _ph // 2)),   # ext_pos
        ]
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)   # rewind — no frame lost

    print("\n Running — Q to quit, S to screenshot\n")

    #Debug window setup 
    cv2.namedWindow("Self-Driving Final Output", cv2.WINDOW_NORMAL)

    if SHOW_INTERMEDIATES:
        cv2.namedWindow("Intermediary 1: Color Mask", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Intermediary 2: Canny Edges", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Intermediary 3: KMeans Segmentation", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Intermediary 4: Lane Polygon Mask", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Intermediary 5: Hough Line Frame", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Intermediary 6: YOLO Warp View", cv2.WINDOW_NORMAL)

        cv2.resizeWindow("Intermediary 1: Color Mask", 400, 300)
        cv2.resizeWindow("Intermediary 2: Canny Edges", 400, 300)
        cv2.resizeWindow("Intermediary 3: KMeans Segmentation", 400, 300)
        cv2.resizeWindow("Intermediary 4: Lane Polygon Mask", 400, 300)
        cv2.resizeWindow("Intermediary 5: Hough Line Frame", 400, 300)
        cv2.resizeWindow("Intermediary 6: YOLO Warp View", 400, 300)

    cv2.resizeWindow("Self-Driving Final Output", 1000, 600)


    #  MAIN LOOP
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_id += 1
        img = cv2.resize(frame, (frame.shape[1] // 2, frame.shape[0] // 2))

        # Cap wide videos to 1280 px wide so downstream maths stays consistent
        if img.shape[1] > 1280:
            scale = 1280 / img.shape[1]
            img = cv2.resize(img, (1280, int(img.shape[0] * scale)))
            if frame_id == 1:
                print(f"Wide video resized: now {img.shape[1]}x{img.shape[0]}")
        else:
            if frame_id == 1:
                print(f"Normal video size: {img.shape[1]}x{img.shape[0]}")

        # 1. Colour & edge processing
        mask, edges, has_lane = extract_lane_color_mask(img)

        if SHOW_INTERMEDIATES:
            cv2.imshow("Intermediary 1: Color Mask", mask)
            cv2.imshow("Intermediary 2: Canny Edges", edges)

        # 2. Lane detection
        if has_lane:
            # Colour mask gave enough edges — use Hough directly
            lane_img, lane_mask, align = detect_and_draw_two_strongest_lines(img, edges)

            if SHOW_INTERMEDIATES:
                cv2.imshow("Intermediary 4: Lane Polygon Mask", lane_mask)
                cv2.imshow("Intermediary 5: Hough Line Frame", lane_img)

        else:
            # Colour mask failed — fall back to k-means road segmentation
            _kmeans_cooldown += 1
            if _kmeans_cooldown >= KMEANS_COOLDOWN_FRAMES or _last_seg_result is None:
                _last_seg_result = kmeans_segmentation(img)
                _kmeans_cooldown = 0

            # Reuse cached segmentation
            if SHOW_INTERMEDIATES:
                cv2.imshow("Intermediary 3: KMeans Segmentation", _last_seg_result)

            lane_img, lane_mask, align = lane_lines_detection(_last_seg_result, img)

            if SHOW_INTERMEDIATES:
                cv2.imshow("Intermediary 4: Lane Polygon Mask", lane_mask)
                cv2.imshow("Intermediary 5: Hough Line Frame", lane_img)

        # 3. Ghost forgiveness
        # Hold the last known alignment for up to GHOST_FORGIVENESS frames
        # so brief lane-loss events don't immediately trigger a stop.
        if align == "Ghost":
            _ghost_frame_count += 1
            effective_align = (_last_good_align
                               if _ghost_frame_count <= GHOST_FORGIVENESS
                               else "Ghost")
            cv2.putText(lane_img,
                        f"[Ghost grace {_ghost_frame_count}/{GHOST_FORGIVENESS}]",
                        (10, lane_img.shape[0] - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1)
        else:
            _ghost_frame_count = 0
            _last_good_align   = align
            effective_align    = align

        #  4. Dynamic YOLO submission rate 
        # Submit fewer frames when stopped (HOLD) to reduce CPU load,
        # and the most frames during VERIFY for maximum accuracy.
        if   object_detection_state == "MOVE":   YOLO_SUBMIT_EVERY = 2
        elif object_detection_state == "HOLD":   YOLO_SUBMIT_EVERY = 4
        elif object_detection_state == "VERIFY": YOLO_SUBMIT_EVERY = 1



        # Send ONLY latest frame to YOLO worker
        if processor._last_ext_lines and frame_id % YOLO_SUBMIT_EVERY == 0:

            ext_lines_snapshot = processor._last_ext_lines.copy()

            if len(ext_lines_snapshot) != 2:
                continue

            ext_neg = ext_lines_snapshot[0]
            ext_pos = ext_lines_snapshot[1]

            
            try:
                warped_view, _ = warp_lane_area(img, ext_neg, ext_pos)
                if SHOW_INTERMEDIATES:
                    cv2.imshow("Intermediary 6: YOLO Warp View", warped_view)
            except Exception:
                pass

            frame_id_snapshot = frame_id

            
            try:
                if frame_queue.full():
                    try:
                        frame_queue.get_nowait()  # drop oldest frame
                    except:
                        pass

                frame_queue.put_nowait(
                    (frame_id_snapshot, img.copy(), ext_neg, ext_pos)
                )

            except mp.queues.Full:
                pass

        #  5. Collect YOLO result
        got_new_result = False
        try:
            new_dets     = result_queue.get_nowait()
            _latest_dets = new_dets
            got_new_result = True
        except mp.queues.Empty:
            pass

        dets_orig = _latest_dets   # always use the most recent known result
        obj_found = len(dets_orig) > 0

        if obj_found:
            active_object_labels = list(set([d[0] for d in dets_orig]))
        if not obj_found and object_detection_state == "MOVE":
            active_object_labels = []

        # 6. Object state machine 
        # MOVE  → object detected               → HOLD
        # HOLD  → hold expires with no object   → VERIFY
        # VERIFY→ fresh YOLO confirms clear      → MOVE
        # VERIFY→ fresh YOLO re-detects object   → HOLD
        if object_detection_state == "MOVE":
            if obj_found:
                object_detection_state = "HOLD"
                hold_frames   = HOLD_FRAMES_TOTAL
                verify_frames = 0

        elif object_detection_state == "HOLD":
            # YOLO detections ignored for the decision during HOLD,
            # but YOLO keeps running in the background
            hold_frames -= 1
            if obj_found:
                hold_frames = HOLD_FRAMES_TOTAL   # reset if object still present
            if hold_frames <= 0:
                object_detection_state = "VERIFY"
                verify_frames = VERIFY_FRAMES_TOTAL

        elif object_detection_state == "VERIFY":
            # Only act when YOLO returned a fresh result this frame
            if got_new_result:
                if obj_found:
                    object_detection_state = "HOLD"
                    hold_frames   = HOLD_FRAMES_TOTAL
                    verify_frames = 0
                else:
                    verify_frames -= 1
                    if verify_frames <= 0:
                        object_detection_state = "MOVE"

        #  7. Render output 
        det_labels = [d[0] for d in dets_orig]
        if dets_orig:
            lane_img = draw_yolo_boxes(lane_img, dets_orig)

        final_object_block          = (object_detection_state != "MOVE")
        move_decision, steer_decision = decisionControl(effective_align, final_object_block)

        # FPS update every 0.5 s so the display feels responsive
        fps_counter += 1
        elapsed = time.time() - t_start
        if elapsed >= 0.5:
            fps_display = fps_counter / elapsed
            fps_counter = 0
            t_start     = time.time()

        output = draw_hud(lane_img, effective_align, obj_found,
                          move_decision, steer_decision,
                          frame_id, fps_display, det_labels,
                          object_detection_state,
                          active_object_labels)

        cv2.imshow("Self-Driving Final Output", output)

        # Console log every 30 frames
        if frame_id % 30 == 0:
            print(f"Frame {frame_id:05d} | FPS {fps_display:5.1f} | "
                  f"Align: {effective_align:10s} | {move_decision} / {steer_decision} | "
                  f"Objects: {set(det_labels) if det_labels else '—'}")

        #  8. Input handling 
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            fname = f"screenshot_{screenshot_n:03d}.jpg"
            cv2.imwrite(fname, output)
            screenshot_n += 1
            print(f" Saved {fname}")

    # Shutdown
    frame_queue.put(None)      # sentinel tells YOLO worker to exit
    yolo_proc.join(timeout=3)
    cap.release()
    cv2.destroyAllWindows()
    print(f"\n  {frame_id} frames processed.")


# mp.set_start_method('spawn') is required on Windows for multiprocessing
# to work correctly with CUDA / OpenCV in child processes.
if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()
