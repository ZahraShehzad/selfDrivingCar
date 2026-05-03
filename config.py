# YOLO Config 
YOLO_MODEL    = "yolov8n.pt"
YOLO_CONF     = 0.25
YOLO_IMG_SIZE = 320
MIN_BOX_H_RATIO    = 0.025
MAX_BOX_AREA_RATIO = 0.15

CLASS_MIN_CONF = {
    0:  0.45,
    2:  0.30,
    7:  0.30,
    3:  0.25,  
    9:  0.30,
    10: 0.25,
    11: 0.30,
    58: 0.20,
}

CLASSES_WE_CARE_ABOUT = {
    0:  ("person",       (0,  128, 255)),
    2:  ("car",          (0,  255,   0)),
    3:  ("motorcycle",   (255, 165,  0)),
    7:  ("truck",        (0,  200, 200)),
    10: ("barrier",      (0,  165, 255)),
    9:  ("traffic light",(0,    0, 255)),
    11: ("stop sign",    (180,  0, 255)),
    58: ("potted plant", (0,  255, 255)),
}

# Lane / Ghost Config 
GHOST_FORGIVENESS    = 90   # frames to hold last known alignment when lane is lost
KMEANS_COOLDOWN_FRAMES = 4  # re-run k-means only every 4th frame in the fallback path
