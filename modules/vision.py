import time
import logging
from typing import Tuple, List, Dict, Any, Optional
import cv2
import numpy as np
from ultralytics import YOLO

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

logger = logging.getLogger("HumanDetector")


class HumanDetector:
    """
    YOLOv8-based real-time human detection.
    Performs inference on camera frames, filters for the COCO 'person' class (index 0),
    draws bounding boxes, and returns detection metrics.
    """

    def __init__(self, model_path: Optional[str] = None, conf_threshold: Optional[float] = None):
        self.device = config.DEVICE
        self.conf_threshold = conf_threshold if conf_threshold is not None else config.YOLO_CONF_THRESHOLD
        self.model_path = model_path if model_path is not None else str(config.YOLO_MODEL_NAME)
        self.person_class_id = 0  # COCO class index 0 = "person"
        self.is_loaded = False
        self.last_latency_ms = 0.0

        try:
            logger.info(f"[Vision] Loading YOLOv8 on {self.device} from {self.model_path}...")
            self.model = YOLO(self.model_path)
            self.model.to(self.device)
            self.is_loaded = True
        except Exception as e:
            logger.error(f"[Vision] Failed to load YOLO model: {e}")
            self.model = None

    def detect_humans(
        self,
        frame: Optional[np.ndarray],
    ) -> Tuple[bool, Optional[np.ndarray], List[Dict[str, Any]], float]:
        """
        Runs YOLO on a BGR frame.
        
        Returns:
          (human_found: bool, annotated_frame: np.ndarray, detections: list, max_confidence: float)
          detections: list of dicts with keys 'bbox' (x1, y1, x2, y2), 'confidence', 'class_name'
        """
        if (
            frame is None
            or not isinstance(frame, np.ndarray)
            or frame.ndim != 3
            or frame.shape[0] == 0
            or frame.shape[1] == 0
            or not self.is_loaded
            or self.model is None
        ):
            return False, frame, [], 0.0

        t0 = time.time()
        try:
            results = self.model(
                frame,
                conf=self.conf_threshold,
                classes=[self.person_class_id],
                verbose=False,
                device=self.device,
                imgsz=416,
            )
            self.last_latency_ms = (time.time() - t0) * 1000.0

            detections = []
            human_found = False
            max_conf = 0.0

            h, w = frame.shape[:2]

            for r in results:
                for box in r.boxes:
                    human_found = True
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    conf = float(box.conf[0])
                    max_conf = max(max_conf, conf)

                    # Clamp bounding boxes to valid image dimensions
                    bx1 = max(0, min(w - 1, int(x1)))
                    by1 = max(0, min(h - 1, int(y1)))
                    bx2 = max(0, min(w - 1, int(x2)))
                    by2 = max(0, min(h - 1, int(y2)))

                    detections.append({
                        "bbox": (bx1, by1, bx2, by2),
                        "confidence": conf,
                        "class_name": "person",
                    })

            annotated_frame = results[0].plot() if len(results) > 0 else frame
            return human_found, annotated_frame, detections, max_conf

        except Exception as e:
            logger.error(f"[Vision] Error during human detection inference: {e}")
            return False, frame, [], 0.0