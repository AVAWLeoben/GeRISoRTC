# -*- coding: utf-8 -*-
"""
AI Classifier Control Software - spawn-safe startup-config version.

Drop-in notes:
- No multiprocessing child process depends on settings created only inside __main__.
- The producer owns its local model, camera and mask history state.
- VERT_MOVEMENT is a multiprocessing.Value and can be changed from the UI.
- Ultralytics model.track() can be enabled from the UI to estimate vertical movement.
  The estimate is displayed but is NOT applied automatically.
"""

# %% Imports
import multiprocessing
import threading
import queue
from queue import Full, Empty

import os
import time
import cv2
import numpy as np
from pypylon import pylon
from ultralytics import YOLO
import yaml
import datetime 

from UI_LAYER import display

from NOZZLE_CONTROL_LAYER import (
    nozzle_control_UDP,
    nozzle_control_ARDUINO,
    nozzle_control_MODBUS,
    nozzle_control_SIMULATED
)

from collections import deque
movement_history = deque(maxlen=30)

import platform
import subprocess
# ---------------------------------------------------------------------------
# Persistent runtime settings helpers
# ---------------------------------------------------------------------------
RUNTIME_CONFIG_FILE = "runtime_settings.yaml"

def load_runtime_config():
    if os.path.exists(RUNTIME_CONFIG_FILE):
        try:
            with open(RUNTIME_CONFIG_FILE, "r") as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            print(f"Failed to load runtime settings: {e}")
    return {}


def _to_safe_yaml_data(value):
    """Convert multiprocessing/list-proxy/numpy-ish values into safe YAML types."""
    if isinstance(value, dict):
        return {str(k): _to_safe_yaml_data(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_safe_yaml_data(v) for v in value]
    try:
        if isinstance(value, np.generic):
            return value.item()
    except Exception:
        pass
    return value


def save_runtime_config(config):
    try:
        with open(RUNTIME_CONFIG_FILE, "w") as f:
            yaml.safe_dump(_to_safe_yaml_data(config), f, default_flow_style=False, sort_keys=False)
    except Exception as e:
        print(f"Failed to save runtime settings: {e}")



# ---------------------------------------------------------------------------
# Camera helpers
# ---------------------------------------------------------------------------

def reverseString(string):
    return string[::-1]


def connectCamera(camera_type, pfs_path, FPS, rerun=False, VIDEO_PATH=None, USB_SETTINGS_PATH=None, MVIMPACT_NIR_SETTINGS_PATH=None, NIR_CLASSIFIER_PATH=None, NIR_CLASSIFIER_KIND="SAM_PLACEHOLDER"):
    camera_type = camera_type.lower()

    if camera_type == "basler":
        return connectCamera_BASLER(pfs_path, FPS, rerun)

    elif camera_type == "usb":
        return connectCamera_USB(settings_path=USB_SETTINGS_PATH)

    elif camera_type == "simulated":
        return connectCamera_Simulated(VIDEO_PATH)

    elif camera_type in ("mvimpact_nir", "mvimpact-nir", "nir"):
        return connectCamera_MVIMPACT_NIR(
            settings_path=MVIMPACT_NIR_SETTINGS_PATH,
            fps=FPS.value if hasattr(FPS, "value") else FPS,
            classifier_path=NIR_CLASSIFIER_PATH,
            classifier_kind=NIR_CLASSIFIER_KIND,
        )

    else:
        raise ValueError(
            f"Unsupported CAMERA_TYPE: {camera_type}. Must be USB, Basler, SIMULATED, or MVIMPACT_NIR."
        )


def connectCamera_BASLER(pfs_path, FPS, rerun=False):
    tl_factory = pylon.TlFactory.GetInstance()
    devices = tl_factory.EnumerateDevices()
    if not devices:
        print("[Process-1] No camera devices found.")
        return None

    camera = pylon.InstantCamera(tl_factory.CreateDevice(devices[0]))
    camera.Open()
    pylon.FeaturePersistence.Load(pfs_path, camera.GetNodeMap())
    camera.AcquisitionFrameRateEnable.SetValue(True)
    camera_fps = float(FPS.value if hasattr(FPS, "value") else FPS)
    camera.AcquisitionFrameRate.SetValue(camera_fps)
    try:
        FPS.value = int(round(camera.AcquisitionFrameRate.GetValue()))
    except Exception:
        pass
    print(f"[Process-1] Basler camera FPS set to {camera_fps:.1f}")
    try: # To fix Pixel_Format to BGR8 for downstream consistency with cv2, pygame
        camera.PixelFormat.SetValue("BGR8")
    except Exception as e:
        print(f"Error when trying to set PixelFormat=BGR8. {e}")
        print("Check correct Pixel Format in Pylon Viewer")
    return camera


def connectCamera_USB(rerun=False, camera_index=0, width=2560, height=1440, fps=30, settings_path=None):
    settings = load_usb_camera_settings(settings_path)
    cam_cfg = settings.get("camera", {})
    fmt = cam_cfg.get("format", {})
    controls = cam_cfg.get("controls", {})

    camera_index = int(cam_cfg.get("index", camera_index))
    width = int(fmt.get("width", width))
    height = int(fmt.get("height", height))
    fps = float(fmt.get("fps", fps))
    fourcc = fmt.get("fourcc", "MJPG")

    backend = cv2.CAP_V4L2 if platform.system() == "Linux" else cv2.CAP_DSHOW
    camera = cv2.VideoCapture(camera_index, backend)

    if not camera.isOpened():
        print(f"[Process-1] No USB camera found at index {camera_index}.")
        return None

    camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    camera.set(cv2.CAP_PROP_FPS, fps)
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    time.sleep(0.3)

    if platform.system() == "Linux":
        device = cam_cfg.get("device_linux", f"/dev/video{camera_index}")
        apply_usb_settings_linux(device, controls)

    actual_width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = camera.get(cv2.CAP_PROP_FPS)

    print(f"[Process-1] USB camera connected: {actual_width}x{actual_height} @ {actual_fps:.1f} fps")
    print(f"[Process-1] USB settings loaded from: {settings_path}")

    return camera

def connectCamera_Simulated(VIDEO_PATH,fps=30):
    from simulated_camera import simulated_camera
    return simulated_camera(fps=fps,VIDEO_PATH=VIDEO_PATH).connect()

def connectCamera_MVIMPACT_NIR(settings_path=None, fps=30, classifier_path=None, classifier_kind="SAM_PLACEHOLDER"):
    from mvimpact_nir_camera import MvImpactNIRCamera
    return MvImpactNIRCamera(
        settings_path=settings_path,
        fps=fps,
        classifier_path=classifier_path,
        classifier_kind=classifier_kind,
    ).connect()

def load_usb_camera_settings(path):
    if not path or not os.path.exists(path):
        print(f"[USB] Settings YAML not found: {path}")
        return {}

    try:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        print(f"[USB] Failed to load USB settings YAML: {e}")
        return {}


def apply_usb_settings_linux(device, controls):
    for name, meta in controls.items():
        if not isinstance(meta, dict) or "value" not in meta:
            continue

        subprocess.run(
            ["v4l2-ctl", "-d", device, "-c", f"{name}={int(meta['value'])}"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )

def center_crop_square(img, size=640, offset_x=0, offset_y=0):
    if img is None:
        raise ValueError("Input image is None")

    h, w = img.shape[:2]
    if h < size or w < size:
        raise ValueError(f"Image too small for {size}x{size} crop: got {w}x{h}")

    cx = w // 2 + int(offset_x)
    cy = h // 2 + int(offset_y)
    half = size // 2

    x0 = max(0, min(w - size, cx - half))
    y0 = max(0, min(h - size, cy - half))

    return img[y0:y0 + size, x0:x0 + size]


def prepare_nir_frame_for_pipeline(img, output_size=640):
    """
    The placeholder NIR camera can intentionally produce native sensor-shaped
    data such as 312x220 frames or 312x1 classified lines.  The rest of this
    application expects image-like frames, so this helper creates a square
    visual/inference image while preserving the camera's native aspect/pattern.
    """
    if img is None:
        raise ValueError("Input NIR image is None")

    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    h, w = img.shape[:2]
    interpolation = cv2.INTER_NEAREST if h <= 4 else cv2.INTER_LINEAR

    if h <= 4:
        # Classified 312x1 lines are repeated vertically first so they become
        # visible and safe for model/display processing.
        img = np.repeat(img, 220, axis=0)
        h, w = img.shape[:2]
        interpolation = cv2.INTER_NEAREST

    scale = float(output_size) / max(h, w)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(img, (new_w, new_h), interpolation=interpolation)

    canvas = np.zeros((output_size, output_size, 3), dtype=resized.dtype)
    y0 = (output_size - new_h) // 2
    x0 = (output_size - new_w) // 2
    canvas[y0:y0 + new_h, x0:x0 + new_w] = resized
    return canvas


# ---------------------------------------------------------------------------
# Mask generation and vertical movement calibration
# ---------------------------------------------------------------------------



def is_nir_camera_type(camera_type):
    return str(camera_type).lower() in ("mvimpact_nir", "mvimpact-nir", "nir")


def load_nir_camera_settings(path):
    if not path or not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return data.get("camera", data)
    except Exception as e:
        print(f"[NIR] Failed to load NIR settings YAML: {e}")
        return {}


def make_empty_nozzle_mask(n_nozzles):
    return np.zeros((640, int(n_nozzles)), dtype=np.uint8)


NIR_RAW_CHUNK_LINES = 6000


def _nir_raw_mode(camera):
    mode = str(getattr(camera, "input_mode", "nir")).lower()
    if mode in ("classified", "classified_line", "classified_line_312x1", "smart"):
        return "classified"
    return "spectral"


def _get_nir_raw_record_sample(camera):
    sample = getattr(camera, "last_raw_record_sample", None)
    if sample is None:
        sample = getattr(camera, "last_spectral_sample", None)
    if sample is None:
        sample = getattr(camera, "last_classified_line", None)
    if sample is None:
        return None
    return np.asarray(sample).copy()


def _flush_nir_raw_chunk(state, output_dir, reason="chunk"):
    if not state["buffer"]:
        return

    os.makedirs(output_dir, exist_ok=True)
    data = np.stack(state["buffer"], axis=0)
    timestamps = np.asarray(state["timestamps"], dtype=np.float64)

    first_ts = state["wall_start"] or datetime.datetime.now()
    last_ts = datetime.datetime.now()
    first_str = first_ts.strftime("%Y%m%d_%H%M%S_%f")[:-3]
    last_str = last_ts.strftime("%Y%m%d_%H%M%S_%f")[:-3]
    mode = state.get("mode") or "nir"
    chunk_idx = int(state.get("chunk_index", 0))

    base = f"nir_raw_{mode}_{first_str}_to_{last_str}_chunk{chunk_idx:05d}_{reason}"
    data_path = os.path.join(output_dir, base + ".npy")
    ts_path = os.path.join(output_dir, base + "_timestamps.npy")

    np.save(data_path, data)
    np.save(ts_path, timestamps)
    print(f"Saved NIR raw chunk: {data_path} shape={data.shape} dtype={data.dtype}")

    state["buffer"].clear()
    state["timestamps"].clear()
    state["wall_start"] = None
    state["chunk_index"] = chunk_idx + 1


def _append_nir_raw_line(state, camera, output_dir, chunk_lines=NIR_RAW_CHUNK_LINES):
    sample = _get_nir_raw_record_sample(camera)
    if sample is None:
        return

    mode = _nir_raw_mode(camera)
    if state.get("mode") != mode:
        _flush_nir_raw_chunk(state, output_dir, reason="modechange")
        state["mode"] = mode

    if state["wall_start"] is None:
        state["wall_start"] = datetime.datetime.now()

    state["buffer"].append(sample)
    state["timestamps"].append(float(getattr(camera, "last_line_timestamp", 0.0) or time.monotonic()))

    if len(state["buffer"]) >= int(chunk_lines):
        _flush_nir_raw_chunk(state, output_dir, reason="full")


DEFAULT_NIR_CLASS_COLORS = [
    (0, 0, 0),        # Background
    (255, 64, 64),    # Class 1
    (64, 220, 64),    # Class 2
    (64, 128, 255),   # Class 3
    (255, 220, 64),
    (220, 64, 255),
    (64, 220, 220),
    (255, 140, 64),
]


def normalise_nir_class_colors(raw_colors, n_classes):
    colors = []
    if isinstance(raw_colors, (list, tuple)):
        for item in raw_colors:
            try:
                if isinstance(item, str):
                    item = item.strip().lstrip('#')
                    if len(item) == 6:
                        rgb = tuple(int(item[i:i+2], 16) for i in (0, 2, 4))
                    else:
                        continue
                else:
                    rgb = tuple(int(v) for v in item[:3])
                colors.append(tuple(max(0, min(255, v)) for v in rgb))
            except Exception:
                continue

    while len(colors) < int(n_classes):
        colors.append(DEFAULT_NIR_CLASS_COLORS[len(colors) % len(DEFAULT_NIR_CLASS_COLORS)])
    return colors[:int(n_classes)]


def colorize_nir_class_buffer(class_buffer, class_colors):
    buf = np.asarray(class_buffer, dtype=np.uint8)
    if buf.ndim == 3:
        buf = buf[:, :, 0]
    colors = np.asarray(list(class_colors), dtype=np.uint8)
    if colors.ndim != 2 or colors.shape[1] != 3 or colors.shape[0] == 0:
        colors = np.asarray(DEFAULT_NIR_CLASS_COLORS, dtype=np.uint8)
    idx = np.clip(buf, 0, colors.shape[0] - 1)
    rgb = colors[idx]
    # Downstream display code assumes OpenCV BGR and converts to RGB.
    return rgb[:, :, ::-1].copy()


def classified_line_to_nozzle_mask(classified_line, target_classes, n_nozzles, height=1, beischuss=0):
    """Map one NIR class-id line to a nozzle activation mask.

    In NIR line-scan mode the executable ejection event is a single timestamped
    line, not a crop from the rolling display buffer.  The returned default
    shape is therefore 1 x N_NOZZLES.  A larger height is only for display.
    """
    line = np.asarray(classified_line, dtype=np.uint8).reshape(-1)
    if line.size == 0:
        return np.zeros((int(height), int(n_nozzles)), dtype=np.uint8)

    targets = list(target_classes)
    active_classes = [idx for idx, enabled in enumerate(targets) if int(enabled) == 1]
    if not active_classes:
        return np.zeros((int(height), int(n_nozzles)), dtype=np.uint8)

    active_line = np.isin(line, active_classes).astype(np.uint8) * 255
    nozzle_line = cv2.resize(
        active_line[None, :],
        (int(n_nozzles), 1),
        interpolation=cv2.INTER_NEAREST,
    )[0]

    beischuss = int(max(0, beischuss))
    if beischuss > 0 and np.any(nozzle_line):
        kernel = np.ones((1, beischuss * 2 + 1), dtype=np.uint8)
        nozzle_line = cv2.dilate(nozzle_line[None, :], kernel, iterations=1)[0]

    return np.repeat(nozzle_line[None, :], int(height), axis=0).astype(np.uint8)


def classified_rolling_buffer_to_nozzle_mask(classified_buffer, target_classes, n_nozzles, beischuss=0):
    """Map the NIR rolling class history to a display-only nozzle history.

    Rows are time/line history, columns are belt width/nozzles.  This fixes the
    90-degree mismatch caused by showing a repeated single-line mask beside the
    rolling NIR image.
    """
    buf = np.asarray(classified_buffer, dtype=np.uint8)
    if buf.ndim == 1:
        buf = buf[None, :]
    h = max(1, int(buf.shape[0]))

    targets = list(target_classes)
    active_classes = [idx for idx, enabled in enumerate(targets) if int(enabled) == 1]
    if not active_classes:
        return np.zeros((h, int(n_nozzles)), dtype=np.uint8)

    active = np.isin(buf, active_classes).astype(np.uint8) * 255
    mapped = cv2.resize(
        active,
        (int(n_nozzles), h),
        interpolation=cv2.INTER_NEAREST,
    )

    beischuss = int(max(0, beischuss))
    if beischuss > 0 and np.any(mapped):
        kernel = np.ones((1, beischuss * 2 + 1), dtype=np.uint8)
        mapped = cv2.dilate(mapped, kernel, iterations=1)
    return mapped.astype(np.uint8)

def drawNozzleMask(
    boxes,
    target_classes,
    VORSCHUSS,
    NACHSCHUSS,
    BEISCHUSS,
    THRESHOLD,
    N_NOZZLES,
    last_mask,
    vert_movement,
):
    """
    Build the nozzle activation mask and shift the previous mask downward by
    vert_movement pixels. No globals are used; last_mask is owned by produce().
    """
    nozzle_mask = np.zeros((640, N_NOZZLES), dtype=np.uint8)
    height, width = nozzle_mask.shape

    for box in boxes:
        if int(box.cls) not in target_classes:
            continue

        x1, y1, x2, y2 = box.xyxyn[0].cpu().numpy()
        x_min = max(int(x1 * width) - BEISCHUSS.value, 0)
        y_min = max(int(y1 * height) - NACHSCHUSS.value, 0)
        x_max = min(int(x2 * width) + BEISCHUSS.value, width)
        y_max = min(int(y2 * height) + VORSCHUSS.value, height)
        nozzle_mask[y_min:y_max, x_min:x_max] = 255

    # Preserve active region from the previous frame.
    np.maximum(last_mask, nozzle_mask, out=nozzle_mask)

    vert_movement = int(max(0, min(639, vert_movement)))
    if vert_movement > 0:
        vert_padding = np.zeros((vert_movement, N_NOZZLES), dtype=np.uint8)
        new_last_mask = np.vstack((vert_padding, nozzle_mask[:-vert_movement, :]))
    else:
        new_last_mask = nozzle_mask.copy()

    return nozzle_mask, new_last_mask


def update_vertical_movement_estimate(track_results, previous_track_centers, CALIBRATED_VERT_MOVEMENT):
    """
    Estimate vertical object displacement between consecutive frames using
    Ultralytics tracking IDs. This only writes the suggested value; it does not
    modify the active VERT_MOVEMENT setting.
    """
    try:
        boxes = track_results[0].boxes
        if boxes is None or boxes.id is None:
            return previous_track_centers

        ids = boxes.id.cpu().numpy().astype(int)
        xyxy = boxes.xyxy.cpu().numpy()

        current_centers = {}

        for obj_id, box in zip(ids, xyxy):
            x1, y1, x2, y2 = box
            cy = float((y1 + y2) / 2.0)
            current_centers[int(obj_id)] = cy

        movements = []

        for obj_id, cy in current_centers.items():
            if obj_id in previous_track_centers:
                dy = cy - previous_track_centers[obj_id]
                if -200.0 <= dy <= 200.0:
                    movements.append(dy)

        if movements:
            frame_median = float(np.median(movements))
            movement_history.append(frame_median)

            CALIBRATED_VERT_MOVEMENT.value = float(
                np.median(movement_history)
            )

        return current_centers

    except Exception as e:
        print(f"[Process-1] Error in vertical movement calibration: {e}")
        return previous_track_centers


# ---------------------------------------------------------------------------
# Producer / consumer pipeline
# ---------------------------------------------------------------------------

def produce(
    DISPLAY_QUEUE,
    MASK_QUEUE,
    TARGET_CLASSES,
    STOP_FLAG,
    CONF,
    IOU,
    VORSCHUSS,
    NACHSCHUSS,
    BEISCHUSS,
    THRESHOLD,
    RECORD_RAW,
    DRAW_BBOXES,
    CAMERA_TYPE,
    PFS_PATH,
    MODEL_PATH,
    MODEL_VERBOSE,
    N_NOZZLES,
    VERT_MOVEMENT,
    CALIBRATED_VERT_MOVEMENT,
    RUN_VERT_CALIBRATION,
    ROTATE,
    FLIP_H,
    FLIP_V,
    VIDEO_PATH,
    RAW_RECORDING_FPS,
    RECORDING_PATHS,
    FPS,
    USB_SETTINGS_PATH,
    MVIMPACT_NIR_SETTINGS_PATH,
    NIR_CLASSIFIER_PATH="",
    NIR_CLASSIFIER_KIND="SAM_PLACEHOLDER",
    NIR_CLASS_COLORS=None,
    NIR_RAW_CHUNK_LINES_VALUE=NIR_RAW_CHUNK_LINES,
):
    backup_image = cv2.imread("preheat_image.png")
    is_nir_camera = is_nir_camera_type(CAMERA_TYPE)

    last_save_time = 0
    save_name = "aufnahme"
    nir_raw_state = {"buffer": [], "timestamps": [], "mode": None, "chunk_index": 0, "wall_start": None}
    nir_recording_was_active = False

    # Cammera Reconnection
    camera_connected = False
    camera = None
    camera_fail_count = 0
    MAX_CAMERA_FAILS = 5
    last_reconnect_attempt = 0.0
    RECONNECT_COOLDOWN = 2.0
    
    # USB Camera Settings
    usb_settings = load_usb_camera_settings(USB_SETTINGS_PATH)
    usb_reticule = usb_settings.get("camera", {}).get("reticule", {})
    usb_crop_size = int(usb_reticule.get("crop_size", 640))
    usb_crop_offset_x = int(usb_reticule.get("offset_x", 0))
    usb_crop_offset_y = int(usb_reticule.get("offset_y", 0))
    
    
    last_calibration_state = None
    
    
    # Producer-local mask state. This replaces global LAST_MASK / vert_padding.
    last_mask = np.zeros((640, N_NOZZLES), dtype=np.uint8)
    previous_track_centers = {}

    try:
        camera = connectCamera(
            CAMERA_TYPE,
            PFS_PATH,
            FPS,
            VIDEO_PATH=VIDEO_PATH,
            USB_SETTINGS_PATH=USB_SETTINGS_PATH,
            MVIMPACT_NIR_SETTINGS_PATH=MVIMPACT_NIR_SETTINGS_PATH,
            NIR_CLASSIFIER_PATH=NIR_CLASSIFIER_PATH,
            NIR_CLASSIFIER_KIND=NIR_CLASSIFIER_KIND,
        )
        camera_connected = camera is not None
        if camera_connected and CAMERA_TYPE.lower() == "basler":
            camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
    except Exception as e:
        print(f"[Process-1] Error in Camera Connection: {e}")
        camera_connected = False
        camera = None

    model = None
    all_classes = []
    if is_nir_camera:
        print("[Process-1] NIR camera selected: skipping YOLO model load/preheat.")
    else:
        try:
            model = YOLO(MODEL_PATH, task="detect")
            all_classes = list(range(len(model.names)))
        except Exception as e:
            print(f"Error in Loading Model: {e}")
            print("Downloading yolov8n base model")
            model = YOLO("yolov8n.pt", task="detect")
            all_classes = list(range(len(model.names)))

        try:
            _ = model("preheat_image.png", conf=CONF.value, iou=IOU.value, verbose=MODEL_VERBOSE)
            print("Preheat Complete")
        except Exception as e:
            print(f"[Process-1] Error in Model Preheating: {e}")

    def reconnect_camera():
        nonlocal camera, camera_connected, last_reconnect_attempt
    
        now = time.monotonic()
        if now - last_reconnect_attempt < RECONNECT_COOLDOWN:
            return
    
        last_reconnect_attempt = now
        print("[Process-1] Attempting camera reconnect...")
    
        try:
            if camera is not None:
                try:
                    if CAMERA_TYPE.lower() == "basler":
                        if camera.IsGrabbing():
                            camera.StopGrabbing()
                        if camera.IsOpen():
                            camera.Close()
                    else:
                        camera.release()
                except Exception:
                    pass
    
            camera = connectCamera(
            CAMERA_TYPE,
            PFS_PATH,
            FPS,
            VIDEO_PATH=VIDEO_PATH,
            USB_SETTINGS_PATH=USB_SETTINGS_PATH,
            MVIMPACT_NIR_SETTINGS_PATH=MVIMPACT_NIR_SETTINGS_PATH,
            NIR_CLASSIFIER_PATH=NIR_CLASSIFIER_PATH,
            NIR_CLASSIFIER_KIND=NIR_CLASSIFIER_KIND,
        )
            camera_connected = camera is not None
    
            if camera_connected and CAMERA_TYPE.lower() == "basler":
                camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
                print("[Process-1] Basler reconnect successful.")
            elif camera_connected:
                print("[Process-1] Camera reconnect successful.")
            else:
                print("[Process-1] Camera reconnect failed.")
    
        except Exception as e:
            camera_connected = False
            camera = None
            print(f"[Process-1] Camera reconnect error: {e}")

    while not STOP_FLAG.is_set():
        try:
            if camera_connected and CAMERA_TYPE.lower() == "basler":
                if camera.IsGrabbing():
                    grab = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
                    if grab.GrabSucceeded():
                        img = grab.GetArray()
                        camera_fail_count = 0
                    grab.Release()
                else:
                    raise RuntimeError("Basler camera is not grabbing")
            
            elif CAMERA_TYPE.lower() == "simulated" and camera_connected:
                ret, img = camera.get_frame()
                if not ret or img is None:
                    img = backup_image
                    
            elif camera_connected and CAMERA_TYPE.lower() == "usb":
                ret, img = camera.read()
                if ret and img is not None:
                    img = center_crop_square(img,size=usb_crop_size,offset_x=usb_crop_offset_x,offset_y=usb_crop_offset_y,)
                else:
                    raise RuntimeError("USB camera frame grab failed")

            elif camera_connected and is_nir_camera:
                # Re-use the existing THRESHOLD runtime value as the live
                # background-angle threshold for the placeholder SAM classifier.
                try:
                    if hasattr(camera, "set_background_threshold"):
                        camera.set_background_threshold(float(THRESHOLD.value))
                    else:
                        camera.background_threshold = float(THRESHOLD.value)
                        clf = getattr(camera, "_classifier", None)
                        if hasattr(clf, "background_threshold"):
                            clf.background_threshold = float(THRESHOLD.value)
                except Exception:
                    pass
                ret, img = camera.read()
                if not ret or img is None:
                    raise RuntimeError("mvImpact NIR camera frame grab failed")
            else:
                img = backup_image

        except Exception as e:
            img = backup_image
            camera_fail_count += 1
        
            print(f"[Process-1] Error in Camera Acquisition ({camera_fail_count}/{MAX_CAMERA_FAILS}): {e}")
            print("Using Backup Image")
        
            if (CAMERA_TYPE.lower() == "basler" or is_nir_camera) and camera_fail_count >= MAX_CAMERA_FAILS:
                reconnect_camera()
                camera_fail_count = 0
        
        try:
            if ROTATE.value > 0:
                for i in range(ROTATE.value):
                    img = cv2.rotate(img,cv2.ROTATE_90_CLOCKWISE)
            if FLIP_H.value:
                img = cv2.flip(img,1,0)
            if FLIP_V.value:
                img = cv2.flip(img,0,1)
        except Exception as e:
            img = backup_image
            print(f"[Process-1] Error in Image Rotation: {e}")
            print("Using Backup Image")
        
        if is_nir_camera:
            # NIR frames are rolling classified buffers from the camera adapter.
            # They must not be sent through YOLO. The latest classified line is
            # mapped directly to the existing mask/nozzle pipeline using the GUI
            # class selection toggles.
            latest_line = getattr(camera, "last_classified_line", np.zeros((1,), dtype=np.uint8))
            line_timestamp = float(getattr(camera, "last_line_timestamp", 0.0) or time.monotonic())

            # Executable NIR ejection is one timestamped line.  Do not create a
            # 640-high mask and do not later crop it at DETECTION_POS.
            nozzle_line_mask = classified_line_to_nozzle_mask(
                latest_line,
                TARGET_CLASSES,
                N_NOZZLES,
                height=1,
                beischuss=BEISCHUSS.value,
            )

            rolling_classes = getattr(camera, "rolling_class_buffer", None)
            if rolling_classes is None:
                display_nozzle_mask = classified_line_to_nozzle_mask(
                    latest_line, TARGET_CLASSES, N_NOZZLES, height=img.shape[0], beischuss=BEISCHUSS.value
                )
            else:
                display_nozzle_mask = classified_rolling_buffer_to_nozzle_mask(
                    rolling_classes, TARGET_CLASSES, N_NOZZLES, beischuss=BEISCHUSS.value
                )
                img = colorize_nir_class_buffer(rolling_classes, NIR_CLASS_COLORS)

            try:
                MASK_QUEUE.put_nowait((nozzle_line_mask, line_timestamp))
            except Full:
                pass

            try:
                try:
                    DISPLAY_QUEUE.get_nowait()
                except Empty:
                    pass
                DISPLAY_QUEUE.put_nowait((display_nozzle_mask, img))
            except Full:
                pass
            except Exception as e:
                print(f"[Process-1] Error in NIR Display Queue Update: {e}")
                STOP_FLAG.set()

            try:
                if RECORD_RAW.value:
                    _append_nir_raw_line(
                        nir_raw_state,
                        camera,
                        RECORDING_PATHS.raw_dir,
                        chunk_lines=NIR_RAW_CHUNK_LINES_VALUE,
                    )
                    nir_recording_was_active = True
                elif nir_recording_was_active:
                    # Flush a partial chunk as soon as recording is switched off.
                    _flush_nir_raw_chunk(nir_raw_state, RECORDING_PATHS.raw_dir, reason="stopped")
                    nir_recording_was_active = False
            except Exception as e:
                print(f"[Process-1] Error in Saving of NIR Raw Chunk: {e}")

            continue

        try:
            # If switching between calibration and inference -> drain stale queues
            current_calibration_state = bool(RUN_VERT_CALIBRATION.value)
            
            if current_calibration_state != last_calibration_state:
            
                # Drain display queue
                while True:
                    try:
                        DISPLAY_QUEUE.get_nowait()
                    except Empty:
                        break
            
                # Drain mask queue
                while True:
                    try:
                        MASK_QUEUE.get_nowait()
                    except Empty:
                        break
            
                # Reset producer-local state
                last_mask.fill(0)
                previous_track_centers.clear()
            
                last_calibration_state = current_calibration_state
            
            if RUN_VERT_CALIBRATION.value:
                results = model.track(
                    img,
                    conf=CONF.value,
                    iou=IOU.value,
                    verbose=MODEL_VERBOSE,
                    agnostic_nms=True,
                    persist=True,
                )
                previous_track_centers = update_vertical_movement_estimate(
                    results,
                    previous_track_centers,
                    CALIBRATED_VERT_MOVEMENT,
                )
            else:
                results = model.predict(
                    img,
                    conf=CONF.value,
                    iou=IOU.value,
                    verbose=MODEL_VERBOSE,
                    agnostic_nms=True,
                )
        except Exception as e:            
            print(f"[Process-1] Error in Model Inference: {e}")
            print(f"{img.shape}")

        try:
            timestamp = time.monotonic()
            boxes = results[0].boxes
            target_classes = [cls for cls in all_classes if TARGET_CLASSES[cls] == 1]

            nozzle_mask, last_mask = drawNozzleMask(
                boxes,
                target_classes,
                VORSCHUSS,
                NACHSCHUSS,
                BEISCHUSS,
                THRESHOLD,
                N_NOZZLES,
                last_mask,
                VERT_MOVEMENT.value,
            )

            MASK_QUEUE.put_nowait((nozzle_mask, timestamp))
        except Full:
            pass
        except Exception as e:
            print(f"[Process-1] Error in Mask Creation: {e}")
            STOP_FLAG.set()
        
        try:
            if DRAW_BBOXES.value:
                display_image = results[0].plot()
            else:
                display_image = img
                
            # Display queue is live-preview only: keep newest frame, discard stale one.
            try:
                DISPLAY_QUEUE.get_nowait()
            except Empty:
                pass
            
            DISPLAY_QUEUE.put_nowait((nozzle_mask, display_image))
        except Full:
            pass
        except Exception as e:
            print(f"[Process-1] Error in Display Queue Update: {e}")
            if camera_connected:
                if CAMERA_TYPE.lower() == "basler":
                    camera.Close()
                else:
                    camera.release()
            STOP_FLAG.set()
            
        # Record RAW Camera Images
        try:
            raw_fps = max(0.1, RAW_RECORDING_FPS.value)
            current_time = time.time()
        
            if RECORD_RAW.value and current_time > last_save_time + (1.0 / raw_fps):
                os.makedirs(RECORDING_PATHS.raw_dir, exist_ok=True)
        
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        
                cv2.imwrite(
                    os.path.join(
                        RECORDING_PATHS.raw_dir,
                        f"{save_name}_{timestamp}.jpg"
                    ),
                    img
                )
        
                print(f"Saved Raw Image: {timestamp}")
                last_save_time = current_time
        
        except Exception as e:
            print(f"[Process-1] Error in Saving of Raw Image: {e}")

    # End of While Loop -> Cleaning up after STOP FLAG has been set
    if is_nir_camera:
        try:
            _flush_nir_raw_chunk(nir_raw_state, RECORDING_PATHS.raw_dir, reason="shutdown")
        except Exception as e:
            print(f"[Process-1] Error flushing final NIR raw chunk: {e}")

    if camera_connected:
        if CAMERA_TYPE.lower() == "basler":
            camera.Close()
        else:
            camera.release()
    STOP_FLAG.set()


def createMask(MASK_QUEUE, NOZZLE_ACTIVATION_QUEUE, STOP_FLAG, DELAY, DETECTION_POS,RUN_VERT_CALIBRATION):
    last_timestamp = None

    while not STOP_FLAG.is_set():
        try:
            if last_timestamp is None:
                last_mask, last_timestamp = MASK_QUEUE.get(timeout=0.05)

            now = time.monotonic()
            remaining = (last_timestamp + DELAY.value) - now
            if remaining > 0:
                STOP_FLAG.wait(min(remaining, 0.01))
                continue

            # Area-camera/YOLO mode still uses a y crop around DETECTION_POS.
            # NIR line-scan mode enqueues a 1 x N_NOZZLES line mask whose
            # timestamp is the line acquisition/classification time; for that
            # case there is no meaningful y-position to crop.
            if np.asarray(last_mask).shape[0] <= 1:
                activation_mask = np.asarray(last_mask, dtype=np.uint8).reshape(1, -1).copy()
            else:
                h = last_mask.shape[0]
                y0 = max(0, min(h - 1, DETECTION_POS - 6))
                y1 = max(y0 + 1, min(h, DETECTION_POS + 6))
                activation_mask = last_mask[y0:y1, :].copy()

            if not RUN_VERT_CALIBRATION.value:
                NOZZLE_ACTIVATION_QUEUE.put_nowait(activation_mask)
            
            last_timestamp = None

        except Empty:
            STOP_FLAG.wait(0.01)
        except Full:
            STOP_FLAG.wait(0.01)
        except Exception as e:
            print(f"[Process-1] Error in Creating Mask: {e}")
            STOP_FLAG.set()


def consume(MASK_QUEUE, STOP_FLAG, DELAY, NOZZLE_CONTROL_FUNCTION, DETECTION_POS,RUN_VERT_CALIBRATION):
    NOZZLE_ACTIVATION_QUEUE = queue.Queue(maxsize=500)
    create_mask_thread = threading.Thread(
        target=createMask,
        args=(MASK_QUEUE, NOZZLE_ACTIVATION_QUEUE, STOP_FLAG, DELAY, DETECTION_POS,RUN_VERT_CALIBRATION),
    )
    control_nozzle_thread = threading.Thread(
        target=NOZZLE_CONTROL_FUNCTION,
        args=(NOZZLE_ACTIVATION_QUEUE, STOP_FLAG),
    )

    create_mask_thread.start()
    control_nozzle_thread.start()

    create_mask_thread.join()
    control_nozzle_thread.join()


# ---------------------------------------------------------------------------
# Main startup
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from STARTUP_SETTINGS_GUI import startup_config_gui

    startup_cfg = startup_config_gui()

    CAMERA_TYPE = startup_cfg["CAMERA_TYPE"]
    PFS_PATH = startup_cfg["PFS_PATH"]
    CONNECTION_TYPE = startup_cfg["CONNECTION_TYPE"].lower()
    MODEL_PATH = startup_cfg["MODEL_PATH"]
    MODEL_VERBOSE = startup_cfg["MODEL_VERBOSE"]
    VIDEO_PATH = startup_cfg["VIDEO_PATH"]
    SCALEABLE_UI = startup_cfg.get("SCALEABLE_UI", False)
    FPS_value = int(startup_cfg.get("FPS", 30))
    USB_SETTINGS_PATH = startup_cfg.get("USB_CAMERA_SETTINGS_PATH", "")
    MVIMPACT_NIR_SETTINGS_PATH = startup_cfg.get("MVIMPACT_NIR_SETTINGS_PATH", "")
    NIR_CLASSIFIER_PATH = startup_cfg.get("NIR_CLASSIFIER_PATH", "")
    NIR_CLASSIFIER_KIND = startup_cfg.get("NIR_CLASSIFIER_KIND", "SAM_PLACEHOLDER")
    try:
        NIR_RAW_CHUNK_LINES_VALUE = max(1, int(startup_cfg.get("NIR_RAW_CHUNK_LINES", NIR_RAW_CHUNK_LINES)))
    except Exception:
        NIR_RAW_CHUNK_LINES_VALUE = NIR_RAW_CHUNK_LINES
    
    if CAMERA_TYPE not in ("USB", "Basler", "SIMULATED", "MVIMPACT_NIR"):
        raise ValueError("CAMERA_TYPE must be USB, SIMULATED, Basler or MVIMPACT_NIR.")

    if CONNECTION_TYPE == "udp":
        N_NOZZLES = 128
        NOZZLE_CONTROL_FUNCTION = nozzle_control_UDP
    elif CONNECTION_TYPE == "serial":
        N_NOZZLES = 8
        NOZZLE_CONTROL_FUNCTION = nozzle_control_ARDUINO
    elif CONNECTION_TYPE == "modbus":
        N_NOZZLES = 80
        NOZZLE_CONTROL_FUNCTION = nozzle_control_MODBUS
    elif CONNECTION_TYPE == "simulated":
        N_NOZZLES = 10
        NOZZLE_CONTROL_FUNCTION = nozzle_control_SIMULATED
    else:
        raise ValueError("CONNECTION_TYPE must be UDP, SERIAL, or MODBUS.")

    DETECTION_POS = 600

    multiprocessing.set_start_method("spawn", force=True)

    if is_nir_camera_type(CAMERA_TYPE):
        nir_settings = load_nir_camera_settings(MVIMPACT_NIR_SETTINGS_PATH)
        n_nir_classes = int(nir_settings.get("synthetic_classes", nir_settings.get("classes", 4)))
        n_nir_classes = max(1, n_nir_classes)
        configured_names = nir_settings.get("class_names", [])
        if isinstance(configured_names, list) and configured_names:
            MODEL_NAMES = {i: str(configured_names[i]) if i < len(configured_names) else f"NIR Class {i}" for i in range(n_nir_classes)}
        else:
            MODEL_NAMES = {0: "Background"}
            MODEL_NAMES.update({i: f"NIR Class {i}" for i in range(1, n_nir_classes)})
        ALL_CLASSES = list(range(len(MODEL_NAMES)))
        print("[Main] NIR camera selected: skipping YOLO model load for UI.")
        print(f"[Main] NIR classifier kind: {NIR_CLASSIFIER_KIND}")
        if NIR_CLASSIFIER_PATH:
            print(f"[Main] NIR classifier path: {NIR_CLASSIFIER_PATH}")
        print(f"[Main] NIR raw recording chunk size: {NIR_RAW_CHUNK_LINES_VALUE} lines")
    else:
        try:
            model_temp = YOLO(MODEL_PATH, task="detect")
            if isinstance(model_temp.names, dict):
                MODEL_NAMES = dict(model_temp.names)
            else:
                MODEL_NAMES = {i: name for i, name in enumerate(model_temp.names)}
            ALL_CLASSES = list(range(len(MODEL_NAMES)))
        except Exception as e:
            print(f"Error in Loading Model for UI. Error: {e}")
            model_temp = YOLO("yolov8n.pt", task="detect")
            if isinstance(model_temp.names, dict):
                MODEL_NAMES = dict(model_temp.names)
            else:
                MODEL_NAMES = {i: name for i, name in enumerate(model_temp.names)}
            ALL_CLASSES = list(range(len(MODEL_NAMES)))

    runtime_cfg = load_runtime_config()
    manager = multiprocessing.Manager()
    targets = runtime_cfg.get(
        "TARGET_CLASSES",
        np.zeros(len(MODEL_NAMES)).astype(int).tolist()
    )
    
    # Make sure target list matches current model class count
    if len(targets) != len(MODEL_NAMES):
        targets = np.zeros(len(MODEL_NAMES)).astype(int).tolist()
    
    TARGET_CLASSES = multiprocessing.Array("i", targets)

    nir_settings_for_colors = load_nir_camera_settings(MVIMPACT_NIR_SETTINGS_PATH) if is_nir_camera_type(CAMERA_TYPE) else {}
    default_color_cfg = nir_settings_for_colors.get("class_colors", [])
    runtime_color_cfg = runtime_cfg.get("NIR_CLASS_COLORS", default_color_cfg)
    nir_color_list = normalise_nir_class_colors(runtime_color_cfg, len(MODEL_NAMES))
    NIR_CLASS_COLORS = manager.list([tuple(c) for c in nir_color_list])
    
    CONF = multiprocessing.Value("d", runtime_cfg.get("CONF", 0.10))
    IOU = multiprocessing.Value("d", runtime_cfg.get("IOU", 0.00))
    DELAY = multiprocessing.Value("d", runtime_cfg.get("DELAY", 0.000))
    
    VORSCHUSS = multiprocessing.Value("i", runtime_cfg.get("VORSCHUSS", 0))
    NACHSCHUSS = multiprocessing.Value("i", runtime_cfg.get("NACHSCHUSS", 0))
    BEISCHUSS = multiprocessing.Value("i", runtime_cfg.get("BEISCHUSS", 0))

    # For YOLO mode this value is kept for backward compatibility.
    # For NIR mode it is the raw ALU background threshold used by SAM:
    #     mean(spectrum) < THRESHOLD -> background class 0
    if is_nir_camera_type(CAMERA_TYPE):
        nir_threshold_default = float(
            nir_settings_for_colors.get(
                "background_threshold",
                nir_settings_for_colors.get("sam_background_threshold", 300.0)
            )
        )
        runtime_threshold = float(runtime_cfg.get("THRESHOLD", nir_threshold_default))
        # Older UI versions stored 0..1 values. Treat those as stale and reset
        # to the NIR YAML default so the control starts in useful ALU units.
        if 0.0 <= runtime_threshold <= 1.0:
            runtime_threshold = nir_threshold_default
    else:
        runtime_threshold = float(runtime_cfg.get("THRESHOLD", 0.1))
    THRESHOLD = multiprocessing.Value("d", runtime_threshold)
    
    FPS = multiprocessing.Value("i", FPS_value)
    
    VERT_MOVEMENT = multiprocessing.Value(
        "i",
        runtime_cfg.get("VERT_MOVEMENT", 40)
    )
    
    CALIBRATED_VERT_MOVEMENT = multiprocessing.Value(
        "d",
        runtime_cfg.get("CALIBRATED_VERT_MOVEMENT", 40.0)
    )
    
    RUN_VERT_CALIBRATION = multiprocessing.Value(
        "b",
        runtime_cfg.get("RUN_VERT_CALIBRATION", False)
    )
    
    STOP_FLAG = multiprocessing.Event()
    
    RECORD_RAW = multiprocessing.Value(
        "b",
        runtime_cfg.get("RECORD_RAW", False)
    )
    
    DRAW_BBOXES = multiprocessing.Value(
        "b",
        runtime_cfg.get("DRAW_BBOXES", True)
    )
    
    ROTATE = multiprocessing.Value("i", runtime_cfg.get("ROTATE", 0))
    FLIP_H = multiprocessing.Value("b", runtime_cfg.get("FLIP_H", False))
    FLIP_V = multiprocessing.Value("b", runtime_cfg.get("FLIP_V", False))

    # Recording Variables
    RAW_RECORDING_FPS = multiprocessing.Value(
    "d",
    runtime_cfg.get("RAW_RECORDING_FPS", 2.0)
    )
    
    GUI_RECORDING_FPS = multiprocessing.Value(
        "d",
        runtime_cfg.get("GUI_RECORDING_FPS", 2.0)
    )
    
    RECORDING_PATHS = manager.Namespace()
    RECORDING_PATHS.raw_dir = runtime_cfg.get("RAW_RECORDING_DIR", "Recordings_Camera")
    RECORDING_PATHS.gui_dir = runtime_cfg.get("GUI_RECORDING_DIR", "Recordings_GUI")


    # QUEUES
    MASK_QUEUE = multiprocessing.Queue(maxsize=500)
    DISPLAY_QUEUE = multiprocessing.Queue(maxsize=1)

    producer = multiprocessing.Process(
        target=produce,
        args=(
            DISPLAY_QUEUE,
            MASK_QUEUE,
            TARGET_CLASSES,
            STOP_FLAG,
            CONF,
            IOU,
            VORSCHUSS,
            NACHSCHUSS,
            BEISCHUSS,
            THRESHOLD,
            RECORD_RAW,
            DRAW_BBOXES,
            CAMERA_TYPE,
            PFS_PATH,
            MODEL_PATH,
            MODEL_VERBOSE,
            N_NOZZLES,
            VERT_MOVEMENT,
            CALIBRATED_VERT_MOVEMENT,
            RUN_VERT_CALIBRATION,
            ROTATE,
            FLIP_H,
            FLIP_V,
            VIDEO_PATH,
            RAW_RECORDING_FPS,
            RECORDING_PATHS,
            FPS,
            USB_SETTINGS_PATH,
            MVIMPACT_NIR_SETTINGS_PATH,
            NIR_CLASSIFIER_PATH,
            NIR_CLASSIFIER_KIND,
            NIR_CLASS_COLORS,
            NIR_RAW_CHUNK_LINES_VALUE,
        ),
    )

    displayer = multiprocessing.Process(
        target=display,
        args=(
            DISPLAY_QUEUE,
            STOP_FLAG,
            DELAY,
            TARGET_CLASSES,
            CONF,
            IOU,
            VORSCHUSS,
            NACHSCHUSS,
            BEISCHUSS,
            THRESHOLD,
            RECORD_RAW,
            DRAW_BBOXES,
            ALL_CLASSES,
            MODEL_NAMES,
            DETECTION_POS,
            VERT_MOVEMENT,
            CALIBRATED_VERT_MOVEMENT,
            RUN_VERT_CALIBRATION,
            ROTATE,
            FLIP_H,
            FLIP_V,
            RAW_RECORDING_FPS,
            GUI_RECORDING_FPS,
            RECORDING_PATHS,
            FPS,
            SCALEABLE_UI,
            is_nir_camera_type(CAMERA_TYPE),
            NIR_CLASSIFIER_KIND,
            NIR_CLASS_COLORS,
        ),
    )

    consumer = multiprocessing.Process(
        target=consume,
        args=(MASK_QUEUE, STOP_FLAG, DELAY, NOZZLE_CONTROL_FUNCTION, DETECTION_POS,RUN_VERT_CALIBRATION),
    )

    processes = [producer, displayer, consumer]
    p_names = ["producer", "displayer", "consumer"]

    for process in processes:
        process.start()

    try:
        while not STOP_FLAG.is_set():
            time.sleep(0.05)
    except KeyboardInterrupt:
        STOP_FLAG.set()

    for i, process in enumerate(processes):
        process.join(timeout=10)
        if process.is_alive():
            print(f"Warning: Process: \"{p_names[i]}\" did not terminate gracefully")
            process.terminate()
        else:
            print(f"Process: \"{p_names[i]}\" terminated gracefully")

    runtime_cfg_out = {
    "TARGET_CLASSES": list(TARGET_CLASSES),

    "CONF": CONF.value,
    "IOU": IOU.value,
    "DELAY": DELAY.value,

    "VORSCHUSS": VORSCHUSS.value,
    "NACHSCHUSS": NACHSCHUSS.value,
    "BEISCHUSS": BEISCHUSS.value,
    "THRESHOLD": THRESHOLD.value,

    "VERT_MOVEMENT": VERT_MOVEMENT.value,
    "CALIBRATED_VERT_MOVEMENT": CALIBRATED_VERT_MOVEMENT.value,
    "RUN_VERT_CALIBRATION": False,

    "RECORD_RAW": False,
    "DRAW_BBOXES": bool(DRAW_BBOXES.value),

    "ROTATE": ROTATE.value,
    "FLIP_H": bool(FLIP_H.value),
    "FLIP_V": bool(FLIP_V.value),
    
    "RAW_RECORDING_FPS": RAW_RECORDING_FPS.value,
    "GUI_RECORDING_FPS": GUI_RECORDING_FPS.value,
    "FPS": FPS.value,
    "RAW_RECORDING_DIR": RECORDING_PATHS.raw_dir,
    "GUI_RECORDING_DIR": RECORDING_PATHS.gui_dir,
    "NIR_CLASS_COLORS": [list(c) for c in list(NIR_CLASS_COLORS)],
    }
    
    save_runtime_config(runtime_cfg_out)
    
    print("Runtime settings saved.")
    print("Main process shutdown complete")
