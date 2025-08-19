import os
import urllib.request
import mediapipe as mp

det_model_url = "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/latest/blaze_face_short_range.tflite"
det_model_path = "blaze_face_short_range.tflite"

Face_detector = None

def ensure_face_model(path=det_model_path, url=det_model_url):
    os.path.exists(path) or urllib.request.urlretrieve(url, path)
    return path

def init_face_detector():
    global Face_detector
    ensure_face_model()
    BaseOptions = mp.tasks.BaseOptions
    FaceDetector = mp.tasks.vision.FaceDetector
    FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
    VisionRunningMode = mp.tasks.vision.RunningMode
    options = FaceDetectorOptions(
        base_options=BaseOptions(model_asset_path=det_model_path, delegate=BaseOptions.Delegate.CPU),
        running_mode=VisionRunningMode.IMAGE,
        min_detection_confidence=0.5
    )
    getattr(Face_detector, "close", lambda: None)()
    Face_detector = FaceDetector.create_from_options(options)
    return Face_detector

def detect_faces_image(img_rgb):
    det = Face_detector or init_face_detector()
    h, w = img_rgb.shape[:2]
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
    result = det.detect(mp_image)
    dets = result.detections or []
    boxes = {i: ((int(d.bounding_box.origin_x), int(d.bounding_box.origin_y)),
                 (int(d.bounding_box.origin_x + d.bounding_box.width), int(d.bounding_box.origin_y + d.bounding_box.height)))
             for i, d in enumerate(dets)}
    scores = {i: float((d.categories and d.categories[0].score) or 0.0) for i, d in enumerate(dets)}
    return boxes, scores

def close_face_detector():
    global Face_detector
    getattr(Face_detector, "close", lambda: None)()
    Face_detector = None
