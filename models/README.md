# Models Directory
Place your custom YOLO smoke detection weights here as `smoke.pt` (`models/smoke.pt`).
If present, `DualModePlumeDetector` will automatically load it as Primary Mode with `imgsz=480` and `vid_stride=3`.
If absent, the detector automatically falls back to Zero-Weight OpenCV ROI + MOG2 Background Subtraction with 100% reliability.
