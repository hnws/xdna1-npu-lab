"""Run Frigate's own LocalObjectDetector + zmq plugin against the NPU sidecar.

  docker run --rm --network frigate_default --entrypoint python3 -w /opt/frigate \
    -e PYTHONPATH=/opt/frigate -v $PWD/../models:/config/model_cache:ro \
    -v $PWD/test_frigate_side.py:/w/test.py:ro -v /path/to/image.jpg:/w/image.jpg:ro \
    ghcr.io/blakeblackshear/frigate:<your tag> /w/test.py
"""
import time
import numpy as np
from PIL import Image
from frigate.detectors.detector_config import ModelConfig
from frigate.detectors.plugins.zmq_ipc import ZmqDetectorConfig
from frigate.object_detection.base import LocalObjectDetector

model = ModelConfig(path="/config/model_cache/yolo_nas_l_320_xint8_adaround.onnx", model_type="ssd",
                    width=320, height=320, input_tensor="nchw", input_pixel_format="rgb",
                    input_dtype="float", labelmap_path="/labelmap/coco-80.txt")
cfg = ZmqDetectorConfig(type="zmq", endpoint="tcp://frigate-npu:5555", model=model)
det = LocalObjectDetector(detector_config=cfg)

frame = np.asarray(Image.open("/w/image.jpg").convert("RGB").resize((320, 320)))[None]
print("detections:", [(l, round(s, 2), tuple(round(float(v), 3) for v in b)) for l, s, b in det.detect(frame, 0.5)])
for _ in range(10):
    det.detect(frame)
lat = []
for _ in range(200):
    t0 = time.perf_counter(); det.detect(frame); lat.append((time.perf_counter() - t0) * 1000)
print(f"Frigate-side round trip: median {np.median(lat):.1f} ms, p99 {np.percentile(lat, 99):.1f} ms, "
      f"{1000 / np.mean(lat):.0f} detections/s max")
