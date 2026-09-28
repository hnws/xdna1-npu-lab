#!/usr/bin/env python3
"""Frigate ZMQ detector backend running YOLO-NAS (XINT8) on the XDNA 1 NPU.

Frigate 0.18's built-in `zmq` detector sends each preprocessed input tensor
over ZeroMQ (REQ/REP) and expects Frigate's (20, 6) float32 detection array
back: rows [class_id, score, y_min, x_min, y_max, x_max], coordinates 0..1.
This server answers those requests with AMD's Vitis AI runtime on the NPU, so
Frigate itself needs no custom image -- only a detector config change.

Frigate config (the model file must exist in Frigate's /config/model_cache,
but inference happens here):

    detectors:
      npu:
        type: zmq
        endpoint: tcp://frigate-npu:5555
    model:
      path: /config/model_cache/yolo_nas_l_320_xint8_adaround.onnx
      model_type: ssd
      width: 320
      height: 320
      input_tensor: nchw
      input_pixel_format: rgb
      input_dtype: float
      labelmap_path: /labelmap/coco-80.txt

Environment: MODEL (path to the XINT8 raw-head ONNX), BIND (default
tcp://*:5555), CACHE_DIR (Vitis AI compile cache), CONF (0.4), IOU (0.5).
"""
import json
import logging
import os
import sys
import time

import numpy as np
import onnxruntime as ort
import zmq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nas_raw  # noqa: E402  (raw-head decode + NMS)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("frigate-npu")

MODEL = os.environ.get("MODEL", "/models/yolo_nas_l_320_xint8_adaround.onnx")
BIND = os.environ.get("BIND", "tcp://*:5555")
CACHE_DIR = os.environ.get("CACHE_DIR", "/cache")
CONF = float(os.environ.get("CONF", "0.4"))
IOU = float(os.environ.get("IOU", "0.5"))
MAX_DET = 20
SIZE = nas_raw.SIZE


def npu_session():
    report = f"/tmp/vitisai_report_{os.getpid()}.json"
    os.environ["XLNX_ONNX_EP_REPORT_FILE"] = report
    popts = {
        "target": "X1",
        "xclbin": os.environ["XLNX_VART_FIRMWARE"],
        "cache_dir": CACHE_DIR,
        "cache_key": os.path.splitext(os.path.basename(MODEL))[0],
        "enable_cache_file_io_in_mem": "0",
        "xlnx_enable_py3_round": 0,  # required on Phoenix, else silent CPU fallback
    }
    so = ort.SessionOptions()
    so.log_severity_level = 3
    t0 = time.time()
    sess = ort.InferenceSession(MODEL, sess_options=so, providers=["VitisAIExecutionProvider"],
                                provider_options=[popts])
    log.info("loaded %s in %.0fs", MODEL, time.time() - t0)
    if os.path.exists(report):
        stats = {d.get("name"): d for d in json.load(open(report)).get("deviceStat", [])}
        npu, total = stats.get("NPU", {}).get("nodeNum", 0), stats.get("all", {}).get("nodeNum", 0)
        log.info("partition: %d/%d nodes on the NPU", npu, total)
        if not npu:
            raise SystemExit("nothing placed on the NPU -- refusing to serve detections from the CPU")
    return sess


def to_model_input(header, payload):
    """Frigate's preprocessed tensor -> (1, 3, 320, 320) float32 RGB 0..1."""
    x = np.frombuffer(payload, dtype=np.dtype(header["dtype"])).reshape(header["shape"])
    if x.ndim == 3:
        x = x[None]
    if x.shape[-1] == 3 and x.shape[1] != 3:  # nhwc -> nchw
        x = x.transpose(0, 3, 1, 2)
    x = x.astype(np.float32)
    if x.max() > 1.5:  # uint8 or float_denorm input
        x /= 255.0
    return np.ascontiguousarray(x)


def to_frigate(dets):
    """nas_raw.decode rows [x1, y1, x2, y2, score, cls] (320 px) -> Frigate (20, 6)."""
    out = np.zeros((MAX_DET, 6), np.float32)
    for i, (x1, y1, x2, y2, score, cls) in enumerate(dets[:MAX_DET]):
        box = np.clip(np.array([y1, x1, y2, x2]) / SIZE, 0.0, 1.0)
        out[i] = [cls, score, *box]
    return out


def main():
    sess = npu_session()
    sess.run(None, {"images": np.zeros((1, 3, SIZE, SIZE), np.float32)})  # warm-up
    sock = zmq.Context().socket(zmq.REP)
    sock.bind(BIND)
    log.info("serving Frigate zmq detector requests on %s", BIND)
    n, busy, t_log = 0, 0.0, time.time()
    while True:
        frames = sock.recv_multipart()
        try:
            header = json.loads(frames[0])
            if header.get("model_request") or header.get("model_data"):
                # The model is fixed on this side; tell Frigate it is ready so
                # it never tries to transfer its file.
                log.info("Frigate asked for model %s; serving %s", header.get("model_name"),
                         os.path.basename(MODEL))
                sock.send_multipart([json.dumps({"model_available": True, "model_loaded": True,
                                                 "model_saved": True}).encode()])
                continue
            t0 = time.perf_counter()
            x = to_model_input(header, frames[1])
            dets = nas_raw.decode(sess.run(None, {"images": x}), conf=CONF, iou=IOU, max_det=MAX_DET)
            out = to_frigate(dets)
            busy += time.perf_counter() - t0
            sock.send_multipart([json.dumps({"shape": [MAX_DET, 6], "dtype": "float32"}).encode(),
                                 out.tobytes()])
            n += 1
        except Exception as exc:  # never leave the REQ side hanging
            log.exception("request failed: %s", exc)
            sock.send(np.zeros((MAX_DET, 6), np.float32).tobytes())
        if time.time() - t_log > 300:
            dt = time.time() - t_log
            log.info("%d detections in %.0fs (%.1f/s), %.1f ms average", n, dt, n / dt,
                     busy / max(n, 1) * 1000)
            n, busy, t_log = 0, 0.0, time.time()


if __name__ == "__main__":
    main()
