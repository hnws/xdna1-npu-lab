"""Benchmark an XINT8 YOLO-NAS model on the XDNA 1 NPU via the Vitis AI EP.

Runs inside the npu-vitisai container (see npu_bench.sh). Session options are
those of Frigate PR #24470's `vitisai` detector. Reports:
  * how much of the graph the Vitis AI partitioner put on the NPU,
  * detections vs the FP32 model (data/eval.npy, held-out Frigate snapshots),
  * single-stream latency and sustained detections/s,
  * APU package power (amdgpu PPT sensor) above idle.
"""
import glob
import json
import os
import statistics
import sys
import threading
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import nas_raw  # noqa: E402
from evaluate import THRESH, compare, run as run_cpu, REF, NAMES  # noqa: E402


class PowerMeter:
    def __init__(self):
        paths = [p for p in glob.glob("/sys/class/drm/card*/device/hwmon/hwmon*/power1_input")
                 if open(p.replace("_input", "_label")).read().strip() == "PPT"]
        self.path = paths[0] if paths else None
        self.samples, self._stop = [], threading.Event()

    def __enter__(self):
        self.samples, _ = [], self._stop.clear()
        if self.path:
            self._t = threading.Thread(target=self._run, daemon=True)
            self._t.start()
        return self

    def _run(self):
        while not self._stop.is_set():
            self.samples.append(int(open(self.path).read()) / 1e6)
            time.sleep(0.1)

    def __exit__(self, *exc):
        self._stop.set()
        if self.path:
            self._t.join()

    @property
    def mean(self):
        return statistics.fmean(self.samples) if self.samples else float("nan")


def npu_session(model):
    report = f"/tmp/vitisai_report_{os.getpid()}.json"
    os.environ["XLNX_ONNX_EP_REPORT_FILE"] = report
    popts = {
        "target": "X1",
        "xclbin": os.environ["XLNX_VART_FIRMWARE"],
        "cache_dir": str(HERE / "vaip_cache"),
        "cache_key": Path(model).stem,
        "enable_cache_file_io_in_mem": "0",
        "xlnx_enable_py3_round": 0,  # required on Phoenix, else silent CPU fallback
    }
    so = ort.SessionOptions()
    so.log_severity_level = 3
    t0 = time.time()
    s = ort.InferenceSession(model, sess_options=so, providers=["VitisAIExecutionProvider"],
                             provider_options=[popts])
    print(f"session ready in {time.time() - t0:.0f}s (first time compiles; cached in vaip_cache/)")
    npu, total, ops = 0, 0, []
    try:
        for dev in json.load(open(report)).get("deviceStat", []):
            if dev.get("name") == "all":
                total = dev.get("nodeNum", 0)
            elif dev.get("name") == "NPU":
                npu = dev.get("nodeNum", 0)
                ops = [o.strip(":") for o in dev.get("supportedOpType", [])]
        print(f"partition: {npu}/{total} nodes on the NPU, ops there: {sorted(set(ops))}")
    except FileNotFoundError:
        print("partition: no report (cache hit without saved report)")
    return s


def sustained(sess, x, threads, seconds, idle):
    stop, counts = threading.Event(), [0] * threads

    def work(i):
        while not stop.is_set():
            sess.run(None, {"images": x})
            counts[i] += 1
    meter = PowerMeter()
    with meter:
        ts = [threading.Thread(target=work, args=(i,)) for i in range(threads)]
        t0 = time.perf_counter()
        [t.start() for t in ts]
        time.sleep(seconds)
        stop.set()
        [t.join() for t in ts]
        dt = time.perf_counter() - t0
    fps = sum(counts) / dt
    return fps, meter.mean, (meter.mean - idle) / fps


def main():
    model = sys.argv[1]
    images = np.load(HERE / "data/eval.npy")
    sess = npu_session(model)

    # accuracy vs FP32 (CPU) on held-out snapshots
    ref, _ = run_cpu(REF, images)
    test = [nas_raw.decode(sess.run(None, {"images": nas_raw.preprocess(im)}), conf=0.3) for im in images]
    tp, fn, fp, ds, pc = compare(ref, test)
    print(f"accuracy vs FP32 on {len(images)} held-out snapshots: kept {tp}/{tp + fn} "
          f"({tp / max(tp + fn, 1):.1%}) detections >= {THRESH}, {fp} extra, score shift {np.mean(ds):+.3f}; "
          + ", ".join(f"{NAMES.get(c, c)} {k}/{n}" for c, (k, n) in sorted(pc.items(), key=lambda kv: -kv[1][1])[:4]))

    # latency
    x = nas_raw.preprocess(images[0])
    for _ in range(10):
        sess.run(None, {"images": x})
    lat = []
    for _ in range(100):
        t0 = time.perf_counter()
        sess.run(None, {"images": x})
        lat.append((time.perf_counter() - t0) * 1000)
    t0 = time.perf_counter()
    for im in images[:50]:
        nas_raw.decode(sess.run(None, {"images": nas_raw.preprocess(im)}))
    e2e = (time.perf_counter() - t0) / 50 * 1000
    print(f"latency: median {statistics.median(lat):.1f} ms, p90 {np.percentile(lat, 90):.1f} ms "
          f"(NPU inference); {e2e:.1f} ms incl. preprocess+decode+NMS")

    idle_m = PowerMeter()
    with idle_m:
        time.sleep(5)
    idle = idle_m.mean
    for th in (1, 2):
        fps, w, j = sustained(sess, x, th, 20, idle)
        print(f"sustained, {th} thread(s): {fps:.1f} detections/s, package {w:.1f} W "
              f"(idle {idle:.1f} W), {j:.2f} J/detection above idle")


if __name__ == "__main__":
    main()
