"""YOLO-NAS raw-head model + decode, in the format of Frigate PR #24470's
`vitisai` detector (input RGB float 0..1 NCHW; 6 outputs: per stride 8/16/32
a cls [1,80,H,W] and a reg [1,68,H,W] map; decode and NMS on the CPU).

make_raw() cuts it from the Frigate-exported yolo_nas_l_320.onnx (which takes
BGR uint8 and has decode+NMS inside), so no retraining or new weights needed.
decode() is a port of the PR's _decode_nas + class-aware NMS.
"""
import numpy as np
import onnx
from onnx import numpy_helper

SIZE = 320
PRE_OUT = "/pre_process/pre_process.1/Mul_output_0"  # float BGR 0..1


def make_raw(src, dst):
    m = onnx.load(src)
    heads = sorted(n.output[0] for n in m.graph.node
                   if n.op_type == "Conv" and ("cls_pred" in n.name or "reg_pred" in n.name))
    assert len(heads) == 6, heads
    ex = onnx.utils.Extractor(onnx.shape_inference.infer_shapes(m))
    raw = ex.extract_model([PRE_OUT], heads)
    # Rename the input to "images" and make the stem conv read RGB instead of BGR.
    for n in raw.graph.node:
        n.input[:] = ["images" if i == PRE_OUT else i for i in n.input]
    raw.graph.input[0].name = "images"
    stem = next(n for n in raw.graph.node if n.op_type == "Conv" and "images" in n.input)
    init = {i.name: i for i in raw.graph.initializer}
    w = numpy_helper.to_array(init[stem.input[1]])
    init[stem.input[1]].CopyFrom(numpy_helper.from_array(np.ascontiguousarray(w[:, ::-1]), stem.input[1]))
    onnx.checker.check_model(raw)
    onnx.save(raw, dst)
    return heads


def preprocess(rgb_320):
    """(320,320,3) uint8 RGB -> (1,3,320,320) float 0..1, what Frigate feeds the detector."""
    return np.ascontiguousarray(rgb_320.transpose(2, 0, 1)[None], dtype=np.float32) / 255.0


def _anchors():
    levels = []
    for stride in (8, 16, 32):
        s = SIZE // stride
        yy, xx = np.meshgrid(np.arange(s, dtype=np.float32) + 0.5,
                             np.arange(s, dtype=np.float32) + 0.5, indexing="ij")
        levels.append((stride, xx.ravel(), yy.ravel()))
    return levels


_LEVELS = _anchors()


def _softmax_expect(logits, bins):
    logits = logits - logits.max(axis=1, keepdims=True)
    p = np.exp(logits)
    p /= p.sum(axis=1, keepdims=True)
    return (p * np.arange(bins, dtype=np.float32)[None, :, None]).sum(axis=1)


def _nms(boxes, scores, iou):
    order = np.argsort(-scores)
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        r = order[1:]
        xx1 = np.maximum(boxes[i, 0], boxes[r, 0]); yy1 = np.maximum(boxes[i, 1], boxes[r, 1])
        xx2 = np.minimum(boxes[i, 2], boxes[r, 2]); yy2 = np.minimum(boxes[i, 3], boxes[r, 3])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        area = lambda b: (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])
        o = inter / (area(boxes[i]) + area(boxes[r]) - inter + 1e-9)
        order = r[o <= iou]
    return np.array(keep, dtype=np.int64)


def decode(outputs, conf=0.4, iou=0.4, max_det=20):
    """Raw head outputs -> (N, 6) rows [x1, y1, x2, y2, score, class] in 320-px units."""
    by_size = {}
    for arr in outputs:
        a = arr[0]
        by_size.setdefault(a.shape[-1] * a.shape[-2], {})["cls" if a.shape[0] == 80 else "reg"] = a
    boxes_l, scores_l = [], []
    for stride, gx, gy in _LEVELS:
        lv = by_size[(SIZE // stride) ** 2]
        cls, reg = lv["cls"], lv["reg"]
        n = cls.shape[1] * cls.shape[2]
        d = _softmax_expect(reg.reshape(4, 17, n), 17) * stride
        ax, ay = gx * stride, gy * stride
        boxes_l.append(np.stack([ax - d[0], ay - d[1], ax + d[2], ay + d[3]], axis=1))
        scores_l.append(1 / (1 + np.exp(-cls.reshape(80, n).T)))
    boxes, sc = np.concatenate(boxes_l), np.concatenate(scores_l)
    best, cls_id = sc.max(1), sc.argmax(1)
    k = best > conf
    boxes, best, cls_id = boxes[k], best[k], cls_id[k]
    if not len(best):
        return np.zeros((0, 6), np.float32)
    keep = _nms(boxes + cls_id[:, None] * 8192.0, best, iou)[:max_det]
    return np.concatenate([boxes[keep], best[keep, None], cls_id[keep, None]], 1).astype(np.float32)
