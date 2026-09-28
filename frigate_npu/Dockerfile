# Frigate ZMQ detector backend on the XDNA 1 NPU. Base: npu-ryzenai (AMD
# Ryzen AI 1.7.1 runtime, built by yolonas_int8/npu_bench.sh). Build context =
# repo root (see docker-compose.yml); the model is mounted at runtime.
FROM npu-ryzenai
RUN /opt/ryzenai/bin/uv pip install --python /opt/ryzenai/bin/python pyzmq
COPY yolonas_int8/nas_raw.py frigate_npu/npu_zmq_detector.py /app/
ENV XILINX_XRT=/usr \
    XLNX_VART_FIRMWARE=/opt/ryzenai/lib/python3.12/site-packages/voe-4.0-linux_x86_64/xclbins/phoenix/4x4.xclbin \
    VITISAI_EP_JSON_CONFIG=/opt/ryzenai/lib/python3.12/site-packages/voe-4.0-linux_x86_64/vaip_config.json \
    MODEL=/models/yolo_nas_l_320_xint8_adaround.onnx \
    CACHE_DIR=/cache
EXPOSE 5555
CMD ["bash", "-c", "source /opt/ryzenai/bin/activate && exec python /app/npu_zmq_detector.py"]
