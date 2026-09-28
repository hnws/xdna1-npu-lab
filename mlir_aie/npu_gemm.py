"""Shape-agnostic bf16 GEMM for the Ryzen AI NPU (XDNA 1 / npu1).

Why this exists: the XDNA 1 driver keeps at most 6 hardware contexts (loaded
NPU programs) alive, and a stock mlir-aie GEMM compiles one program per
(M, K, N). A network with dozens of layer shapes then reloads a program on
nearly every call, which costs far more than the math.

This design (derived from mlir-aie's whole_array example, Apache-2.0 WITH
LLVM-exception) keeps everything shape-specific out of the compiled program:
the per-core loop counts (output tiles per core, K blocks per tile) are
runtime parameters (RTPs) written by the instruction stream, and the tensor
walk is also in the instruction stream. So one program per tile config
(m, k, n) serves every (M, K, N); only the small per-shape instruction
stream changes between calls, and the loaded program stays resident.

Layout: C[M, N] (f32) = A[M, K] (bf16, row-major) @ B[K, N] (bf16, row-major),
on a 4x4 grid of AIE cores. Needs M % (4*m) == 0, K % k == 0, N % (4*n) == 0.
"""

from pathlib import Path

import numpy as np

import aie.iron as iron
from aie.helpers.taplib import TensorTiler2D
from aie.iron import (
    Buffer,
    CompileTime,
    In,
    ObjectFifo,
    Out,
    Program,
    Runtime,
    TaskGroup,
    Worker,
    WorkerRuntimeBarrier,
    kernels,
    str_to_dtype,
)
from aie.iron.controlflow import range_
from aie.utils.npukernel import NPUKernel

ROWS, COLS = 4, 4


@iron.jit
def ugemm(
    A: In,
    B: In,
    C: Out,
    *,
    M: CompileTime[int],
    K: CompileTime[int],
    N: CompileTime[int],
    m: CompileTime[int],
    k: CompileTime[int],
    n: CompileTime[int],
):
    dtype_in, dtype_out = str_to_dtype("bf16"), np.float32
    assert M % (m * ROWS) == 0 and K % k == 0 and N % (n * COLS) == 0, (M, K, N)

    mm = kernels.mm(dim_m=m, dim_k=k, dim_n=n, input_dtype=dtype_in, output_dtype=dtype_out)
    zero = mm.zero
    r, s, t = mm.mac_dims
    assert m % r == 0 and k % s == 0 and n % t == 0

    A_l2_ty = np.ndarray[(m * k,), np.dtype[dtype_in]]
    B_l2_ty = np.ndarray[(k * n,), np.dtype[dtype_in]]
    C_l2_ty = np.ndarray[(m * n * ROWS,), np.dtype[dtype_out]]
    A_l1_ty = np.ndarray[(m, k), np.dtype[dtype_in]]
    B_l1_ty = np.ndarray[(k, n), np.dtype[dtype_in]]
    C_l1_ty = np.ndarray[(m, n), np.dtype[dtype_out]]

    # ---- data movement: identical to whole_array for n_aie_cols == 4 ----
    A_l3l2, A_l2l1 = [], []
    for i in range(ROWS):
        f = ObjectFifo(A_l2_ty, name=f"A_L3L2_{i}", depth=2)
        A_l3l2.append(f)
        A_l2l1.extend(f.cons().split(
            [0], obj_types=[A_l1_ty], names=[f"A_L2L1_{i}"],
            dims_to_stream=[[(m // r, r * k), (k // s, s), (r, k), (s, 1)]]))
    B_l3l2, B_l2l1, C_l2l3 = [], [], []
    C_l1l2 = [[] for _ in range(ROWS)]
    for col in range(COLS):
        f = ObjectFifo(B_l2_ty, name=f"B_L3L2_{col}", depth=2)
        B_l3l2.append(f)
        B_l2l1.append(f.cons().forward(
            obj_type=B_l1_ty, name=f"B_L2L1_{col}",
            dims_to_stream=[(k // s, s * n), (n // t, t), (s, n), (t, 1)]))
        c = ObjectFifo(C_l2_ty, name=f"C_L2L3_{col}", depth=2,
                       dims_to_stream=[(m // r, r * n), (r, t), (n // t, r * t), (t, 1)])
        C_l2l3.append(c)
        parts = c.prod().join([m * n * i for i in range(ROWS)], obj_types=[C_l1_ty] * ROWS,
                              names=[f"C_L1L2_{col}_{row}" for row in range(ROWS)],
                              depths=[2] * ROWS)
        for row in range(ROWS):
            C_l1l2[row].append(parts[row])

    # ---- compute: loop counts come from RTPs, not from the compiled program ----
    # rtp[0] = output tiles this core computes, rtp[1] = K // k.
    rtps = [[Buffer(np.ndarray[(2,), np.dtype[np.int32]], name=f"rtp_{row}_{col}",
                    initial_value=np.zeros(2, np.int32), use_write_rtp=True)
             for col in range(COLS)] for row in range(ROWS)]
    barriers = [[WorkerRuntimeBarrier() for _ in range(COLS)] for _ in range(ROWS)]

    def core_fn(in_a, in_b, out_c, zero_k, mm_k, rtp, barrier):
        barrier.wait_for_value(1)
        n_tiles = rtp[0]
        k_blocks = rtp[1]
        for _ in range_(n_tiles):
            c = out_c.acquire(1)
            zero_k(c)
            for _ in range_(k_blocks):
                a = in_a.acquire(1)
                b = in_b.acquire(1)
                mm_k(a, b, c)
                in_a.release(1)
                in_b.release(1)
            out_c.release(1)
        barrier.release_with_value(1)

    workers = [Worker(core_fn, [A_l2l1[row].cons(), B_l2l1[col].cons(), C_l1l2[row][col].prod(),
                                zero, mm, rtps[row][col], barriers[row][col]], stack_size=0xD00)
               for row in range(ROWS) for col in range(COLS)]

    # ---- per-shape runtime sequence (goes in the instruction stream) ----
    tiles_per_core = (M // m) * (N // n) // (ROWS * COLS)
    A_tiles = TensorTiler2D.group_tiler((M, K), (m, k), (1, K // k),
                                        pattern_repeat=N // n // COLS, prune_step=False)
    B_tiles = TensorTiler2D.step_tiler((K, N), (k, n), tile_group_repeats=(K // k, N // n // COLS),
                                       tile_group_steps=(1, COLS), tile_group_col_major=True,
                                       prune_step=False)
    # Output is drained per column in groups of 2 row blocks (1 if the count is odd).
    row_blocks = M // m // ROWS
    per_drain = 2 if row_blocks % 2 == 0 else 1
    C_tiles = TensorTiler2D.step_tiler((M, N), (m * ROWS, n),
                                       tile_group_repeats=(per_drain, N // n // COLS),
                                       tile_group_steps=(1, COLS), prune_step=False)

    def sequence(A, B, C, A_hs, B_hs, C_hs):
        for row in range(ROWS):
            for col in range(COLS):
                rtps[row][col][0] = tiles_per_core
                rtps[row][col][1] = K // k
                barriers[row][col].set(1)
        tg = TaskGroup()
        for g in range(row_blocks // per_drain):
            for col in range(COLS):
                C_hs[col].drain(C, tap=C_tiles[g * COLS + col], wait=True, group=tg)
                for tile_row in range(per_drain):
                    A_hs[col].fill(A, tap=A_tiles[(g * per_drain + tile_row) * ROWS + col], group=tg)
                    B_hs[col].fill(B, tap=B_tiles[col], group=tg)
            if g > 0:  # keep two groups in flight
                tg.finish()
                tg = TaskGroup()
        tg.finish()

    rt = Runtime(sequence, [np.ndarray[(M * K,), np.dtype[dtype_in]],
                            np.ndarray[(K * N,), np.dtype[dtype_in]],
                            np.ndarray[(M * N,), np.dtype[dtype_out]],
                            [f.prod() for f in A_l3l2], [f.prod() for f in B_l3l2],
                            [f.cons() for f in C_l2l3]])
    return Program(iron.get_current_device(), rt, workers=workers).resolve_program()


class UGemm:
    """Runs C = A @ B for any compatible (M, K, N) with ONE resident program.

    The program (xclbin) is compiled once for the tile config; each new shape
    only compiles its instruction stream, which is then paired with that
    shared program.
    """

    def __init__(self, m=32, k=64, n=32):
        self.m, self.k, self.n = m, k, n
        self.align = (m * ROWS, k, n * COLS)
        self.xclbin = None
        self.kernels = {}

    def kernel(self, M, K, N):
        key = (M, K, N)
        if key not in self.kernels:
            xclbin, insts = ugemm.specialize(M=M, K=K, N=N, m=self.m, k=self.k, n=self.n).compile()
            if self.xclbin is None:
                self.xclbin = Path(xclbin)
            self.kernels[key] = NPUKernel(self.xclbin, insts, kernel_name="MLIR_AIE")
        return self.kernels[key]

    def prepare(self, shapes):
        """Compile the instruction streams for all shapes up front.

        Compiling a new shape after this process has started running NPU work
        made that shape's first call fail with EINVAL from the driver, so
        compile everything before the first run.
        """
        for M, K, N in shapes:
            self.kernel(M, K, N)

    def __call__(self, a_t, b_t, c_t, M, K, N):
        self.kernel(M, K, N)(a_t, b_t, c_t)

    def bind(self, a_t, b_t, c_t, M, K, N):
        """Return a zero-argument function that runs this GEMM on these buffers.

        Skips the generic runtime's per-call checks and buffer lookups: the
        loaded kernel, instruction buffer and buffer objects are resolved once.
        Host writes (via tensor.overwrite()/mutate()) are still flushed, and
        c_t.numpy() still reads the result back from the device.
        """
        from aie.utils import DefaultNPURuntime
        import pyxrt

        handle = DefaultNPURuntime.load(self.kernel(M, K, N))
        tensors = (a_t, b_t, c_t)
        for t in tensors:
            t.to("npu")
        bos = [t.buffer_object() for t in tensors]
        insts_bo = handle.insts_bo
        if not insts_bo:
            insts_t = DefaultNPURuntime._tensor_class(
                handle.insts, flags=pyxrt.bo.cacheable, group_id=handle.kernel.group_id(1),
                xrt_device=DefaultNPURuntime._device)
            insts_bo = insts_t.buffer_object()
            tensors = tensors + (insts_t,)  # keep it alive
        nbytes = handle.insts.nbytes
        kernel = handle.kernel
        done = pyxrt.ert_cmd_state.ERT_CMD_STATE_COMPLETED

        def run():
            for t in tensors[:3]:
                t.to("npu")  # flush host-written ranges; no-op if clean
            r = kernel(3, insts_bo, nbytes, *bos).wait()
            if r != done:
                raise RuntimeError(f"NPU kernel returned {r}")
        run.tensors = tensors
        return run
