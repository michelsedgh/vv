"""Benchmark custom TorchScript biLSTM vs cuDNN LSTM on DPRNN dimensions."""
import torch
import torch.jit as jit
import time
import numpy as np
from torch import Tensor
from typing import Tuple, List

device = torch.device("cuda")
torch.backends.cudnn.benchmark = True


# === Custom TorchScript LSTM Cell with fused gates ===
@jit.script
def lstm_cell(x: Tensor, hx: Tensor, cx: Tensor,
              w_ih: Tensor, w_hh: Tensor,
              b_ih: Tensor, b_hh: Tensor) -> Tuple[Tensor, Tensor]:
    gates = torch.mm(x, w_ih.t()) + b_ih + torch.mm(hx, w_hh.t()) + b_hh
    ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)
    ingate = torch.sigmoid(ingate)
    forgetgate = torch.sigmoid(forgetgate)
    cellgate = torch.tanh(cellgate)
    outgate = torch.sigmoid(outgate)
    cy = forgetgate * cx + ingate * cellgate
    hy = outgate * torch.tanh(cy)
    return hy, cy


@jit.script
def lstm_layer_fwd(input: Tensor, h0: Tensor, c0: Tensor,
                   w_ih: Tensor, w_hh: Tensor,
                   b_ih: Tensor, b_hh: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    seq_len = input.size(0)
    hx = h0
    cx = c0
    outputs: List[Tensor] = []
    for i in range(seq_len):
        hx, cx = lstm_cell(input[i], hx, cx, w_ih, w_hh, b_ih, b_hh)
        outputs.append(hx)
    return torch.stack(outputs, 0), hx, cx


@jit.script
def bilstm_fwd(input: Tensor, h0_f: Tensor, c0_f: Tensor,
               h0_b: Tensor, c0_b: Tensor,
               w_ih_f: Tensor, w_hh_f: Tensor, b_ih_f: Tensor, b_hh_f: Tensor,
               w_ih_b: Tensor, w_hh_b: Tensor, b_ih_b: Tensor, b_hh_b: Tensor
               ) -> Tensor:
    out_f, _, _ = lstm_layer_fwd(input, h0_f, c0_f,
                                 w_ih_f, w_hh_f, b_ih_f, b_hh_f)
    out_b, _, _ = lstm_layer_fwd(input.flip(0), h0_b, c0_b,
                                 w_ih_b, w_hh_b, b_ih_b, b_hh_b)
    return torch.cat([out_f, out_b.flip(0)], dim=-1)


def bench(fn, name, n=20):
    torch.cuda.synchronize()
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(n):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    m = np.median(times)
    print(f"{name:50s}: {m:.1f}ms")
    return m


def main():
    batch, seq_len, input_size, hidden_size = 102, 100, 128, 128

    # cuDNN LSTM
    cudnn_lstm = torch.nn.LSTM(
        input_size, hidden_size, batch_first=True, bidirectional=True
    ).cuda().half()
    x_bf = torch.randn(batch, seq_len, input_size, device=device, dtype=torch.float16)

    # Extract weights for custom TorchScript LSTM
    w_ih_f = cudnn_lstm.weight_ih_l0
    w_hh_f = cudnn_lstm.weight_hh_l0
    b_ih_f = cudnn_lstm.bias_ih_l0
    b_hh_f = cudnn_lstm.bias_hh_l0
    w_ih_b = cudnn_lstm.weight_ih_l0_reverse
    w_hh_b = cudnn_lstm.weight_hh_l0_reverse
    b_ih_b = cudnn_lstm.bias_ih_l0_reverse
    b_hh_b = cudnn_lstm.bias_hh_l0_reverse

    x_sf = x_bf.transpose(0, 1).contiguous()  # (seq, batch, input)
    h0 = torch.zeros(batch, hidden_size, device=device, dtype=torch.float16)
    c0 = torch.zeros(batch, hidden_size, device=device, dtype=torch.float16)

    print("=== LSTM Benchmark (batch=102, seq=100, hidden=128, bidir) ===\n")

    with torch.inference_mode():
        t_cudnn = bench(lambda: cudnn_lstm(x_bf), "cuDNN LSTM")

    with torch.no_grad():
        t_ts = bench(
            lambda: bilstm_fwd(
                x_sf, h0, c0, h0, c0,
                w_ih_f, w_hh_f, b_ih_f, b_hh_f,
                w_ih_b, w_hh_b, b_ih_b, b_hh_b,
            ),
            "TorchScript biLSTM",
        )

    # Verify correctness
    with torch.no_grad():
        cudnn_out, _ = cudnn_lstm(x_bf)
        ts_out = bilstm_fwd(
            x_sf, h0, c0, h0, c0,
            w_ih_f, w_hh_f, b_ih_f, b_hh_f,
            w_ih_b, w_hh_b, b_ih_b, b_hh_b,
        )
        ts_out_bf = ts_out.transpose(0, 1)
        diff = (cudnn_out - ts_out_bf).abs().max().item()
        print(f"\nMax diff cuDNN vs TorchScript: {diff:.6f}")
        print(f"Speedup: {t_cudnn / t_ts:.2f}x")

    # Also test inter dimensions (batch=100, seq=102)
    print("\n=== LSTM Benchmark (batch=100, seq=102, hidden=128, bidir) ===\n")
    batch2, seq2 = 100, 102
    cudnn_lstm2 = torch.nn.LSTM(
        input_size, hidden_size, batch_first=True, bidirectional=True
    ).cuda().half()
    x_bf2 = torch.randn(batch2, seq2, input_size, device=device, dtype=torch.float16)
    x_sf2 = x_bf2.transpose(0, 1).contiguous()
    h02 = torch.zeros(batch2, hidden_size, device=device, dtype=torch.float16)

    w2 = [getattr(cudnn_lstm2, f"weight_ih_l0"),
          getattr(cudnn_lstm2, f"weight_hh_l0"),
          getattr(cudnn_lstm2, f"bias_ih_l0"),
          getattr(cudnn_lstm2, f"bias_hh_l0"),
          getattr(cudnn_lstm2, f"weight_ih_l0_reverse"),
          getattr(cudnn_lstm2, f"weight_hh_l0_reverse"),
          getattr(cudnn_lstm2, f"bias_ih_l0_reverse"),
          getattr(cudnn_lstm2, f"bias_hh_l0_reverse")]

    with torch.inference_mode():
        t2_cudnn = bench(lambda: cudnn_lstm2(x_bf2), "cuDNN LSTM")

    with torch.no_grad():
        t2_ts = bench(
            lambda: bilstm_fwd(x_sf2, h02, h02, h02, h02, *w2),
            "TorchScript biLSTM",
        )
    print(f"Speedup: {t2_cudnn / t2_ts:.2f}x")


if __name__ == "__main__":
    main()
