import torch
import torch.nn.functional as F
import torchaudio
import pandas as pd

# STFT function with real and imaginary concatenation
def stft_torch(signal, n_fft=512, hop_length=128, win_length=512, concat_dim=None):
    """
    Compute the STFT and return real|imag concatenated for MeanFlow (input_dim = 2 * (n_fft//2+1)).

    For **batched** audio (B, time), stacks along channel dim → (B, 2F, T).
    For **unbatched** (time,) or (F, T) after stft as 2D (F, T), stacks along dim 0 → (2F, T), then unsqueeze(0).

    Using concat_dim=0 with (B, F, T) wrongly doubles the batch (2, F, T) and breaks UDiT torch.cat(enroll, x).
    """
    window = torch.hann_window(win_length).to(signal.device)
    spec = torch.stft(
        signal,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        return_complex=True,
    )
    spec_real = spec.real
    spec_imag = spec.imag
    if concat_dim is None:
        if spec_real.dim() == 2:
            cdim = 0
        elif spec_real.dim() == 3:
            cdim = 1
        else:
            raise ValueError(f"stft_torch: unexpected real spec dim {spec_real.dim()} shape={spec_real.shape}")
    else:
        cdim = concat_dim
    spec_concat = torch.cat([spec_real, spec_imag], dim=cdim)
    if spec_concat.dim() == 2:
        spec_concat = spec_concat.unsqueeze(0)
    return spec_concat

# ISTFT function with real and imaginary reconstruction
def istft_torch(spec_concat, n_fft=512, hop_length=128, win_length=512, length=None):
    """
    Reconstruct the signal from the concatenated real and imaginary parts of the STFT.

    Parameters:
        spec_concat (torch.Tensor): Concatenated real and imaginary parts of the STFT, shape (batch, freq*2, time).
        n_fft (int): FFT size.
        hop_length (int): Hop length.
        win_length (int): Window length.
        length (int, optional): Desired output length of the reconstructed signal.

    Returns:
        torch.Tensor: Reconstructed signal of shape (batch, time).
    """
    # Split the concatenated tensor into real and imaginary parts
    freq = n_fft // 2 + 1
    spec_real = spec_concat[:, :freq, :]  # Shape: (batch, freq, time)
    spec_imag = spec_concat[:, freq:, :]  # Shape: (batch, freq, time)
    spec = torch.complex(spec_real, spec_imag)  # Reconstruct complex tensor

    window = torch.hann_window(win_length).to(spec.device)
    recon_signal = torch.istft(spec,
                               n_fft=n_fft,
                               hop_length=hop_length,
                               win_length=win_length,
                               window=window,
                               length=length)
    return recon_signal