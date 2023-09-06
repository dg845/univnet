import argparse

import numpy as np
import torch
from omegaconf import OmegaConf

from utils.stft import TacotronSTFT


def main(args):
    checkpoint = torch.load(args.checkpoint_path, map_location="cpu")
    if args.config is not None:
        hp = OmegaConf.load(args.config)
    else:
        hp = OmegaConf.create(checkpoint['hp_str'])
    
    device = "cuda" if args.use_cuda else "cpu"

    # print(f"Sampling rate: {hp.audio.sampling_rate}")
    # print(f"Filter length: {hp.audio.filter_length}")
    
    stft = TacotronSTFT(
        filter_length=hp.audio.filter_length,
        hop_length=hp.audio.hop_length,
        win_length=hp.audio.win_length,
        n_mel_channels=hp.audio.n_mel_channels,
        sampling_rate=hp.audio.sampling_rate,
        mel_fmin=hp.audio.mel_fmin,
        mel_fmax=hp.audio.mel_fmax,
        center=False,
        device=device,
    )

    # Load raw audio
    with open(args.audio, 'rb') as audio_file:
        audio = np.load(audio_file, allow_pickle=False)
    
    # print(f"Audio shape: {audio.shape}")
    # print(f"Seconds of audio: {audio.shape[0] / hp.audio.sampling_rate}")
    # print(f"Audio sample: {audio[:30]}")

    # Convert audio into a torch tensor and batch
    # audio = torch.tensor(audio[:1600], device=device)
    audio = torch.tensor(audio, device=device)
    audio = audio.unsqueeze(0).float()

    # print(f"Audio shape: {audio.shape}")
    # print(f"Audio dtype: {audio.dtype}")

    # Get MEL spectrogram.
    with torch.no_grad():
        mel_spectrogram = stft.mel_spectrogram(audio)
    
    mel_spectrogram = mel_spectrogram.detach().cpu()
    # print(f"MEL spectrogram: {mel_spectrogram}")
    # print(f"MEL spectrogram shape: {mel_spectrogram.shape}")

    # Get mean and stddev of mel_spectrogram.
    print(f"Mean of MEL spectrogram: {torch.mean(mel_spectrogram)}")
    print(f"Std Dev of MEL spectrogram: {torch.std(mel_spectrogram)}")
    
    # Get MEL spectrogram expected slice
    expected_slice = mel_spectrogram[0, 0, :30]
    print(f"MEL spectrogram slice: {expected_slice}")

    # Serialize full mel_spectrogram as numpy array
    mel_np = mel_spectrogram.numpy()
    np.save("test_mel_spectrogram.npy", mel_np, allow_pickle=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-c',
        '--config',
        type=str,
        default=None,
        help="yaml file for config. will use hp_str from checkpoint if not given.",
    )
    parser.add_argument(
        '-p',
        '--checkpoint_path',
        type=str,
        required=True,
        help="path of checkpoint pt file for evaluation",
    )
    parser.add_argument(
        '-a',
        '--audio',
        type=str,
        required=True,
        help="Path to raw audio wavefrom as serialized NumPy array in .npy format.",
    )
    parser.add_argument(
        '--use_cuda',
        action="store_true",
        help="whether to use cuda for testing",
    )
    
    args = parser.parse_args()

    main(args)