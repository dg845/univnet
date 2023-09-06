import argparse

import numpy as np
import torch
from omegaconf import OmegaConf

from model.generator import Generator
from utils.stft import TacotronSTFT


def main(args):
    checkpoint = torch.load(args.checkpoint_path, map_location="cpu")
    if args.config is not None:
        hp = OmegaConf.load(args.config)
    else:
        hp = OmegaConf.create(checkpoint['hp_str'])
    
    device = "cuda" if args.use_cuda else "cpu"

    noise_dim = hp.gen.noise_dim

    model = Generator(hp)
    saved_state_dict = checkpoint['model_g']
    new_state_dict = {}
    
    for k, v in saved_state_dict.items():
        try:
            new_state_dict[k] = saved_state_dict['module.' + k]
        except:
            new_state_dict[k] = v
    model.load_state_dict(new_state_dict)
    model.eval(inference=True)
    if args.use_cuda:
        model.cuda()
    
    # print(model)
    # for name, param in model.res_stack[0].kernel_predictor.input_conv.named_parameters():
    #     print(f"Kernel Predictor input_conv param {name}: {param.data}")
    #     print(f"Kernel Predictor input_conv param {name} shape: {param.data.shape}")

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
    
    generator = torch.random.manual_seed(args.seed)

    # Load raw audio waveform and convert to MEL spectrogram
    with open(args.audio, 'rb') as audio_file:
        audio = np.load(audio_file, allow_pickle=False)

    # Convert audio into a torch tensor and batch
    audio = torch.tensor(audio, device=device)
    audio = audio.unsqueeze(0).float()
    audio = audio.repeat(args.num_samples, 1)

    # Get the MEL spectrogram
    with torch.no_grad():
        mel = stft.mel_spectrogram(audio)

    # print(f"MEL spectrogram: {mel}")
    # print(f"MEL spectrogram shape: {mel.shape}")

    # Create noise waveform
    noise_shape = (args.num_samples, noise_dim, mel.shape[-1])
    noise = torch.randn(noise_shape, generator=generator)
    # print(f"Noise waveform: {noise}")
    # print(f"Noise waveform shape: {noise.shape}")

    with torch.no_grad():
        if len(mel.shape) == 2:
            mel = mel.unsqueeze(0)
            noise = noise.unsqueeze(0)
        
        if args.use_cuda:
            mel = mel.cuda()
            noise = noise.cuda()

        audio = model(mel, noise)
        audio = audio.cpu().detach()
        # print(f"Audio: {audio}")
        # print(f"Audio shape: {audio.shape}")

        # Get mean and stddev of output audio.
        print(f"Mean of audio: {torch.mean(audio)}")
        print(f"Std Dev of audio: {torch.std(audio)}")

        # Get test expected slice
        expected_slice = audio[-1, -1, -9:].squeeze(0)
        print(f"Expected slice: {expected_slice}")
        print(f"Expected slice shape: {expected_slice.shape}")


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
        '-n',
        '--num_samples',
        type=int,
        default=1,
        help="number of test samples to generate",
    )
    parser.add_argument(
        '-s',
        '--seed',
        type=int,
        default=0,
        help="rng seed for reproducibility",
    )
    parser.add_argument(
        '--use_cuda',
        action="store_true",
        help="whether to use cuda for testing",
    )
    
    args = parser.parse_args()

    main(args)
