import argparse
import random

import torch
from omegaconf import OmegaConf

from model.generator import Generator


global_rng = random.Random()


# Modified from transformers.tests.test_modeling_common.floats_tensor
# Don't create tensor on torch_device
def floats_tensor(shape, scale=1.0, rng=None, name=None):
    """Creates a random float32 tensor"""
    if rng is None:
        rng = global_rng

    total_dims = 1
    for dim in shape:
        total_dims *= dim

    values = []
    for _ in range(total_dims):
        values.append(rng.random() * scale)

    return torch.tensor(data=values, dtype=torch.float).view(shape).contiguous()


def main(args):
    checkpoint = torch.load(args.checkpoint_path, map_location="cpu")
    if args.config is not None:
        hp = OmegaConf.load(args.config)
    else:
        hp = OmegaConf.create(checkpoint['hp_str'])
    
    mel_channels = hp.audio.n_mel_channels
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
    
    generator = torch.random.manual_seed(args.seed)

    # Create noise waveform
    if args.num_samples == 1:
        noise_shape = (noise_dim, args.noise_length)
    else:
        noise_shape = (args.num_samples, noise_dim, args.noise_length)
    noise = torch.randn(noise_shape, generator=generator)
    # print(f"Noise waveform: {noise}")
    # print(f"Noise waveform shape: {noise.shape}")

    # Create random mel spectrogram
    if args.num_samples == 1:
        mel_shape = (mel_channels, args.noise_length)
    else:
        mel_shape = (args.num_samples, mel_channels, args.noise_length)
    mel = floats_tensor(mel_shape, scale=1.0, rng=random.Random(args.seed))
    # print(f"MEL spectrogram: {mel}")
    # print(f"MEL spectrogram shape: {mel.shape}")

    with torch.no_grad():
        if len(mel.shape) == 2:
            mel = mel.unsqueeze(0)
            noise = noise.unsqueeze(0)
        
        if args.use_cuda:
            mel = mel.cuda()
            noise = noise.cuda()

        audio = model(mel, noise)
        audio = audio.cpu().detach()
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
        '-n',
        '--num_samples',
        type=int,
        default=1,
        help="number of test samples to generate",
    )
    parser.add_argument(
        '-l',
        '--noise_length',
        type=int,
        default=10,
        help="length in frames of input noise and spectrogram"
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
