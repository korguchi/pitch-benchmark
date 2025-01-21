import torch
import torchaudio
import random
import os
from abc import ABC, abstractmethod
from typing import List


class Noise(ABC):
    def __init__(self, snr_range=[15, 30], train=True):
        if isinstance(snr_range, (int, float)):
            self.snr_range = [snr_range, snr_range]
        elif (
            isinstance(snr_range, list)
            and len(snr_range) == 2
            and all(isinstance(x, (int, float)) for x in snr_range)
        ):
            if snr_range[0] > snr_range[1]:
                raise ValueError(
                    "Invalid SNR range. The first value should be less than the second."
                )
            self.snr_range = snr_range
        else:
            raise ValueError(
                "snr_range must be either a single number or a list of two numbers"
            )
        self.train = train
        self.seed = 44 if train else 42
        self.rng = random.Random(self.seed)
        self._initialize()
        self.reset_state()

    @abstractmethod
    def _initialize(self):
        pass

    def reset_state(self):
        self.rng = random.Random(self.seed)

    @abstractmethod
    def generate_noise(self, shape, device="cpu"):
        pass

    def _apply_snr(self, signal, noise):
        device = signal.device
        batch_size = signal.shape[0] if signal.dim() == 2 else 1

        snr = (
            torch.tensor([self.rng.uniform(*self.snr_range) for _ in range(batch_size)])
            .to(device)
            .view(-1, 1)
        )
        signal_power = torch.mean(signal**2, dim=-1, keepdim=True)
        noise_power = torch.mean(noise**2, dim=-1, keepdim=True)

        eps = torch.finfo(signal.dtype).eps
        signal_power = torch.clamp(signal_power, min=eps)
        noise_power = torch.clamp(noise_power, min=eps)

        scale = torch.sqrt(signal_power / (noise_power * torch.pow(10, snr / 10)))

        max_scale = 1e6
        scale = torch.clamp(scale, max=max_scale)

        return noise * scale

    def mix_noise(self, signal):
        try:
            if not isinstance(signal, torch.Tensor):
                raise TypeError("Signal must be a torch.Tensor")
            if signal.dim() not in [1, 2]:
                raise ValueError("Signal must be a 1D or 2D tensor")

            device = signal.device
            noise = self.generate_noise(signal.shape, device)
            noise = self._apply_snr(signal, noise)
            return torch.clamp(signal + noise, -1, 1)
        except Exception as e:
            raise RuntimeError(f"Error in mix_noise: {str(e)}")

    def prepend_noise(self, signal, noise_length_range=[0, 1]):
        try:
            if not isinstance(signal, torch.Tensor):
                raise TypeError("Signal must be a torch.Tensor")
            if signal.dim() not in [1, 2]:
                raise ValueError("Signal must be a 1D or 2D tensor")
            if (
                not isinstance(noise_length_range, list)
                or len(noise_length_range) != 2
                or not all(0 <= x <= 1 for x in noise_length_range)
            ):
                raise ValueError(
                    "noise_length_range must be a list of two numbers between 0 and 1"
                )

            device = signal.device
            batch_size = signal.shape[0] if signal.dim() == 2 else 1
            noise_lengths = [
                int(self.rng.uniform(*noise_length_range) * signal.shape[-1])
                for _ in range(batch_size)
            ]
            max_noise_length = max(noise_lengths)

            if max_noise_length == 0:
                return signal.clone()

            noise_shape = (
                (batch_size, max_noise_length)
                if signal.dim() == 2
                else (max_noise_length,)
            )
            noise = self.generate_noise(noise_shape, device)

            if batch_size > 1:
                padded_noise = torch.zeros_like(noise)
                for i, length in enumerate(noise_lengths):
                    padded_noise[i, :length] = noise[i, :length]
                noise = padded_noise
            else:
                noise = noise[: noise_lengths[0]]

            noise = self._apply_snr(signal[..., : noise.shape[-1]], noise)
            return torch.clamp(torch.cat([noise, signal], dim=-1), -1, 1)
        except Exception as e:
            raise RuntimeError(f"Error in prepend_noise: {str(e)}")

    def append_noise(self, signal, noise_length_range=[0, 1]):
        try:
            if not isinstance(signal, torch.Tensor):
                raise TypeError("Signal must be a torch.Tensor")
            if signal.dim() not in [1, 2]:
                raise ValueError("Signal must be a 1D or 2D tensor")
            if (
                not isinstance(noise_length_range, list)
                or len(noise_length_range) != 2
                or not all(0 <= x <= 1 for x in noise_length_range)
            ):
                raise ValueError(
                    "noise_length_range must be a list of two numbers between 0 and 1"
                )

            device = signal.device
            batch_size = signal.shape[0] if signal.dim() == 2 else 1
            noise_lengths = [
                int(self.rng.uniform(*noise_length_range) * signal.shape[-1])
                for _ in range(batch_size)
            ]
            max_noise_length = max(noise_lengths)

            if max_noise_length == 0:
                return signal.clone()

            noise_shape = (
                (batch_size, max_noise_length)
                if signal.dim() == 2
                else (max_noise_length,)
            )
            noise = self.generate_noise(noise_shape, device)

            if batch_size > 1:
                padded_noise = torch.zeros_like(noise)
                for i, length in enumerate(noise_lengths):
                    padded_noise[i, :length] = noise[i, :length]
                noise = padded_noise
            else:
                noise = noise[: noise_lengths[0]]

            noise = self._apply_snr(signal[..., -noise.shape[-1] :], noise)
            return torch.clamp(torch.cat([signal, noise], dim=-1), -1, 1)
        except Exception as e:
            raise RuntimeError(f"Error in append_noise: {str(e)}")


class WavNoise(Noise):
    def __init__(
        self,
        file_paths: List[str],
        train_split: float = 0.8,
        cache_audio: bool = False,
        target_sample_rate: int = None,
        **kwargs,
    ):
        if not isinstance(file_paths, list) or not all(
            isinstance(f, str) for f in file_paths
        ):
            raise ValueError("file_paths must be a list of strings")
        if not 0 < train_split < 1:
            raise ValueError("train_split must be between 0 and 1")

        self.file_paths = file_paths
        self.train_split = train_split
        self.cache_audio = cache_audio
        self.audio_cache = {}
        self.target_sample_rate = target_sample_rate
        super().__init__(**kwargs)

    def _initialize(self):
        if not self.file_paths:
            raise ValueError("file_paths cannot be empty")
        self._split_data()

    def _split_data(self):
        self.rng.shuffle(self.file_paths)
        split_idx = max(
            1, int(len(self.file_paths) * self.train_split)
        )  # Ensure at least one file in each split
        self.train_files = self.file_paths[:split_idx]
        self.eval_files = (
            self.file_paths[split_idx:]
            if split_idx < len(self.file_paths)
            else self.file_paths[:1]
        )

    def reset_state(self):
        super().reset_state()
        self.files_to_use = self.train_files if self.train else self.eval_files
        self.rng.shuffle(self.files_to_use)
        self.file_iterator = iter(self.files_to_use)

    def _load_next_noise(self, length, device):
        max_attempts = len(self.files_to_use)
        attempts = 0

        while attempts < max_attempts:
            try:
                file_path = next(self.file_iterator)
            except StopIteration:
                self.reset_state()
                file_path = next(self.file_iterator)

            try:
                if self.cache_audio and file_path in self.audio_cache:
                    noise, sample_rate = self.audio_cache[file_path]
                else:
                    noise, sample_rate = torchaudio.load(file_path)
                    noise = noise.squeeze()
                    if noise.dim() != 1:
                        raise ValueError(
                            f"Audio file {file_path} has unexpected number of channels"
                        )
                    if self.cache_audio:
                        self.audio_cache[file_path] = (noise, sample_rate)

                noise = noise.to(device)  # Move noise to the correct device

                if self.target_sample_rate and sample_rate != self.target_sample_rate:
                    noise = torchaudio.functional.resample(
                        noise, sample_rate, self.target_sample_rate
                    )

                # Ensure the noise is long enough
                if len(noise) < length:
                    repeat_count = length // len(noise) + 1
                    noise = noise.repeat(repeat_count)

                # Randomly select a starting point
                start = self.rng.randint(0, len(noise) - length)
                return noise[start : start + length]

            except FileNotFoundError:
                print(f"File not found: {file_path}. Skipping.")
            except Exception as e:
                print(f"Error loading audio file {file_path}: {str(e)}. Skipping.")

            attempts += 1

        raise RuntimeError(
            f"Failed to load a suitable noise file after {max_attempts} attempts"
        )

    def generate_noise(self, shape, device="cpu"):
        batch_size = shape[0] if len(shape) > 1 else 1
        length = shape[-1]

        noises = []
        for _ in range(batch_size):
            noise = self._load_next_noise(length, device)
            noises.append(noise)

        return torch.stack(noises) if batch_size > 1 else noises[0]


class CHiMEHomeNoise(WavNoise):
    def __init__(self, data_dir: str, **kwargs):
        self.data_sampling_rate = 16000
        chunks_dir = os.path.join(data_dir, "chunks")
        if not os.path.exists(chunks_dir):
            raise FileNotFoundError(f"Directory not found: {chunks_dir}")

        file_suffix = f".{self.data_sampling_rate // 1000}kHz.wav"
        file_paths = [
            os.path.join(chunks_dir, f)
            for f in os.listdir(chunks_dir)
            if f.endswith(file_suffix)
        ]

        if not file_paths:
            raise ValueError(
                f"No .wav files found in {chunks_dir} with suffix {file_suffix}"
            )

        super().__init__(file_paths, **kwargs)


class ESC50Noise(WavNoise):
    def __init__(self, data_dir: str, **kwargs):
        audio_dir = os.path.join(data_dir, "audio")
        if not os.path.exists(audio_dir):
            raise FileNotFoundError(f"Directory not found: {audio_dir}")
        file_paths = [
            os.path.join(audio_dir, f)
            for f in os.listdir(audio_dir)
            if f.endswith(".wav")
        ]
        if not file_paths:
            raise ValueError(f"No .wav files found in {audio_dir}")
        super().__init__(file_paths, **kwargs)


class ColoredNoise(Noise):
    def __init__(self, color, **kwargs):
        if color not in ["white", "pink", "brown"]:
            raise ValueError(f"Unsupported noise color: {color}")
        self.color = color
        super().__init__(**kwargs)

    def _initialize(self):
        self.torch_rng = None
        self.reset_state()

    def reset_state(self):
        super().reset_state()

    def generate_noise(self, shape, device="cpu"):
        if self.torch_rng is None or self.torch_rng.device != device:
            self.torch_rng = torch.Generator(device=device)
            self.torch_rng.manual_seed(self.seed)

        batch_size = shape[0] if len(shape) > 1 else 1

        white_noise = torch.randn(shape, generator=self.torch_rng, device=device)

        if self.color == "white":
            return white_noise
        elif self.color == "pink":
            return self._color_noise(white_noise, -1)
        elif self.color == "brown":
            return self._color_noise(white_noise, -2)

    def _color_noise(self, white_noise, exponent):
        fft = torch.fft.rfft(white_noise, dim=-1)
        frequencies = torch.fft.rfftfreq(
            white_noise.shape[-1], device=white_noise.device
        )

        # Avoid division by zero for DC component
        fft[..., 1:] *= torch.pow(frequencies[1:], exponent / 2)
        fft[..., 0] = 0  # Set DC component to zero

        colored_noise = torch.fft.irfft(fft, n=white_noise.shape[-1], dim=-1)

        # Normalize to have unit variance
        return colored_noise / torch.std(colored_noise, dim=-1, keepdim=True)


class WhiteNoise(ColoredNoise):
    def __init__(self, **kwargs):
        super().__init__("white", **kwargs)


class PinkNoise(ColoredNoise):
    def __init__(self, **kwargs):
        super().__init__("pink", **kwargs)


class BrownNoise(ColoredNoise):
    def __init__(self, **kwargs):
        super().__init__("brown", **kwargs)
