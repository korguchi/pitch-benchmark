import torch
import torchaudio
import random
import warnings
from typing import Dict, Union, Optional
from pathlib import Path


class NoiseAugmentedDataset(torch.utils.data.Dataset):
    """
    A wrapper dataset that adds background noise to any PitchDataset.

    This class wraps an existing PitchDataset and applies noise augmentation
    to the audio samples while preserving pitch and periodicity information.

    Args:
        base_dataset: The underlying PitchDataset to wrap
        noise_source: Either a path to noise files directory or 'white' for white noise
        snr_db: Signal-to-noise ratio in dB (higher = less noise)
        noise_probability: Probability of applying noise to each sample (0.0 to 1.0)
        noise_sample_rate: Sample rate of noise files (if using file-based noise)
        enable_caching: Whether to cache loaded noise files in memory
        **kwargs: Additional arguments for future extensibility
    """

    def __init__(
        self,
        base_dataset,  # Should be a PitchDataset instance
        noise_source: Union[str, Path] = "white",
        snr_db: float = 20.0,
        noise_probability: float = 1.0,
        noise_sample_rate: Optional[int] = None,
        enable_caching: bool = True,
        **kwargs,
    ):
        self.base_dataset = base_dataset
        self.snr_db = snr_db
        self.noise_probability = noise_probability
        self.enable_caching = enable_caching

        # Caches for deterministic noise generation
        self.audio_cache: Dict[str, torch.Tensor] = {}  # Cached noise files
        self.noise_decisions: Dict[
            int, bool
        ] = {}  # Noise application decisions per index
        self.noise_audio_cache: Dict[
            int, torch.Tensor
        ] = {}  # Final noisy audio per index

        # Get target sample rate from base dataset
        self.target_sample_rate = base_dataset.sample_rate
        # Give access to important base dataset attributes
        self.sample_rate = self.target_sample_rate
        self.hop_size = base_dataset.hop_size

        # Setup noise source
        if noise_source == "white":
            self.noise_type = "white"
            self.noise_files = []
        else:
            self.noise_type = "file"
            self._setup_file_noise(noise_source, noise_sample_rate)

    def _setup_file_noise(
        self, noise_dir: Union[str, Path], noise_sample_rate: Optional[int]
    ):
        """Setup file-based noise source."""
        noise_dir = Path(noise_dir)

        if not noise_dir.exists():
            raise ValueError(f"Noise directory not found: {noise_dir}")

        # Find audio files (support common formats)
        audio_extensions = ["*.wav", "*.mp3", "*.flac", "*.m4a"]
        self.noise_files = []

        for ext in audio_extensions:
            self.noise_files.extend(list(noise_dir.glob(f"**/{ext}")))

        if not self.noise_files:
            raise ValueError(f"No audio files found in {noise_dir}")

        self.noise_sample_rate = noise_sample_rate
        print(f"Loaded {len(self.noise_files)} noise files")

    def __len__(self) -> int:
        """Return length of the underlying dataset."""
        return len(self.base_dataset)

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, Path]]:
        """
        Get item from base dataset and optionally apply deterministic noise.
        """
        # Return cached version if available
        if idx in self.noise_audio_cache:
            return self.noise_audio_cache[idx]

        # Get original sample from base dataset
        sample = self.base_dataset[idx]

        # Make deterministic noise decision based on index
        if idx not in self.noise_decisions:
            # Create index-specific RNG for reproducible decisions
            rng = random.Random(idx)
            self.noise_decisions[idx] = rng.random() < self.noise_probability

        if self.noise_decisions[idx]:
            # Apply deterministic noise
            noisy_audio = self._add_deterministic_noise(sample["audio"], idx)
            # Create new sample with noisy audio
            noisy_sample = {**sample, "audio": noisy_audio}
            # Cache for future access
            self.noise_audio_cache[idx] = noisy_sample
            return noisy_sample

        return sample

    def _add_deterministic_noise(self, audio: torch.Tensor, idx: int) -> torch.Tensor:
        """Add deterministic noise using index-based generation"""
        audio_length = audio.shape[-1]

        # Generate noise based on source type
        if self.noise_type == "white":
            noise = self._generate_deterministic_white_noise(audio_length, idx)
        else:
            noise = self._generate_deterministic_file_noise(audio_length, idx)

        # Move noise to same device as audio
        noise = noise.to(audio.device)

        # Handle channel dimension matching
        if audio.dim() == 2:  # Audio has channel dimension (C, T)
            # Add channel dimension to noise if missing
            if noise.dim() == 1:
                noise = noise.unsqueeze(0)  # Shape: (1, T)
        else:  # Audio is 1D (T,)
            # Remove channel dimension if present
            if noise.dim() == 2:
                noise = noise.squeeze(0)

        # Calculate signal and noise power
        signal_power = torch.mean(audio**2)
        noise_power = torch.mean(noise**2)

        # Avoid division by zero
        if noise_power == 0:
            warnings.warn("Generated noise has zero power, returning original audio")
            return audio

        # Calculate scaling factor for desired SNR
        snr_linear = 10 ** (self.snr_db / 10)
        noise_scale = torch.sqrt(signal_power / (snr_linear * noise_power))

        # Add scaled noise to signal
        noisy_audio = audio + noise_scale * noise

        # Normalize to prevent clipping
        max_abs = torch.max(torch.abs(noisy_audio))
        if max_abs > 1.0:
            noisy_audio = noisy_audio / max_abs

        return noisy_audio

    def _generate_deterministic_white_noise(
        self, length: int, idx: int
    ) -> torch.Tensor:
        """White noise with index-based seeding"""
        # Seed numpy (used by torch) with index-derived value
        seed = hash(idx) % 2**32
        torch.manual_seed(seed)
        return torch.randn(length)

    def _generate_deterministic_file_noise(self, length: int, idx: int) -> torch.Tensor:
        """File-based noise with index-based deterministic selection"""
        # Create index-specific RNG
        rng = random.Random(idx)

        # 1. Select file deterministically
        noise_idx = rng.randint(0, len(self.noise_files) - 1)
        noise_path = self.noise_files[noise_idx]
        noise_filename = noise_path.name

        # 2. Load with caching
        if self.enable_caching and noise_filename in self.audio_cache:
            noise_audio = self.audio_cache[noise_filename]
        else:
            try:
                noise_audio, file_sr = torchaudio.load(str(noise_path))

                # Convert to mono if stereo
                if noise_audio.shape[0] > 1:
                    noise_audio = torch.mean(noise_audio, dim=0, keepdim=True)

                # Resample to target sample rate if needed
                if file_sr != self.target_sample_rate:
                    resampler = torchaudio.transforms.Resample(
                        file_sr, self.target_sample_rate
                    )
                    noise_audio = resampler(noise_audio)

                # Cache the processed audio
                if self.enable_caching:
                    self.audio_cache[noise_filename] = noise_audio.clone()

            except Exception as e:
                warnings.warn(f"Could not load noise file {noise_path}: {e}")
                return torch.zeros(1, length)  # Return with channel dimension

        # 3. Handle length mismatch
        current_length = noise_audio.shape[-1]

        if current_length < length:
            # Repeat the noise if it's shorter than needed
            repeats = (length // current_length) + 1
            noise_audio = noise_audio.repeat(1, repeats)
            current_length = noise_audio.shape[-1]

        # 4. Deterministic cropping
        if current_length > length:
            max_start = current_length - length
            start_idx = rng.randint(0, max_start)  # Deterministic based on idx
            noise_audio = noise_audio[:, start_idx : start_idx + length]
        elif current_length < length:
            # Pad with zeros if still too short
            padding = length - current_length
            noise_audio = torch.nn.functional.pad(noise_audio, (0, padding))

        return noise_audio

    def clear_caches(self):
        """Clear all noise caches"""
        self.noise_decisions = {}
        self.noise_audio_cache = {}


class CHiMeNoiseDataset(NoiseAugmentedDataset):
    """Convenience wrapper specifically for CHiME-Home noise."""

    def __init__(
        self,
        base_dataset,
        chime_home_dir: Union[str, Path],
        chime_sample_rate: int = 16000,
        **kwargs,
    ):
        """
        Initialize with CHiME-Home dataset.

        Args:
            base_dataset: The underlying PitchDataset to wrap
            chime_home_dir: Path to CHiME-Home dataset directory
            chime_sample_rate: CHiME sample rate (16000 or 48000)
            **kwargs: All arguments from NoiseAugmentedDataset (snr_db, noise_probability, etc.)
        """
        chime_home_dir = Path(chime_home_dir)
        chunks_dir = chime_home_dir / "chunks"

        if not chunks_dir.exists():
            raise ValueError(f"CHiME chunks directory not found: {chunks_dir}")

        # Determine audio file suffix based on sample rate
        if chime_sample_rate == 16000:
            audio_suffix = ".16kHz.wav"
        elif chime_sample_rate == 48000:
            audio_suffix = ".48kHz.wav"
        else:
            raise ValueError(f"Unsupported CHiME sample rate: {chime_sample_rate}")

        # Create temporary directory structure for the parent class
        super().__init__(
            base_dataset=base_dataset,
            noise_source=chunks_dir,
            noise_sample_rate=chime_sample_rate,
            **kwargs,
        )

        # Filter files to only include the desired sample rate
        self.noise_files = [
            f for f in self.noise_files if f.name.endswith(audio_suffix)
        ]

        if not self.noise_files:
            raise ValueError(f"No CHiME files found with suffix {audio_suffix}")

        print(f"CHiMeNoiseDataset initialized with {len(self.noise_files)} CHiME files")
