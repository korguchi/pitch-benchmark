import numpy as np
import torch
import penn
from typing import Tuple
from .base import ContinuousPitchAlgorithm


class PENNPitchAlgorithm(ContinuousPitchAlgorithm):
    def __init__(
        self,
        device: str = None,
        batch_size: int = 2048,
        center: str = "half-hop",
        **kwargs,
    ):
        """Initialize PENN pitch detector.
        Args:
            device: Computation device ("cpu", "cuda", or specific GPU index)
            batch_size: Number of frames to process per batch
            center: Frame centering strategy ('half-window', 'half-hop', or 'zero')
        """
        super().__init__(**kwargs)

        # Convert hop_size from samples to seconds as required by PENN
        self.hopsize_seconds = float(self.hop_size) / self.sample_rate

        # Handle device selection
        if device is None:
            self.gpu = 0 if torch.cuda.is_available() else None
        elif device == "cpu":
            self.gpu = None
        elif device == "cuda":
            self.gpu = 0 if torch.cuda.is_available() else None
        elif isinstance(device, int):
            if device >= torch.cuda.device_count():
                raise ValueError(f"GPU index {device} is out of range")
            self.gpu = device
        else:
            raise ValueError(f"Unsupported device specification: {device}")

        self.batch_size = batch_size
        self.center = center

    def _extract_raw_pitch_and_periodicity(
        self, audio: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        # Ensure audio is float32 and in correct shape
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        # Convert to torch tensor and add batch dimension if needed
        if len(audio.shape) == 1:
            audio_tensor = torch.from_numpy(audio).unsqueeze(0)
        else:
            audio_tensor = torch.from_numpy(audio)

        pitch, periodicity = penn.from_audio(
            audio=audio_tensor,
            sample_rate=self.sample_rate,
            hopsize=self.hopsize_seconds,
            fmin=self.fmin,
            fmax=self.fmax,
            batch_size=self.batch_size,
            center=self.center,
            gpu=self.gpu,
        )

        # Convert to numpy and remove batch dimension
        return pitch.squeeze().cpu().numpy(), periodicity.squeeze().cpu().numpy()
