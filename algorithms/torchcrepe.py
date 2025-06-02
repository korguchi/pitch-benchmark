import numpy as np
import torchcrepe
import torch
from typing import Tuple, Callable
from .base import PitchAlgorithm


class TorchCREPEPitchAlgorithm(PitchAlgorithm):
    def __init__(
        self,
        sample_rate: int,
        hop_size: int,
        fmin: float,
        fmax: float,
        decoder: Callable = torchcrepe.decode.viterbi,
        model: str = "full",
        device: str = None,
    ):
        """Initialize TorchCREPE pitch detector.

        Args:
            sample_rate: Audio sampling rate in Hz
            hop_size: Number of samples between successive frames
            fmin: Minimum detectable frequency in Hz
            fmax: Maximum detectable frequency in Hz
            decoder: Strategy for converting network output to pitch ('weighted_argmax', 'argmax' or 'viterbi')
            model: Model capacity ('tiny', or 'full')
            device: Computation device ("cpu" or "cuda")
        """
        super().__init__(sample_rate, hop_size, fmin, fmax)
        self.decoder = decoder
        self.model = model

        if device is None:
            # Auto-select: prioritize CUDA if available, otherwise fall back to CPU
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif device == "cuda" and not torch.cuda.is_available():
            # User requested CUDA, but it's not available; fall back to CPU
            print("Warning: CUDA requested but not available. Falling back to CPU.")
            self.device = torch.device("cpu")
        else:
            self.device = device

    def extract_pitch_and_periodicity(
        self, audio: np.ndarray, threshold: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract pitch using TorchCREPE.

        Args:
            audio: Input audio signal
            threshold: Not used (CREPE has its own confidence measure)

        Returns:
            Tuple containing:
                - Pitch frequencies in Hz
                - Confidence values
        """
        audio_tensor = torch.from_numpy(audio).to(self.device).unsqueeze(0)

        pitch, periodicity = torchcrepe.predict(
            audio_tensor,
            self.sample_rate,
            self.hop_size,
            self.fmin,
            self.fmax,
            model=self.model,
            return_periodicity=True,
            decoder=self.decoder,
            device=self.device,
            batch_size=2048,
        )

        return pitch.squeeze().cpu().numpy(), periodicity.squeeze().cpu().numpy()
