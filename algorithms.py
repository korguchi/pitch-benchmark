import numpy as np
import librosa
from pysptk import sptk
import torchcrepe
import torch
import penn
import parselmouth
from typing import Tuple, Callable, Optional
from abc import ABC, abstractmethod
import amfm_decompy.pYAAPT as pYAAPT
import amfm_decompy.basic_tools as basic


class PitchAlgorithm(ABC):
    """Abstract base class for pitch detection algorithms.

    Implements common functionality for pitch extraction including:
    - Input validation and preprocessing
    - Output resampling and alignment
    - Sanity checks and post-processing
    """

    def __init__(self, sample_rate: int, hop_size: int, fmin: float, fmax: float):
        """Initialize pitch algorithm with required parameters.

        Args:
            sample_rate: Audio sampling rate in Hz
            hop_size: Number of samples between successive frames
            fmin: Minimum detectable frequency in Hz
            fmax: Maximum detectable frequency in Hz

        Raises:
            ValueError: If parameters are invalid
        """
        if fmin >= fmax:
            raise ValueError(f"fmin ({fmin}) must be less than fmax ({fmax})")
        if sample_rate <= 0:
            raise ValueError(f"Sample rate must be positive, got {sample_rate}")
        if hop_size <= 0:
            raise ValueError(f"Hop size must be positive, got {hop_size}")

        self.sample_rate = sample_rate
        self.hop_size = hop_size
        self.fmin = fmin
        self.fmax = fmax

    def _validate_audio(self, audio: np.ndarray) -> None:
        """Validate input audio array.

        Args:
            audio: Input audio signal

        Raises:
            ValueError: If audio is invalid or empty
        """
        if audio.size == 0:
            raise ValueError("Empty audio input")
        if not np.isfinite(audio).all():
            raise ValueError("Audio contains non-finite values")
        if not (-1.0 <= audio).all() and (audio <= 1.0).all():
            raise ValueError("Audio must be normalized to [-1, 1]")

    def resample(
        self, pitch: np.ndarray, periodicity: np.ndarray, audio_length: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Align pitch and periodicity to match the audio length based on hop size.

        Args:
            pitch: Pitch values of shape (frames,)
            periodicity: Periodicity values of shape (frames,)
            audio_length: Length of the input audio in samples

        Returns:
            Tuple containing:
                - Aligned pitch values
                - Aligned periodicity values
        """
        target_length = 1 + audio_length // self.hop_size

        if target_length <= 0 or len(pitch) == 0:
            return np.zeros(max(0, target_length)), np.zeros(max(0, target_length))

        original_points = np.linspace(0, 1, len(pitch))
        target_points = np.linspace(0, 1, target_length)

        aligned_pitch = np.interp(target_points, original_points, pitch)
        aligned_periodicity = np.interp(target_points, original_points, periodicity)

        return aligned_pitch, aligned_periodicity

    def sanity_check(
        self, pitch: np.ndarray, periodicity: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Perform sanity checks and cleaning on pitch and periodicity values.

        Args:
            pitch: Array of pitch values
            periodicity: Array of periodicity/confidence values

        Returns:
            Tuple containing sanitized pitch and periodicity arrays
        """
        periodicity = np.nan_to_num(periodicity, nan=0.0)
        pitch = np.nan_to_num(pitch, nan=0.0)
        pitch = np.clip(pitch, self.fmin, self.fmax)
        periodicity = np.clip(periodicity, 0.0, 1.0)

        return pitch, periodicity

    @abstractmethod
    def extract_pitch_and_periodicity(
        self, audio: np.ndarray, threshold: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract raw pitch and periodicity values from audio.

        Args:
            audio: Input audio array
            threshold: Voicing decision threshold

        Returns:
            Tuple containing:
                - Pitch values in Hz
                - Periodicity/confidence values
        """
        pass

    def __call__(
        self, audio: np.ndarray, threshold: float = 0.5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Process audio to extract pitch and voicing information.

        The function handles pitch extraction, periodicity calculation, and resampling
        if needed. The output length is standardized to 1 + audio.size(-1) // hop_size
        to match ground truth pitches, accounting for centering effects.

        Args:
            audio: Input audio signal, normalized to [-1, 1]
            threshold: Voicing decision threshold

        Returns:
            Tuple containing:
                - Pitch values (Hz) with unvoiced frames set to 0
                - Binary voicing decisions

        Raises:
            ValueError: If input audio is invalid

        Notes:
            Resampling is only applied for algorithms that don't naturally output
            the expected number of samples (e.g., Praat) or use different internal
            sampling rates (e.g., CREPE at 16kHz/160 hop). Algorithms with default
            centering like pYIN typically don't require resampling.
        """
        self._validate_audio(audio)
        pitch, periodicity = self.extract_pitch_and_periodicity(audio, threshold)
        pitch, periodicity = self.sanity_check(pitch, periodicity)
        pitch, periodicity = self.resample(pitch, periodicity, len(audio))
        voicing = periodicity >= threshold
        return pitch * voicing, voicing


class PraatPitchAlgorithm(PitchAlgorithm):
    def extract_pitch_and_periodicity(
        self, audio: np.ndarray, threshold: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract pitch using Praat's algorithm.

        Args:
            audio: Input audio signal
            threshold: Not used in Praat algorithm

        Returns:
            Tuple containing:
                - Pitch frequencies in Hz
                - Pitch strength values
        """
        sound = parselmouth.Sound(audio, sampling_frequency=self.sample_rate)
        time_step = self.hop_size / self.sample_rate

        pitch_obj = sound.to_pitch(
            time_step=time_step, pitch_floor=self.fmin, pitch_ceiling=self.fmax
        )

        pitch_values = pitch_obj.selected_array["frequency"]
        strength_values = pitch_obj.selected_array["strength"]

        if len(pitch_values) > 0:
            valid_mask = strength_values > 0
            if valid_mask.any():
                valid_indices = np.where(valid_mask)[0]
                pitch_values = np.interp(
                    np.arange(len(pitch_values)),
                    valid_indices,
                    pitch_values[valid_indices],
                )

        return pitch_values, strength_values


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


class CREPEPitchAlgorithm(PitchAlgorithm):
    def __init__(
        self,
        sample_rate: int,
        hop_size: int,
        fmin: float,
        fmax: float,
        viterbi: bool = True,
        model: str = "full",
        device: str = None,
    ):
        """Initialize CREPE pitch detector.
        Args:
            sample_rate: Audio sampling rate in Hz
            hop_size: Number of samples between successive frames
            fmin: Minimum detectable frequency in Hz
            fmax: Maximum detectable frequency in Hz
            viterbi: Whether to use Viterbi decoding
            model: Model capacity ('tiny', 'small', 'medium', 'large', or 'full')
            device: Device to use ('cpu' or 'cuda')
        """
        super().__init__(sample_rate, hop_size, fmin, fmax)
        self.viterbi = viterbi
        self.model = model
        self.step_size = (self.hop_size / self.sample_rate) * 1000

        import tensorflow as tf

        # Set up TensorFlow device
        if device is None:
            self.tf_device = (
                "/GPU:0" if tf.config.list_physical_devices("GPU") else "/CPU:0"
            )
        elif device == "cuda":
            if not tf.config.list_physical_devices("GPU"):
                print(
                    "Warning: CUDA requested but no GPU available. Falling back to CPU."
                )
                self.tf_device = "/CPU:0"
            else:
                self.tf_device = "/GPU:0"
        else:  # device == 'cpu'
            self.tf_device = "/CPU:0"

        # Force TensorFlow to only use the specified device
        if self.tf_device == "/CPU:0":
            tf.config.set_visible_devices([], "GPU")

    def extract_pitch_and_periodicity(
        self, audio: np.ndarray, threshold: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract pitch using CREPE.
        Args:
            audio: Input audio signal
            threshold: Not used (CREPE has its own confidence measure)
        Returns:
            Tuple containing:
                - Pitch frequencies in Hz
                - Confidence values
        """
        import crepe
        import tensorflow as tf

        with tf.device(self.tf_device):
            _, frequency, confidence, _ = crepe.predict(
                audio,
                self.sample_rate,
                model_capacity=self.model,
                viterbi=self.viterbi,
                step_size=self.step_size,
                verbose=0,
            )
        return frequency, confidence


class PENNPitchAlgorithm(PitchAlgorithm):
    def __init__(
        self,
        sample_rate: int,
        hop_size: int,
        fmin: float,
        fmax: float,
        device: str = None,
        batch_size: int = 2048,
        center: str = "half-hop",
    ):
        """Initialize PENN pitch detector.
        Args:
            sample_rate: Audio sampling rate in Hz
            hop_size: Number of samples between successive frames
            fmin: Minimum detectable frequency in Hz
            fmax: Maximum detectable frequency in Hz
            device: Computation device ("cpu", "cuda", or specific GPU index)
            batch_size: Number of frames to process per batch
            center: Frame centering strategy ('half-window', 'half-hop', or 'zero')
        """
        super().__init__(sample_rate, hop_size, fmin, fmax)

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

    def extract_pitch_and_periodicity(
        self, audio: np.ndarray, threshold: Optional[float] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract pitch using PENN.
        Args:
            audio: Input audio signal (mono)
            threshold: Threshold value
        Returns:
            Tuple containing:
                - Pitch frequencies in Hz
                - Binary voicing decisions
        """
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


class SWIPEPitchAlgorithm(PitchAlgorithm):
    def extract_pitch_and_periodicity(
        self, audio: np.ndarray, threshold: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract pitch using SWIPE.

        Args:
            audio: Input audio signal
            threshold: Threshold value

        Returns:
            Tuple containing:
                - Pitch frequencies in Hz
                - Binary voicing decisions
        """
        # SWIPE expects a special range.
        # Map threshold from [0,1] to [0.2,0.5]
        norm_threshold = np.clip(0.2 + threshold * (0.5 - 0.2), 0.2, 0.5)

        f0 = sptk.swipe(
            audio,
            self.sample_rate,
            self.hop_size,
            min=self.fmin,
            max=self.fmax,
            threshold=norm_threshold,
            otype="f0",
        )

        return f0, (f0 >= self.fmin).astype(np.float32)


class RAPTPitchAlgorithm(PitchAlgorithm):
    def extract_pitch_and_periodicity(
        self, audio: np.ndarray, threshold: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract pitch using RAPT.

        Args:
            audio: Input audio signal
            threshold: Voice bias parameter

        Returns:
            Tuple containing:
                - Pitch frequencies in Hz
                - Binary voicing decisions
        """
        audio_scaled = np.clip(audio * 32767, -32768, 32767)
        # RAPT expects a special range.
        # Map threshold from [0,1] to [-0.6,0.7]
        norm_threshold = -0.6 + threshold * (0.7 - (-0.6))

        f0 = sptk.rapt(
            audio_scaled,
            self.sample_rate,
            self.hop_size,
            min=self.fmin,
            max=self.fmax,
            voice_bias=norm_threshold,
            otype="f0",
        )

        return f0, (f0 >= self.fmin).astype(np.float32)


class pYINPitchAlgorithm(PitchAlgorithm):
    def extract_pitch_and_periodicity(
        self, audio: np.ndarray, threshold: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract pitch using pYIN.

        Args:
            audio: Input audio signal
            threshold: Not used (pYIN has its own probability measure)

        Returns:
            Tuple containing:
                - Pitch frequencies in Hz
                - Combined voicing/probability flags
        """
        pitch, voiced_flag, prob_flag = librosa.pyin(
            audio,
            fmin=self.fmin,
            fmax=self.fmax,
            sr=self.sample_rate,
            hop_length=self.hop_size,
        )

        return pitch, np.maximum(voiced_flag, prob_flag)


class YAAPTPitchAlgorithm(PitchAlgorithm):
    """YAAPT pitch detection algorithm implementation."""

    def __init__(
        self,
        sample_rate: int,
        hop_size: int,
        fmin: float = 60.0,
        fmax: float = 400.0,
        frame_length: float = 35.0,
    ):
        """Initialize YAAPT algorithm with configuration parameters.

        Args:
            sample_rate: Audio sampling rate in Hz
            hop_size: Number of samples between successive frames
            fmin: Minimum detectable frequency in Hz (default: 60.0)
            fmax: Maximum detectable frequency in Hz (default: 400.0)
            frame_length: Length of each analysis frame in milliseconds (default: 35.0)
        """
        super().__init__(sample_rate, hop_size, fmin, fmax)

        # Convert frame_length from milliseconds to samples
        self.frame_length_samples = int((frame_length / 1000.0) * sample_rate)

        # Calculate frame spacing in milliseconds for YAAPT
        self.frame_space_ms = (hop_size / sample_rate) * 1000.0

        # Configure YAAPT parameters
        self.yaapt_params = {
            "frame_length": frame_length,  # frame length in ms
            "frame_space": self.frame_space_ms,  # frame spacing in ms
            "f0_min": fmin,  # minimum pitch
            "f0_max": fmax,  # maximum pitch
            "nccf_thresh1": 0.25,  # lower NCCF threshold
            "nccf_thresh2": 0.9,  # upper NCCF threshold
            "nccf_maxcands": 4,  # maximum number of candidates
            "shc_maxpeaks": 4,  # maximum number of SHC peaks
            "shc_pwidth": 50,  # SHC window width
            "shc_thresh1": 5,  # SHC threshold 1
            "shc_thresh2": 1.25,  # SHC threshold 2
            "f0_double": 150,  # pitch doubling threshold
            "f0_half": 150,  # pitch halving threshold
            "merit_boost": 0.20,  # merit boost
            "merit_pivot": 0.99,  # merit pivot
            "merit_extra": 0.4,  # merit extra
            "median_value": 7,  # median filter order
            "dp_w1": 0.15,  # DP weight for voiced-voiced transitions
            "dp_w2": 0.5,  # DP weight for voiced-unvoiced transitions
            "dp_w3": 0.1,  # DP weight for unvoiced-unvoiced transitions
            "dp_w4": 0.9,  # DP weight for local costs
            "spec_pitch_min_std": 0.05,  # minimum spectral pitch std dev
        }

    def extract_pitch_and_periodicity(
        self, audio: np.ndarray, threshold: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract pitch and periodicity using YAAPT algorithm.

        Args:
            audio: Input audio signal
            threshold: Voicing decision threshold for NLFER

        Returns:
            Tuple containing:
                - Pitch values in Hz
                - Periodicity/confidence values
        """
        # Create signal object for pYAAPT
        signal = basic.SignalObj(audio, self.sample_rate)

        # Update NLFER threshold based on input threshold
        self.yaapt_params["nlfer_thresh1"] = threshold

        # Extract pitch using YAAPT
        pitch = pYAAPT.yaapt(signal, **self.yaapt_params)

        # Get pitch values and voicing decisions
        pitch_values = pitch.samp_values

        return pitch_values, (pitch_values >= self.fmin).astype(np.float32)
