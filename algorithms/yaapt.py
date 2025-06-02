import numpy as np
from typing import Tuple
import amfm_decompy.pYAAPT as pYAAPT
import amfm_decompy.basic_tools as basic
from .base import PitchAlgorithm


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
