import numpy as np
from typing import Tuple
from basic_pitch.inference import predict, Model
from basic_pitch import ICASSP_2022_MODEL_PATH
from .base import PitchAlgorithm


class BasicPitchPitchAlgorithm(PitchAlgorithm):
    def __init__(
        self,
        sample_rate: int,
        hop_size: int,
        fmin: float,
        fmax: float,
        onset_threshold: float = 0.5,
        frame_threshold: float = 0.3,
        minimum_note_length: float = 0.058,
        melodia_trick: bool = True,
    ):
        """Initialize Basic Pitch algorithm.
        
        Basic Pitch is a polyphonic automatic music transcription model that can detect
        multiple simultaneous pitches. This wrapper extracts the most prominent pitch
        per frame to fit the monophonic PitchAlgorithm interface.
        
        Args:
            sample_rate: Audio sampling rate in Hz
            hop_size: Number of samples between successive frames
            fmin: Minimum detectable frequency in Hz
            fmax: Maximum detectable frequency in Hz
            onset_threshold: Threshold for onset detection (0.0-1.0, default: 0.5)
            frame_threshold: Threshold for frame-level note detection (0.0-1.0, default: 0.3)
            minimum_note_length: Minimum note length in seconds (default: 0.058)
            melodia_trick: Use melodia trick for better monophonic performance (default: True)
        """
        super().__init__(sample_rate, hop_size, fmin, fmax)
        
        self.onset_threshold = onset_threshold
        self.frame_threshold = frame_threshold
        self.minimum_note_length = minimum_note_length
        self.melodia_trick = melodia_trick
        
        # Load the Basic Pitch model once during initialization
        self.model = Model(ICASSP_2022_MODEL_PATH)
        
    def extract_pitch_and_periodicity(
        self, audio: np.ndarray, threshold: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract pitch using Basic Pitch polyphonic transcription model.
        
        Runs Basic Pitch to get polyphonic note activations, then extracts the most
        prominent pitch at each time frame. The threshold parameter is combined with
        the frame_threshold to determine the minimum activation level.
        
        Args:
            audio: Input audio signal (mono, normalized to [-1, 1])
            threshold: Additional confidence threshold applied on top of frame_threshold
        
        Returns:
            Tuple containing:
                - pitch: Pitch frequencies in Hz (most prominent per frame)
                - confidence: Note activation confidence values (0.0-1.0)
        """
        import tempfile
        import soundfile as sf
        import os
        import sys
        from contextlib import redirect_stdout
        from io import StringIO
        
        # Basic Pitch expects a file path, so we need to save the audio temporarily
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            temp_path = temp_file.name
            
        try:
            # Write audio array to temporary file
            sf.write(temp_path, audio, self.sample_rate)
            
            # Run Basic Pitch prediction with file path (suppress print output)
            with redirect_stdout(StringIO()):
                model_output, _, _ = predict(
                    temp_path,
                    self.model,
                    onset_threshold=self.onset_threshold,
                    frame_threshold=max(self.frame_threshold, threshold),
                    minimum_note_length=self.minimum_note_length,
                    minimum_frequency=self.fmin,
                    maximum_frequency=self.fmax,
                    melodia_trick=self.melodia_trick,
                )
        finally:
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.unlink(temp_path)
        
        # Extract note activation matrix: shape (time, pitch)
        note_activations = model_output['note']
        
        # Basic Pitch uses 88 piano keys (A0 to C8), starting from MIDI note 21
        num_pitches = note_activations.shape[1]
        midi_start = 21  # A0
        midi_numbers = np.arange(midi_start, midi_start + num_pitches)
        frequencies = 440.0 * (2.0 ** ((midi_numbers - 69) / 12.0))
        
        # Create frequency mask for valid range (vectorized)
        valid_freq_mask = (frequencies >= self.fmin) & (frequencies <= self.fmax)
        
        # Apply frequency constraints to all frames at once
        masked_activations = note_activations * valid_freq_mask[np.newaxis, :]
        
        # Find most prominent pitch per frame (vectorized)
        max_indices = np.argmax(masked_activations, axis=1)
        max_confidences = masked_activations[np.arange(len(max_indices)), max_indices]
        
        # Convert MIDI indices to frequencies
        pitch_estimates = frequencies[max_indices]
        
        # Set pitch to 0 where confidence is too low
        pitch_estimates = np.where(max_confidences > 0, pitch_estimates, 0.0)
        confidence_estimates = np.where(max_confidences > 0, max_confidences, 0.0)
        
        return pitch_estimates, confidence_estimates