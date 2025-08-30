import urllib.request
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from librosa.filters import mel
from librosa.util import pad_center
from scipy.signal import get_window

from .base import ContinuousPitchAlgorithm

# Constants
SAMPLE_RATE = 16000
N_CLASS = 360
N_MELS = 128
MEL_FMIN = 30
MEL_FMAX = SAMPLE_RATE // 2
WINDOW_LENGTH = 2048

# Model URLs
DEFAULT_MODEL_URL = (
    "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/rmvpe.pt"
)


def get_model_path(model_path: str = None) -> str:
    """
    Get model path, downloading if necessary.

    Args:
        model_path: Custom model path or None for default

    Returns:
        Path to model file
    """
    if model_path is None:
        # Default to current directory or algorithms directory
        try:
            model_path = Path(__file__).parent / "rmvpe.pt"
        except NameError:
            # Fallback if __file__ not available (e.g., in exec)
            model_path = Path("rmvpe.pt")
    else:
        model_path = Path(model_path)

    # Download if doesn't exist
    if not model_path.exists():
        print(f"RMVPE model not found at {model_path}")
        print(f"Downloading from {DEFAULT_MODEL_URL}...")
        model_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            urllib.request.urlretrieve(DEFAULT_MODEL_URL, str(model_path))
            print(f"Model downloaded successfully to {model_path}")
        except Exception as e:
            if model_path.exists():
                model_path.unlink()  # Remove partial file
            raise RuntimeError(f"Failed to download model: {e}")

    return str(model_path)


class STFT(nn.Module):
    def __init__(
        self, filter_length, hop_length, win_length=None, window="hann"
    ):
        super(STFT, self).__init__()
        if win_length is None:
            win_length = filter_length

        self.filter_length = filter_length
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = window
        self.forward_transform = None
        fourier_basis = np.fft.fft(np.eye(self.filter_length))

        cutoff = int((self.filter_length / 2 + 1))
        fourier_basis = np.vstack(
            [
                np.real(fourier_basis[:cutoff, :]),
                np.imag(fourier_basis[:cutoff, :]),
            ]
        )

        forward_basis = torch.FloatTensor(fourier_basis[:, None, :])

        if window is not None:
            assert filter_length >= win_length
            fft_window = get_window(window, win_length, fftbins=True)
            fft_window = pad_center(fft_window, size=filter_length)
            fft_window = torch.from_numpy(fft_window).float()
            forward_basis *= fft_window

        self.register_buffer("forward_basis", forward_basis.float())

    def forward(self, input_data):
        num_batches = input_data.size(0)
        num_samples = input_data.size(1)

        input_data = input_data.view(num_batches, 1, num_samples)
        forward_transform = F.conv1d(
            input_data, self.forward_basis, stride=self.hop_length, padding=0
        )

        cutoff = int((self.filter_length / 2) + 1)
        real_part = forward_transform[:, :cutoff, :]
        imag_part = forward_transform[:, cutoff:, :]

        magnitude = torch.sqrt(real_part**2 + imag_part**2)
        phase = torch.atan2(imag_part, real_part)

        return magnitude, phase


class MelSpectrogram(torch.nn.Module):
    def __init__(
        self,
        n_mels,
        sample_rate,
        filter_length,
        hop_length,
        win_length=None,
        mel_fmin=0.0,
        mel_fmax=None,
    ):
        super(MelSpectrogram, self).__init__()
        self.stft = STFT(filter_length, hop_length, win_length)

        mel_basis = mel(
            sr=sample_rate,
            n_fft=filter_length,
            n_mels=n_mels,
            fmin=mel_fmin,
            fmax=mel_fmax,
            htk=True,
        )
        mel_basis = torch.from_numpy(mel_basis).float()
        self.register_buffer("mel_basis", mel_basis)

    def forward(self, y):
        assert torch.min(y.data) >= -1
        assert torch.max(y.data) <= 1

        magnitudes, phases = self.stft(y)
        magnitudes = magnitudes.data
        mel_output = torch.matmul(self.mel_basis, magnitudes)
        mel_output = torch.log(torch.clamp(mel_output, min=1e-5))
        return mel_output


class ConvBlockRes(nn.Module):
    def __init__(self, in_channels, out_channels, momentum=0.01):
        super(ConvBlockRes, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1),
                bias=False,
            ),
            nn.BatchNorm2d(out_channels, momentum=momentum),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1),
                bias=False,
            ),
            nn.BatchNorm2d(out_channels, momentum=momentum),
            nn.ReLU(),
        )
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, (1, 1))
            self.is_shortcut = True
        else:
            self.is_shortcut = False

    def forward(self, x):
        if self.is_shortcut:
            return self.conv(x) + self.shortcut(x)
        else:
            return self.conv(x) + x


class ResEncoderBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, n_blocks=1, momentum=0.01
    ):
        super(ResEncoderBlock, self).__init__()
        self.n_blocks = n_blocks
        self.conv = nn.ModuleList()
        self.conv.append(ConvBlockRes(in_channels, out_channels, momentum))
        for i in range(n_blocks - 1):
            self.conv.append(ConvBlockRes(out_channels, out_channels, momentum))
        self.kernel_size = kernel_size
        if self.kernel_size is not None:
            self.pool = nn.AvgPool2d(kernel_size=kernel_size)

    def forward(self, x):
        for i in range(self.n_blocks):
            x = self.conv[i](x)
        if self.kernel_size is not None:
            return x, self.pool(x)
        else:
            return x


class ResDecoderBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels, stride, n_blocks=1, momentum=0.01
    ):
        super(ResDecoderBlock, self).__init__()
        # Adjust output padding based on stride
        if stride == (1, 2):
            out_padding = (0, 1)
        elif stride == (2, 2):
            out_padding = (1, 1)
        elif stride == (2, 1):
            out_padding = (1, 0)
        else:
            out_padding = (1, 1)  # Default case
        self.n_blocks = n_blocks
        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(3, 3),
                stride=stride,
                padding=(1, 1),
                output_padding=out_padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels, momentum=momentum),
            nn.ReLU(),
        )
        self.conv2 = nn.ModuleList()
        self.conv2.append(
            ConvBlockRes(out_channels * 2, out_channels, momentum)
        )
        for i in range(n_blocks - 1):
            self.conv2.append(
                ConvBlockRes(out_channels, out_channels, momentum)
            )

    def forward(self, x, concat_tensor):
        x = self.conv1(x)
        x = torch.cat((x, concat_tensor), dim=1)
        for i in range(self.n_blocks):
            x = self.conv2[i](x)
        return x


class Encoder(nn.Module):
    def __init__(
        self,
        in_channels,
        in_size,
        n_encoders,
        kernel_size,
        n_blocks,
        out_channels=16,
        momentum=0.01,
    ):
        super(Encoder, self).__init__()
        self.n_encoders = n_encoders
        self.bn = nn.BatchNorm2d(in_channels, momentum=momentum)
        self.layers = nn.ModuleList()
        self.latent_channels = []
        for i in range(self.n_encoders):
            self.layers.append(
                ResEncoderBlock(
                    in_channels,
                    out_channels,
                    kernel_size,
                    n_blocks,
                    momentum=momentum,
                )
            )
            self.latent_channels.append([out_channels, in_size])
            in_channels = out_channels
            out_channels *= 2
            in_size //= 2
        self.out_size = in_size
        self.out_channel = out_channels

    def forward(self, x):
        concat_tensors = []
        x = self.bn(x)
        for i in range(self.n_encoders):
            _, x = self.layers[i](x)
            concat_tensors.append(_)
        return x, concat_tensors


class Intermediate(nn.Module):
    def __init__(
        self, in_channels, out_channels, n_inters, n_blocks, momentum=0.01
    ):
        super(Intermediate, self).__init__()
        self.n_inters = n_inters
        self.layers = nn.ModuleList()
        self.layers.append(
            ResEncoderBlock(in_channels, out_channels, None, n_blocks, momentum)
        )
        for i in range(self.n_inters - 1):
            self.layers.append(
                ResEncoderBlock(
                    out_channels, out_channels, None, n_blocks, momentum
                )
            )

    def forward(self, x):
        for i in range(self.n_inters):
            x = self.layers[i](x)
        return x


class Decoder(nn.Module):
    def __init__(
        self, in_channels, n_decoders, stride, n_blocks, momentum=0.01
    ):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList()
        self.n_decoders = n_decoders
        for i in range(self.n_decoders):
            out_channels = in_channels // 2
            self.layers.append(
                ResDecoderBlock(
                    in_channels, out_channels, stride, n_blocks, momentum
                )
            )
            in_channels = out_channels

    def forward(self, x, concat_tensors):
        for i in range(self.n_decoders):
            x = self.layers[i](x, concat_tensors[-1 - i])
        return x


class DeepUnet0(nn.Module):
    def __init__(
        self,
        kernel_size,
        n_blocks,
        en_de_layers=5,
        inter_layers=4,
        in_channels=1,
        en_out_channels=16,
    ):
        super(DeepUnet0, self).__init__()
        self.encoder = Encoder(
            in_channels,
            N_MELS,
            en_de_layers,
            kernel_size,
            n_blocks,
            en_out_channels,
        )
        self.intermediate = Intermediate(
            self.encoder.out_channel // 2,
            self.encoder.out_channel,
            inter_layers,
            n_blocks,
        )
        self.decoder = Decoder(
            self.encoder.out_channel, en_de_layers, kernel_size, n_blocks
        )

    def forward(self, x):
        x, concat_tensors = self.encoder(x)
        x = self.intermediate(x)
        x = self.decoder(x, concat_tensors)
        return x


class BiGRU(nn.Module):
    def __init__(self, input_features, hidden_features, num_layers):
        super(BiGRU, self).__init__()
        self.gru = nn.GRU(
            input_features,
            hidden_features,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )

    def forward(self, x):
        return self.gru(x)[0]


class E2E0(nn.Module):
    def __init__(
        self,
        hop_length,
        n_blocks,
        n_gru,
        kernel_size,
        en_de_layers=5,
        inter_layers=4,
        in_channels=1,
        en_out_channels=16,
    ):
        super(E2E0, self).__init__()
        self.mel = MelSpectrogram(
            N_MELS,
            SAMPLE_RATE,
            WINDOW_LENGTH,
            hop_length,
            None,
            MEL_FMIN,
            MEL_FMAX,
        )
        self.unet = DeepUnet0(
            kernel_size,
            n_blocks,
            en_de_layers,
            inter_layers,
            in_channels,
            en_out_channels,
        )
        self.cnn = nn.Conv2d(en_out_channels, 3, (3, 3), padding=(1, 1))
        if n_gru:
            self.fc = nn.Sequential(
                BiGRU(3 * N_MELS, 256, n_gru),
                nn.Linear(512, N_CLASS),
                nn.Dropout(0.25),
                nn.Sigmoid(),
            )
        else:
            self.fc = nn.Sequential(
                nn.Linear(3 * N_MELS, N_CLASS), nn.Dropout(0.25), nn.Sigmoid()
            )

    def forward(self, x):
        mel = (
            self.mel(x.reshape(-1, x.shape[-1])).transpose(-1, -2).unsqueeze(1)
        )
        x = self.cnn(self.unet(mel)).transpose(1, 2).flatten(-2)
        x = self.fc(x)
        return x


def to_local_average_cents(salience, center=None, thred=0.0):
    """Find the weighted average cents near the argmax bin"""
    if not hasattr(to_local_average_cents, "cents_mapping"):
        to_local_average_cents.cents_mapping = (
            np.linspace(0, 7180, 360) + 1997.3794084376191
        )

    if salience.ndim == 1:
        if center is None:
            center = int(np.argmax(salience))
        start = max(0, center - 4)
        end = min(len(salience), center + 5)
        salience = salience[start:end]
        product_sum = np.sum(
            salience * to_local_average_cents.cents_mapping[start:end]
        )
        weight_sum = np.sum(salience)
        return product_sum / weight_sum if np.max(salience) > thred else 0
    if salience.ndim == 2:
        return np.array(
            [
                to_local_average_cents(salience[i, :], None, thred)
                for i in range(salience.shape[0])
            ]
        )

    raise Exception("label should be either 1d or 2d ndarray")


class RMVPEInference:
    def __init__(
        self, model, seg_len, seg_frames, hop_length, batch_size, device
    ):
        super(RMVPEInference, self).__init__()
        self.model = model.eval()
        self.seg_len = seg_len
        self.seg_frames = seg_frames
        self.batch_size = batch_size
        self.hop_length = hop_length
        self.device = device

    def inference(self, audio):
        with torch.no_grad():
            padded_audio = self.pad_audio(audio)
            segments = self.en_frame(padded_audio)
            out_segments = self.forward_in_mini_batch(self.model, segments)
            out_segments = self.de_frame(out_segments, type_seg="pitch")[
                : (len(audio) // self.hop_length + 1)
            ]
            return out_segments

    def pad_audio(self, audio):
        audio_len = len(audio)
        seg_nums = int(np.ceil(audio_len / self.seg_len)) + 1
        pad_len = seg_nums * self.seg_len - audio_len + self.seg_len // 2
        padded_audio = torch.cat(
            [
                torch.zeros(self.seg_len // 4).to(self.device),
                audio,
                torch.zeros(pad_len - self.seg_len // 4).to(self.device),
            ]
        )
        return padded_audio

    def en_frame(self, audio):
        audio_len = len(audio)
        assert audio_len % (self.seg_len // 2) == 0

        audio = torch.cat(
            [
                torch.zeros(1024).to(self.device),
                audio,
                torch.zeros(1024).to(self.device),
            ]
        )
        segments = []
        start = 0
        while start + self.seg_len <= audio_len:
            segments.append(audio[start : start + self.seg_len + 2048])
            start += self.seg_len // 2
        segments = torch.stack(segments, dim=0)
        return segments

    def forward_in_mini_batch(self, model, segments):
        out_segments = []
        segments_num = segments.shape[0]
        batch_start = 0
        while True:
            if batch_start + self.batch_size >= segments_num:
                batch_tmp = segments[batch_start:].shape[0]
                segment_in = torch.cat(
                    [
                        segments[batch_start:],
                        torch.zeros_like(segments)[
                            : self.batch_size - batch_tmp
                        ].to(self.device),
                    ],
                    dim=0,
                )
                out_tmp = model(segment_in)
                out_segments.append(out_tmp[:batch_tmp])
                break
            else:
                segment_in = segments[
                    batch_start : batch_start + self.batch_size
                ]
                out_tmp = model(segment_in)
                out_segments.append(out_tmp)
            batch_start += self.batch_size
        out_segments = torch.cat(out_segments, dim=0)
        return out_segments

    def de_frame(self, segments, type_seg="audio"):
        output = []
        if type_seg == "audio":
            for segment in segments:
                output.append(
                    segment[self.seg_len // 4 : int(self.seg_len * 0.75)]
                )
        else:
            for segment in segments:
                output.append(
                    segment[self.seg_frames // 4 : int(self.seg_frames * 0.75)]
                )
        output = torch.cat(output, dim=0)
        return output


class RMVPEPitchAlgorithm(ContinuousPitchAlgorithm):
    _name = "RMVPE"

    def __init__(
        self,
        model_path: str = None,
        device: str = "cpu",
        batch_size: int = 8,
        **kwargs,
    ):
        """
        Initialize RMVPE pitch estimator

        Args:
            model_path: Path to model weights (.pt file)
            device: Device to run inference on ('cpu' or 'cuda')
            batch_size: Batch size for inference
            **kwargs: Additional arguments for base class (sample_rate, hop_size, fmin, fmax)
        """
        super().__init__(**kwargs)

        # RMVPE always works at 16kHz internally - base class handles alignment

        # Set up device
        if device == "cuda" and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        # RMVPE model requires fixed hop_length=320 samples (20ms at 16kHz)
        # This is baked into the trained model architecture
        self.model_hop_length = 320  # Fixed model hop length
        self.seg_len = (
            40640  # This gives 128 mel frames which properly divides by 2
        )
        self.seg_frames = self.seg_len // self.model_hop_length + 1

        # Get model path (auto-download if needed)
        model_path = get_model_path(model_path)

        # Load model
        self.model = self.load_model(model_path)

        # Initialize inference
        self.inference = RMVPEInference(
            self.model,
            seg_len=self.seg_len,
            seg_frames=self.seg_frames,
            hop_length=self.model_hop_length,
            batch_size=batch_size,
            device=self.device,
        )

    def load_model(self, model_path):
        """Load RMVPE model from checkpoint"""
        model = E2E0(self.model_hop_length, 4, 1, (2, 2))
        checkpoint = torch.load(model_path, map_location=self.device)

        # Handle DataParallel wrapper
        if hasattr(checkpoint, "module"):
            state_dict = checkpoint.module.state_dict()
        elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = (
                checkpoint.state_dict()
                if hasattr(checkpoint, "state_dict")
                else checkpoint
            )

        # Load only the parameters that exist in checkpoint (mel/stft are computed on-the-fly)
        model_dict = model.state_dict()
        filtered_dict = {
            k: v
            for k, v in state_dict.items()
            if k in model_dict and model_dict[k].shape == v.shape
        }
        model_dict.update(filtered_dict)
        model.load_state_dict(model_dict)

        model.to(self.device)
        model.eval()
        return model

    def _preprocess_audio(self, audio: np.ndarray) -> np.ndarray:
        """
        Preprocess audio for RMVPE model - always converts to 16kHz mono.
        Base class handles final alignment to user's hop_size.
        """
        # Convert to mono if stereo
        if len(audio.shape) == 2:
            audio = audio.mean(axis=1)

        # Convert to float32
        audio = audio.astype(np.float32)

        # Always resample to model's native 16kHz (base class handles user alignment)
        if self.sample_rate != SAMPLE_RATE:
            try:
                from resampy import resample

                audio = resample(audio, self.sample_rate, SAMPLE_RATE)
            except ImportError:
                from scipy.signal import resample

                target_length = int(len(audio) * SAMPLE_RATE / self.sample_rate)
                audio = resample(audio, target_length).astype(np.float32)

        # Normalize to [-1, 1]
        audio_max = np.max(np.abs(audio))
        if audio_max > 1.0:
            audio = audio / audio_max

        return audio

    def _extract_raw_pitch_and_periodicity(
        self, audio: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract raw pitch and periodicity from audio

        Args:
            audio: Audio numpy array normalized to [-1, 1]

        Returns:
            times: Time points for each frame
            pitch: Fundamental frequency in Hz
            periodicity: Confidence/periodicity values [0, 1]
        """
        # Preprocess audio (handles resampling and normalization)
        audio_processed = self._preprocess_audio(audio)

        # Convert to torch tensor
        audio_tensor = torch.from_numpy(audio_processed).float().to(self.device)

        # Run inference
        pitch_pred = self.inference.inference(audio_tensor)

        # Convert to frequency and periodicity
        pitch_pred_np = pitch_pred.cpu().numpy()

        # Convert pitch prediction to cents then to Hz
        cents = to_local_average_cents(
            pitch_pred_np, thred=0.0
        )  # No threshold for raw extraction
        f0 = np.array(
            [10 * (2 ** (cent / 1200)) if cent else 0 for cent in cents]
        )

        # Use the max probability as periodicity/confidence
        periodicity = (
            np.max(pitch_pred_np, axis=1)
            if pitch_pred_np.ndim > 1
            else pitch_pred_np
        )

        # Calculate time points based on model's internal hop length
        # The model operates at 16kHz with 320-sample hops (20ms)
        model_hopsize_seconds = self.model_hop_length / SAMPLE_RATE
        n_frames = len(f0)
        times = np.arange(n_frames) * model_hopsize_seconds

        return times, f0, periodicity

    def _get_default_threshold(self) -> float:
        """Default threshold for RMVPE"""
        return 0.750
