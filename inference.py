import argparse
import librosa
import torch

from data_util import audioset_classes
from helpers.decode import batched_decode_preds
from helpers.encode import ManyHotEncoder
from models.transformers.atstframe.ATSTF_wrapper import ATSTWrapper
from models.transformers.beats.BEATs_wrapper import BEATsWrapper
from models.transformers.frame_passt.fpasst_wrapper import FPaSSTWrapper
from models.transformers.m2d.M2D_wrapper import M2DWrapper
from models.transformers.asit.ASIT_wrapper import ASiTWrapper
from models.efficient_cnns.fmn.fmn_wrapper import FrameMNWrapper
from models.prediction_wrapper import PredictionsWrapper
from models.efficient_cnns.fmn.utils import NAME_TO_WIDTH


def sound_event_detection(args):
    """
    Running Sound Event Detection on an audio clip.
    """
    device = torch.device('cuda') if args.cuda and torch.cuda.is_available() else torch.device('cpu')
    model_name = args.model_name

    if model_name == "BEATs":
        beats = BEATsWrapper()
        model = PredictionsWrapper(beats, checkpoint="BEATs_strong_1")
    elif model_name == "ATST-F":
        atst = ATSTWrapper()
        model = PredictionsWrapper(atst, checkpoint="ATST-F_strong_1")
    elif model_name == "fpasst":
        fpasst = FPaSSTWrapper()
        model = PredictionsWrapper(fpasst, checkpoint="fpasst_strong_1")
    elif model_name == "M2D":
        m2d = M2DWrapper()
        model = PredictionsWrapper(m2d, checkpoint="M2D_strong_1", embed_dim=m2d.m2d.cfg.feature_d)
    elif model_name == "ASIT":
        asit = ASiTWrapper()
        model = PredictionsWrapper(asit, checkpoint="ASIT_strong_1")
    elif model_name.startswith("fmn"):
        width = NAME_TO_WIDTH(model_name)
        fmn = FrameMNWrapper(width)
        embed_dim = fmn.state_dict()['fmn.features.16.1.bias'].shape[0]

        # build checkpoint name
        seq_model_name = ""
        if args.seq_model_type:
            seq_model_name = f"+{args.seq_model_type}-{args.seq_model_dim}"

        checkpoint_name = f"{model_name}{seq_model_name}_{args.pretrained}"

        model = PredictionsWrapper(
            fmn,
            checkpoint=checkpoint_name,
            seq_model_type=args.seq_model_type,
            seq_model_dim=args.seq_model_dim,
            embed_dim=embed_dim
        )
    else:
        raise NotImplementedError(f"Model {model_name} not (yet) implemented")

    model.eval()
    model.to(device)

    sample_rate = 16_000  # all our models are trained on 16 kHz audio
    segment_duration = 10  # all models are trained on 10-second pieces
    segment_samples = segment_duration * sample_rate

    # load audio
    (waveform, _) = librosa.core.load(args.audio_file, sr=sample_rate, mono=True)
    waveform = torch.from_numpy(waveform[None, :]).to(device)
    waveform_len = waveform.shape[1]

    audio_len = waveform_len / sample_rate  # in seconds
    print("Audio length (seconds): ", audio_len)

    # encoder manages decoding of model predictions into dataframes
    # containing event labels, onsets and offsets
    encoder = ManyHotEncoder(audioset_classes.as_strong_train_classes, audio_len=audio_len)

    # split audio file into 10-second chunks
    num_chunks = waveform_len // segment_samples + (waveform_len % segment_samples != 0)
    all_predictions = []

    # Process each 10-second chunk
    for i in range(num_chunks):
        start_idx = i * segment_samples
        end_idx = min((i + 1) * segment_samples, waveform_len)
        waveform_chunk = waveform[:, start_idx:end_idx]

        # Pad the last chunk if it's shorter than 10 seconds
        if waveform_chunk.shape[1] < segment_samples:
            pad_size = segment_samples - waveform_chunk.shape[1]
            waveform_chunk = torch.nn.functional.pad(waveform_chunk, (0, pad_size))

        # Run inference for each chunk
        with torch.no_grad():
            mel = model.mel_forward(waveform_chunk)
            y_strong, _ = model(mel)

        # Collect predictions
        all_predictions.append(y_strong)

    # Concatenate all predictions along the time axis
    y_strong = torch.cat(all_predictions, dim=2)
    # convert into probabilities
    y_strong = torch.sigmoid(y_strong)

    (
        scores_unprocessed,
        scores_postprocessed,
        decoded_predictions
    ) = batched_decode_preds(
        y_strong.float(),
        [args.audio_file],
        encoder,
        median_filter=args.median_window,
        thresholds=args.detection_thresholds,
    )

    for th in decoded_predictions:
        print("***************************************")
        print(f"Detected events using threshold {th}:")
        print(decoded_predictions[th].sort_values(by="onset"))
        print("***************************************")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Example of parser. ')
    # model names: [fmn04, fmn06, fmn10, fmn20, fmn30, BEATs, ASIT, ATST-F, fpasst, M2D]
    parser.add_argument('--model_name', type=str,
                        choices=["ATST-F", "BEATs", "fpasst", "M2D", "ASIT"] + \
                                [f"fmn{width}" for width in ["04", "06", "10", "20", "30"]],
                        default='fmn10')
    parser.add_argument('--pretrained', type=str, choices=["strong", "advanced-kd-weak-strong"],
                        default="strong")
    parser.add_argument('--seq_model_type', type=str, choices=[None, "gru", "attn", "tf", "mamba", "tcn", "hybrid"],
                        default=None)
    parser.add_argument('--seq_model_dim', type=int, default=256)
    parser.add_argument('--audio_file', type=str,
                        default='test_files/752547__iscence__milan_metro_coming_in_station.wav')
    parser.add_argument('--detection_thresholds', type=float, default=(0.1, 0.2, 0.5))
    parser.add_argument('--median_window', type=float, default=9)
    parser.add_argument('--cuda', action='store_true', default=False)
    args = parser.parse_args()

    assert args.model_name in ["BEATs", "ASIT", "ATST-F", "fpasst", "M2D"] or args.model_name.startswith("fmn")

    if args.pretrained == "advanced-kd-weak-strong":
        assert args.model_name.startswith("fmn"), "Advanced KD is only available for fmn models"

    sound_event_detection(args)
