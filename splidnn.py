import zipfile
import os
import argparse
from itertools import islice, count
import numpy as np
from mxnet.gluon import nn, data, loss, Trainer
from mxnet import autograd, init, ndarray
from tqdm import tqdm
import soundfile

__all__ = ["fbanks", "create_lngclf_nn", "LngDataset"]

def fbanks(signal, sample_rate):
    """Generate filter banks & MFCC features.
    
    Sourced from https://github.com/tomasz-oponowicz/spoken_language_identification/blob/master/features.py"""
    # Pre-Emphasis
    pre_emphasis = 0.97
    emphasized_signal = np.append(
        signal[0],
        signal[1:] - pre_emphasis * signal[:-1])
    # Framing
    frame_size = 0.025
    frame_stride = 0.01
    # Convert from seconds to samples
    frame_length, frame_step = (
        frame_size * sample_rate,
        frame_stride * sample_rate)
    signal_length = len(emphasized_signal)
    frame_length = int(round(frame_length))
    frame_step = int(round(frame_step))
    # Make sure that we have at least 1 frame
    num_frames = int(
        np.ceil(float(np.abs(signal_length - frame_length)) / frame_step))
    pad_signal_length = num_frames * frame_step + frame_length
    z = np.zeros((pad_signal_length - signal_length))
    # Pad Signal to make sure that all frames have equal
    # number of samples without truncating any samples
    # from the original signal
    pad_signal = np.append(emphasized_signal, z)
    indices = (
        np.tile(np.arange(0, frame_length), (num_frames, 1)) +
        np.tile(
            np.arange(0, num_frames * frame_step, frame_step),
            (frame_length, 1)
        ).T
    )
    frames = pad_signal[indices.astype(np.int32, copy=False)]
    # Window
    frames *= np.hamming(frame_length)
    # Fourier-Transform and Power Spectrum
    NFFT = 512
    # Magnitude of the FFT
    mag_frames = np.absolute(np.fft.rfft(frames, NFFT))
    # Power Spectrum
    pow_frames = ((1.0 / NFFT) * ((mag_frames) ** 2))
    # Filter Banks
    nfilt = 40
    low_freq_mel = 0
    # Convert Hz to Mel
    high_freq_mel = (2595 * np.log10(1 + (sample_rate / 2) / 700))
    # Equally spaced in Mel scale
    mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilt + 2)
    # Convert Mel to Hz
    hz_points = (700 * (10**(mel_points / 2595) - 1))
    bin = np.floor((NFFT + 1) * hz_points / sample_rate)
    fbank = np.zeros((nfilt, int(np.floor(NFFT / 2 + 1))))
    for m in range(1, nfilt + 1):
        f_m_minus = int(bin[m - 1])   # left
        f_m = int(bin[m])             # center
        f_m_plus = int(bin[m + 1])    # right
        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
    filter_banks = np.dot(pow_frames, fbank.T)
    # Numerical Stability
    filter_banks = np.where(
        filter_banks == 0,
        np.finfo(float).eps,
        filter_banks)
    # dB
    filter_banks = 20 * np.log10(filter_banks)
    return filter_banks

def create_lngclf_nn(net):
    """Create the DNN."""
    net.add(
        nn.Conv2D(channels=32, kernel_size=7, activation="relu", padding=3),
        nn.BatchNorm(),
        nn.MaxPool2D(pool_size=3, strides=2, padding=1),
        nn.Conv2D(channels=64, kernel_size=5, activation="relu", padding=2),
        nn.BatchNorm(),
        nn.MaxPool2D(pool_size=3, strides=2, padding=1),
        nn.Conv2D(channels=128, kernel_size=3, activation="relu", padding=1),
        nn.BatchNorm(),
        nn.MaxPool2D(pool_size=3, strides=2, padding=1),
        nn.Conv2D(channels=256, kernel_size=3, activation="relu", padding=1),
        nn.BatchNorm(),
        nn.MaxPool2D(pool_size=3, strides=2, padding=1),
        nn.Conv2D(channels=512, kernel_size=3, activation="relu", padding=1),
        nn.BatchNorm(),
        nn.MaxPool2D(pool_size=3, strides=2, padding=1),
        nn.Dense(units=256, activation="relu"),
        nn.BatchNorm(),
        nn.Dropout(rate=0.5),
        nn.Dense(units=3)
    )
    return net

class LngDataset(data.Dataset):
    """Directory/Archive-based audio dataset wrapper."""
    def __init__(self, root):
        super().__init__()
        self.items = list(root.iterdir())
    
    def __getitem__(self, index):
        item = self.items[index]
        if item.name.startswith("de_"):
            y = 0
        elif item.name.startswith("en_"):
            y = 1
        elif item.name.startswith("es_"):
            y = 2
        else:
            raise ValueError(f"Unable to determine label for: {item.name}")
        signal, sample_rate = soundfile.read(item.open("r"))
        X = fbanks(signal, sample_rate).astype(np.float32)
        X = X.reshape(1, *X.shape)
        return X, y
    
    def __len__(self):
        return len(self.items)
        

def train(net, loader, trainer, criterion):
    """Trains the net. Yields the batch number and the cumulative loss."""
    n = 0
    sigma = 0
    for batch, (X, y) in enumerate(loader):
        with autograd.record():
            output = net(X)
            loss = criterion(output, y)
        loss.backward()
        trainer.step(X.shape[0])
        n += X.shape[0]
        sigma += loss.sum()
        yield batch, sigma.asscalar() / n

def validate(net, loader, criterion):
    """Evaluates the net. Yields the batch number and the cumulative loss."""
    n = 0
    sigma = 0
    for batch, (X, y) in enumerate(loader):
        output = net(X)
        loss = criterion(output, y)
        sigma += loss.sum()
        n += X.shape[0]
        yield batch, sigma.asscalar() / n

def loop(net, trainer, criterion, train_loader, test_loader):
    """Train/validate net. Yields the epoch number and the cumulative training/validation loss."""
    for epoch in count(1):
        for _, train_loss in (bar := tqdm(train(net, train_loader, trainer, criterion), desc=f"EPOCH {epoch} TRAINING LOSS=?", total=len(train_loader), unit="Batch")):
            bar.set_description(f"EPOCH {epoch} TRAINING LOSS={train_loss}")
        for _, validation_loss in (bar := tqdm(validate(net, test_loader, criterion), desc=f"EPOCH {epoch} VALIDATION LOSS=?", total=len(test_loader), unit="Batch")):
            bar.set_description(f"EPOCH {epoch} VALIDATION LOSS={validation_loss}")
        yield epoch, train_loss, validation_loss

def main(archive, epochs=10):
    """Build model, save trainer and model state each epoch."""
    train_data = LngDataset(zipfile.Path(archive, "train/train/"))
    test_data = LngDataset(zipfile.Path(archive, "test/test/"))
    train_loader = data.DataLoader(train_data, batch_size=32, shuffle=True)
    test_loader = data.DataLoader(test_data, batch_size=32, shuffle=True)
    net = create_lngclf_nn(nn.Sequential())
    net.initialize(init.Xavier())
    criterion = loss.SoftmaxCrossEntropyLoss()
    trainer = Trainer(net.collect_params(), optimizer="adam")
    for epoch, train_loss, validation_loss in islice(loop(net, trainer, criterion, train_loader, test_loader), epochs):
        print(f"EPOCH {epoch} END - TRAIN_LOSS={train_loss}, VALIDATION_LOSS={validation_loss}")
        net.save_parameters(f"model_epoch{epoch}")
        trainer.save_states(f"trainer_epoch{epoch}")

def catch(it):
    """Wraps an iterable, ending the iterator on exceptions."""
    try:
        yield from it
    except:
        return

def infer(model, inputs, *, labels=("DE", "EN", "ES")):
    """Infer spoken language in the given audio file(s)."""
    net = create_lngclf_nn(nn.Sequential())
    net.load_parameters(model)
    for audio in inputs:
        with soundfile.SoundFile(audio) as sf:
            out = []
            for signal in catch(sf.blocks(sf.samplerate * 10, fill_value=0)):
                if signal.ndim > 1:
                    signal = np.mean(signal, axis=1)
                X = fbanks(signal, sf.samplerate).astype(np.float32)
                X = ndarray.array(X[np.newaxis, np.newaxis, ...])
                y = net(X).reshape(-1)
                out.append(int(y.argmax().asscalar()))
            print(audio, ",".join(labels[x] for x in out))
        
def cli(parser):
    """CLI interface."""
    subparsers = parser.add_subparsers(help="sub-command help")
    train_parser = subparsers.add_parser("train", help="Train the DNN model")
    infer_parser = subparsers.add_parser("infer", help="Infer using trained model")
    
    train_parser.add_argument("-e", "--epochs", required=True, default=10, type=int, help="Number of epochs")
    train_parser.add_argument("archive", help="Dataset archive")
    train_parser.set_defaults(function=lambda args: main(args.archive, args.epochs))
    
    infer_parser.add_argument("-m", "--model", required=True, help="Model to use")
    infer_parser.add_argument("audio", nargs="+", help="Audio file(s) to process")
    infer_parser.set_defaults(function=lambda args: infer(args.model, args.audio))
    
    args = parser.parse_args()
    if hasattr(args, "function"):
        args.function(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    cli(argparse.ArgumentParser(description="Spoken language identification DNN"))
