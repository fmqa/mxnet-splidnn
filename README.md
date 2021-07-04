# Spoken language identification DNN implemented in mxnet

This is an mxnet solution for the kaggle dataset ["Spoken Language Identification"](https://www.kaggle.com/toponowicz/spoken-language-identification). It uses a simple FFDNN to determine the language spoken in audio data containing human speech.

The included program & model support classification of German, English, and Spanish speech.

## Usage

Training and inference can be performed using the included python program.

```
usage: splidnn.py [-h] {train,infer} ...

Spoken language identification DNN

positional arguments:
  {train,infer}  sub-command help
    train        Train the DNN model
    infer        Infer using trained model

optional arguments:
  -h, --help     show this help message and exit
```

## Training

Training works using the dataset from https://www.kaggle.com/toponowicz/spoken-language-identification -- simply download the archive containing the training/test data and pass it over as an argument. Unpacking the zip archive is not necessary.

The training procedure alternates between training and validation passes, saving a copy of the network (weights) and the optimizer state at the end of every epoch.

```
usage: splidnn.py train [-h] -e EPOCHS archive

positional arguments:
  archive               Dataset archive

optional arguments:
  -h, --help            show this help message and exit
  -e EPOCHS, --epochs EPOCHS
```

## Inference

Specify the model state and any audio file(s) to analyze. The inference procedure will split up the audio input into 10-second chunks and output the predicted language label per block. FLAC and WAV input formats are supported, audio IO is performed using [PySoundFile](https://pysoundfile.readthedocs.io)

```
usage: splidnn.py infer [-h] -m MODEL audio [audio ...]

positional arguments:
  audio                 Audio file(s) to process

optional arguments:
  -h, --help            show this help message and exit
  -m MODEL, --model MODEL
```

The following example infers language labels for a single example audio file:

```
$ python3 splidnn.py infer --model model_epoch5 /tmp/de_example_file.flac 
/tmp/de_example_file.flac DE,DE,DE,DE
```

## Results

The results of the sample training run are included in `output.txt` as well as the best model/trainer pair `model_epoch5`, `trainer_epoch5`. We stop training after 5 epochs to avoid overfitting to the input dataset.
