# Fault Detection of Wind Turbines Using Deep Learning
A simulation based on the Master's thesis, Fault Detection of Wind Turbines Using Deep Learning[1]. 

The models consist of three categories:
- Working
- Problematic
- Not Working

And there are 9 different Keras models built in Python scripts for 3 data representations and 3 model structures.
- MFCC            |       CNN
- Spectrogram     |       RNN
- Mel Spectrogram |       LSTM

# Dependencies

- Qt 5
- Python 3.x
- Python libraries:
  - Numpy
  - Tensorflow
  - Keras
  - imblearn
  - matplotlib
- Tensorflow 2 C API: https://www.tensorflow.org/install/lang_c
- Cppflow: https://github.com/serizba/cppflow
- Aubio 0.49: https://github.com/aubio/aubio

# Build

The project can be built with *qmake* tool. First, open the .pro file and edit dependency path variables...

```
AUBIO_DIR = $$PWD/aubio
TENSORFLOW_DIR = $$PWD/tensorflow
CPPFLOW_DIR = $$PWD/cppflow

```

Then:

```
mkdir build
cd build
qmake .. -spec linux-g++ CONFIG+=qtquickcompiler
make
```
Or you can use Qt Creator interface.

# References:

[1] Kilic, Muratcan. Fault Detection of Wind Turbines Using Deep Learning. MS thesis. Itä-Suomen yliopisto, 2021.
