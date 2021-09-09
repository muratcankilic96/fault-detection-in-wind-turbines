# Fault Detection of Wind Turbines
A simulation based on the Master's thesis, Fault Detection of Wind Turbines Using Deep Learning[1].

# Dependencies

- Python 3.x
- Tensorflow 2 C API: https://www.tensorflow.org/install/lang_c
- Cppflow: https://github.com/serizba/cppflow
- Aubio 0.49: https://github.com/aubio/aubio

# Build

The project can be built with *qmake* tool. First, open the .pro file and edit dependency path variables

```
AUBIO_DIR = $$PWD/aubio
TENSORFLOW_DIR = $$PWD/tensorflow
CPPFLOW_DIR = $$PWD/cppflow

```

```
mkdir build
cd build
qmake ..
make
```

References:
[1] Kilic, Muratcan. Fault Detection of Wind Turbines Using Deep Learning. MS thesis. Itä-Suomen yliopisto, 2021.
