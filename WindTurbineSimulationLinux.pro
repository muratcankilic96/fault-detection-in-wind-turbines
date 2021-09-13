QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

CONFIG += c++14
QMAKE_CXXFLAGS += -D_GLIBCXX_USE_CXX11_ABI=1

OBJECTS_DIR = $$PWD/build
MOC_DIR = $$PWD/build
RCC_DIR = $$PWD/build
UI_DIR = $$PWD/build

# Set paths here
AUBIO_DIR = $$PWD/aubio
TENSORFLOW_DIR = $$PWD/tensorflow
CPPFLOW_DIR = $$PWD/cppflow

SOURCES += \
    dsp.cpp \
    main.cpp \
    mainwindow.cpp \
    read_wav.cpp \
    tensorflowpreprocessor.cpp

HEADERS += \
    dsp.h \
    mainwindow.h \
    read_wav.h \
    tensorflowpreprocessor.h

FORMS += \
    mainwindow.ui

# Default rules for deployment.
qnx: target.path = /tmp/$${TARGET}/bin
else: unix:!android: target.path = /opt/$${TARGET}/bin
!isEmpty(target.path): INSTALLS += target

unix:!macx: LIBS += -L$${AUBIO_DIR}/build/src/ -laubio

INCLUDEPATH += $$PWD/aubio/src
DEPENDPATH += $$PWD/aubio/src

unix:!macx: LIBS += -L$${TENSORFLOW_DIR}/lib -ltensorflow_framework -ltensorflow

INCLUDEPATH += $${TENSORFLOW_DIR}/include
INCLUDEPATH += $${CPPFLOW_DIR}/include
DEPENDPATH += $${TENSORFLOW_DIR}/include
