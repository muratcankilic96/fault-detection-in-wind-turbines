QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

CONFIG += c++14
QMAKE_CXXFLAGS += -D_GLIBCXX_USE_CXX11_ABI=1


# You can make your code fail to compile if it uses deprecated APIs.
# In order to do so, uncomment the following line.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0

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

unix:!macx: LIBS += -L$$PWD/aubio/build/src/ -laubio

INCLUDEPATH += $$PWD/aubio/src
DEPENDPATH += $$PWD/aubio/src


unix:!macx: LIBS += -L$$PWD/../../tensorflow/lib/ -ltensorflow_framework -ltensorflow

INCLUDEPATH += $$PWD/../../tensorflow/include
INCLUDEPATH += /media/murt/Slave/local/boost_1_77_0
INCLUDEPATH += /media/murt/Slave/tensorflow/cppflow/include
DEPENDPATH += $$PWD/../../tensorflow/include

unix:!macx: LIBS += -L$$PWD/../../local/boost_1_77_0/bin.v2/libs/serialization/build/gcc-9/release/link-static/threading-multi/visibility-hidden/ -lboost_serialization

unix:!macx: PRE_TARGETDEPS += $$PWD/../../local/boost_1_77_0/bin.v2/libs/serialization/build/gcc-9/release/link-static/threading-multi/visibility-hidden/libboost_serialization.a
