
from tensorflow.keras.models import Sequential
from tensorflow.keras import regularizers
from tensorflow.keras.layers import (Dense, Conv2D, Flatten, MaxPooling2D, BatchNormalization, LeakyReLU, 
SpatialDropout2D, RNN, SimpleRNNCell, LSTM, Dropout, Input)

class ModelDefinitions:
    
    mfcc_length = 0
    mfcc_width = 0
    
    spectro_length = 0
    spectro_width = 0
    
    mel_spectro_length = 0
    mel_spectro_width = 0
    
    # Model 1 [Representation = MFCC | Model = CNN]
    @staticmethod
    def callMFCC_CNN():
      model = Sequential()
      model.add(Input(shape=(ModelDefinitions.mfcc_length,ModelDefinitions.mfcc_width,1)))
      model.add(Conv2D(32, kernel_size=3, kernel_regularizer=regularizers.l2(0.0005)))
      model.add(LeakyReLU(alpha=0.1))
      model.add(BatchNormalization())
      model.add(SpatialDropout2D(0.07))
      model.add(Conv2D(32, kernel_size=3, kernel_regularizer=regularizers.l2(0.0005)))
      model.add(LeakyReLU(alpha=0.1))
      model.add(BatchNormalization())
      model.add(SpatialDropout2D(0.07))
      model.add(MaxPooling2D())
      model.add(SpatialDropout2D(0.07))
      model.add(Conv2D(64, kernel_size=3, kernel_regularizer=regularizers.l2(0.0005)))
      model.add(LeakyReLU(alpha=0.1))
      model.add(BatchNormalization())
      model.add(SpatialDropout2D(0.14))
      model.add(Conv2D(64, kernel_size=3, kernel_regularizer=regularizers.l2(0.0005)))
      model.add(LeakyReLU(alpha=0.1))
      model.add(BatchNormalization())
      model.add(Flatten())
      model.add(Dense(3, activation='softmax'))
      model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
      model.summary()
      return model
  
    # Model 2 [Representation = Spectrogram | Model = CNN]
    @staticmethod
    def callSpectrogram_CNN():
      model = Sequential()
      model.add(Input(shape=(ModelDefinitions.spectro_length,ModelDefinitions.spectro_width,1)))
      model.add(Conv2D(32, kernel_size=3, kernel_regularizer=regularizers.l2(0.0005)))
      model.add(LeakyReLU(alpha=0.1))
      model.add(Conv2D(32, kernel_size=3, kernel_regularizer=regularizers.l2(0.0005)))
      model.add(LeakyReLU(alpha=0.1))
      model.add(MaxPooling2D())
      model.add(Conv2D(64, kernel_size=3, kernel_regularizer=regularizers.l2(0.0005)))
      model.add(LeakyReLU(alpha=0.1))
      model.add(Conv2D(64, kernel_size=3, kernel_regularizer=regularizers.l2(0.0005)))
      model.add(LeakyReLU(alpha=0.1))
      model.add(Flatten())
      model.add(Dense(3, activation='softmax'))
      model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
      model.summary()
      return model
  
    # Model 3 [Representation = Mel Spectrogram | Model = CNN]
    @staticmethod
    def callMelSpectrogram_CNN():
      model = Sequential()
      model.add(Input(shape=(ModelDefinitions.mel_spectro_length,ModelDefinitions.mel_spectro_width,1)))
      model.add(Conv2D(50, kernel_size=5, activation='relu', kernel_regularizer=regularizers.l2(0.0005)))
      model.add(Conv2D(100, kernel_size=3, activation='relu', kernel_regularizer=regularizers.l2(0.0005)))
      model.add(MaxPooling2D())
      model.add(Conv2D(200, kernel_size=3, activation='relu', kernel_regularizer=regularizers.l2(0.0005)))
      model.add(Conv2D(400, kernel_size=3, activation='relu', kernel_regularizer=regularizers.l2(0.0005)))
      model.add(Flatten())
      model.add(Dense(3, activation='softmax'))
      model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
      model.summary()
      return model
  
    # Model 4 [Representation = MFCC | Model = RNN]
    @staticmethod
    def callMFCC_RNN():
      model = Sequential()
      model.add(Input(shape=(ModelDefinitions.mfcc_length, ModelDefinitions.mfcc_width)))
      model.add(RNN(SimpleRNNCell(100), return_sequences=True))
      model.add(RNN(SimpleRNNCell(100), return_sequences=False))
      model.add(Dropout(0.1))
      model.add(Dense(3, activation='softmax'))
      model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
      model.summary()
      return model
  
    # Model 5 [Representation = Spectrogram | Model = RNN]
    @staticmethod
    def callSpectrogram_RNN():
      model = Sequential()
      model.add(Input(shape=(ModelDefinitions.spectro_length, ModelDefinitions.spectro_width)))
      model.add(RNN(SimpleRNNCell(100), return_sequences=True))
      model.add(RNN(SimpleRNNCell(100), return_sequences=False))
      model.add(Dropout(0.1))
      model.add(Dense(3, activation='softmax'))
      model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
      model.summary()
      return model
  
    # Model 6 [Representation = Mel Spectrogram | Model = RNN]
    @staticmethod
    def callMelSpectrogram_RNN():
      model = Sequential()
      model.add(Input(shape=(ModelDefinitions.mel_spectro_length, ModelDefinitions.mel_spectro_width)))
      model.add(RNN(SimpleRNNCell(256)))
      model.add(Dense(64, activation='relu'))
      model.add(Dense(3, activation='softmax'))
      model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
      model.summary()
      return model
  
    # Model 7 [Representation = MFCC | Model = LSTM]
    @staticmethod
    def callMFCC_LSTM():
      model = Sequential()
      model.add(Input(shape=(ModelDefinitions.mfcc_length, ModelDefinitions.mfcc_width)))
      model.add(LSTM(100, return_sequences=True))
      model.add(LSTM(100, return_sequences=False))
      model.add(Dropout(0.1))
      model.add(Dense(3, activation='softmax'))
      model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
      return model
  
    # Model 8 [Representation = Spectrogram | Model = LSTM]
    @staticmethod
    def callSpectrogram_LSTM():
      model = Sequential()
      model.add(Input())
      model.add(LSTM(512))
      model.add(Dense(3, activation='softmax'))
      model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
      return model
  
    # Model 9 [Representation = Mel Spectrogram | Model = LSTM]
    @staticmethod
    def callMelSpectrogram_LSTM():
      model = Sequential()
      model.add(Input())
      model.add(LSTM(128, return_sequences=True))
      model.add(RNN(SimpleRNNCell(128), return_sequences=True))
      model.add(LSTM(128, return_sequences=True))
      model.add(RNN(SimpleRNNCell(128)))
      model.add(Dense(3, activation='softmax'))
      model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
      return model

        
