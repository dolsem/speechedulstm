import keras
from keras.models import Model
from keras.layers import Input, Reshape, Dropout
from keras.layers import LSTM, Bidirectional, Dense


class EduLSTM:

    def __init__(self,
                 batch_size=1,
                 seqlen: "Number of frames, set to None for variable frames\
number." = None,
                 input_dim: "Frame vector size, 1 for single number" = 1,
                 output_dim: "Output vector size, should be equal to number of\
categories" = 5,
                 frame_sizes: "Number of layer outputs combined together as an\
input to the next layer at a single time step" = (25, 20),
                 hyper_sizes: "Sizes of LSTM cell outputs at each layer" =
                 (16, 16, 256),
                 dropout_ratio=0.2,
                 saves_dir: "Set to None to disable saving weights during\
training" = None,
                 saves_format="weights.{epoch:02d}-{val_loss:.2f}.hdf5"):

        self.saves_dir = saves_dir
        self.saves_format = saves_format
        self.network = build_edulstm(batch_size,
                                     seqlen,
                                     input_dim,
                                     output_dim,
                                     frame_sizes,
                                     hyper_sizes,
                                     dropout_ratio)

    def __call__(self, audio_sample):
        return self.network.predict(audio_sample)

    def __str__(self):
        out = ""
        self.network.summary(print_fn=lambda s: out+s+'\n')
        return out

    def compile(self,
                optimizer='rmsprop',
                loss='categorical_crossentropy',
                metrics=['accuracy']):
        self.network.compile(optimizer, loss, metrics)

    def train(self,
              dataset,
              epochs=3,
              early_stopping=True,
              patience=3,
              monitor='val_loss',
              save=False,
              save_period=1,
              verbose=False):

        train_edulstm(self.network,
                      dataset=dataset,
                      epochs=epochs,
                      early_stopping=early_stopping,
                      patience=patience,
                      monitor=monitor,
                      saves_dir=self.saves_dir,
                      # TODO: raise warning if saves_dir is None.
                      saves_format=self.saves_dir if save else None,
                      save_period=save_period,
                      verbose=verbose)

    def save(self, name, save_dir=None):
        save_dir = self.saves_dir if save_dir is None else save_dir
        path = '' if save_dir is None else save_dir + '/'
        path += name
        self.network.save_weights(path)

    def load(self, name=None, load_dir=None, epoch=None, save_format=None):
        """
        if name is None and epoch is None:
            raise Exception
        if name is None and epoch is not None and format is None:
            raise Exception
        """
        load_dir = self.saves_dir if load_dir is None else load_dir
        path = '' if load_dir is None else load_dir + '/'
        save_format = self.saves_format if save_format is None else save_format
        name = save_format.format(epoch, *(['*']*10)) if name is None else name
        path += name
        self.network.load_weights(path)


def build_edulstm(batch_size=1,
                  seqlen=None,   # Allows for arbitrary sequence length
                  input_dim=1,
                  output_dim=5,
                  frame_sizes=(25, 20),
                  hyper_sizes=(16, 16, 256),
                  dropout_ratio=0.2):

    lstm1_input = Input(batch_shape=(1, seqlen, input_dim))
    lstm1_out = Bidirectional(LSTM(hyper_sizes[0],
                                   return_sequences=True))(lstm1_input)
    if dropout_ratio > 0:
            lstm1_out = Dropout(dropout_ratio)(lstm1_out)

    lstm2_input = Reshape((-1, frame_sizes[0]*2*hyper_sizes[0]))(lstm1_out)
    lstm2_out = Bidirectional(LSTM(hyper_sizes[1],
                                   return_sequences=True))(lstm2_input)
    if dropout_ratio > 0:
            lstm2_out = Dropout(dropout_ratio)(lstm2_out)

    lstm3_input = Reshape((-1, frame_sizes[1]*2*hyper_sizes[1]))(lstm2_out)
    lstm3_out = LSTM(hyper_sizes[2])(lstm3_input)
    out = Dense(output_dim, activation='softmax')(lstm3_out)

    return Model(inputs=lstm1_input, outputs=out)


def train_edulstm(network,
                  dataset,
                  epochs=3,
                  early_stopping=True,
                  patience=3,
                  monitor='val_loss',
                  saves_dir=None,
                  saves_format="weights.{epoch:02d}-{val_loss:.2f}.h5",
                  save_period=1,
                  verbose=False):

    callbacks = []
    if saves_dir is not None:
        filepath = saves_dir + '/' + saves_format
        callbacks.append(keras.callbacks.ModelCheckpoint(filepath,
                                                         monitor=monitor,
                                                         period=save_period))

    if early_stopping is True:
        callbacks.append(keras.callbacks.EarlyStopping(monitor=monitor,
                                                       min_delta=0,
                                                       patience=patience,
                                                       verbose=verbose,
                                                       mode='auto'))

    network.fit_generator(generator=dataset.generator['train'],
                          steps_per_epoch=len(dataset.generator['train']),
                          validation_data=dataset.generator['test'],
                          validation_steps=len(dataset.generator['test']),
                          callbacks=callbacks)
