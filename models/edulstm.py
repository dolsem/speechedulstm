import keras
from keras.models import Model
from keras.layers import Input, LSTM, Bidirectional, Dense
import keras.backend as K


class EduLSTM:

    def __init__(self,
                 batch_size=1,
                 seqlen=None,
                 input_dim=1,
                 frame_sizes=(10, 10),
                 saves_dir=None,
                 saves_format="weights.{epoch:02d}-{val_loss:.2f}.hdf5"):

        self.saves_dir = saves_dir
        self.saves_format = saves_format
        self.network = build_edulstm(batch_size,
                                     seqlen,
                                     input_dim,
                                     frame_sizes)

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
                  frame_sizes=(10, 10)):
    """
    lstm1_input = Input((seqlen, input_dim))
    lstm1_out = Bidirectional(LSTM(1,
                                   return_sequences=True))(lstm1_input)
    lstm2_input = K.reshape(lstm1_out, [frame_sizes[0], -1])
    lstm2_out = Bidirectional(LSTM(1, return_sequences=True))(lstm2_input)
    lstm3_input = K.reshape(lstm2_out, [frame_sizes[1], -1])

    lstm3_out = LSTM(256, input_dim)(lstm3_input)
    out = Dense(output_dim, activation='softmax')(lstm3_out)

    return Model(inputs=lstm1_input, outputs=out)
    """
    model = keras.models.Sequential()
    model.add(Bidirectional(LSTM(1, return_sequences=True),
                            input_shape=(seqlen, input_dim)))
    model.add(keras.layers.Reshape((frame_sizes[0], -1)))
    model.add(Bidirectional(LSTM(1, return_sequences=True)))
    model.add(keras.layers.Reshape((frame_sizes[1], -1)))
    model.add(LSTM(frame_sizes[1], return_sequences=False))
    model.add(keras.layers.Reshape((frame_sizes[1],)))
    model.add(Dense(output_dim, activation='softmax'))
    return model


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

    network.fit_generator(generator=dataset.training_generator,
                          steps_per_epoch=dataset.training_size,
                          validation_data=dataset.test_generator,
                          validation_steps=dataset.test_size,
                          callbacks=callbacks)
