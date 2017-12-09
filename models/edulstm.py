import sys
import numpy
import keras
from keras.models import Model
from keras.layers import Input, Reshape, Dropout
from keras.layers import LSTM, Bidirectional, Dense


class EduLSTM:

    def __init__(self,
                 batch_size=1,
                 truncate=True,
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
        self.truncate = truncate
        self.network = build_edulstm(batch_size=batch_size,
                                     seqlen=seqlen,
                                     stateful=truncate,
                                     input_dim=input_dim,
                                     output_dim=output_dim,
                                     frame_sizes=frame_sizes,
                                     hyper_sizes=hyper_sizes,
                                     dropout_ratio=dropout_ratio)

    def __call__(self, audio_sample):
        return self.network.predict(audio_sample)

    def __str__(self):
        out = ""
        self.network.summary(print_fn=lambda s: out+s+'\n')
        return out

    def compile(self,
                optimizer='rmsprop',
                loss='categorical_crossentropy',
                metrics=['categorical_accuracy']):
        self.network.compile(optimizer, loss, metrics)

    def train(self,
              dataset,
              epochs=3,
              early_stopping=True,
              patience=3,
              #monitor='val_loss',
              save=False,
              save_period=1,
              verbose=False):

        train_edulstm(self.network,
                      dataset=dataset,
                      truncate=self.truncate,
                      epochs=epochs,
                      early_stopping=early_stopping,
                      patience=patience,
                      #monitor=monitor,
                      saves_dir=self.saves_dir,
                      # TODO: raise warning if saves_dir is None.
                      saves_format=self.saves_format if save else None,
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
                  stateful=False,
                  input_dim=1,
                  output_dim=5,
                  frame_sizes=(25, 20),
                  hyper_sizes=(16, 16, 256),
                  dropout_ratio=0.2):

    lstm1_input = Input(batch_shape=(batch_size, seqlen, input_dim))
    lstm1_out = Bidirectional(LSTM(hyper_sizes[0],
                                   return_sequences=True,
                                   stateful=stateful))(lstm1_input)
    if dropout_ratio > 0:
            lstm1_out = Dropout(dropout_ratio)(lstm1_out)

    lstm2_input = Reshape((-1, frame_sizes[0]*2*hyper_sizes[0]))(lstm1_out)
    lstm2_out = Bidirectional(LSTM(hyper_sizes[1],
                                   return_sequences=True,
                                   stateful=stateful))(lstm2_input)
    if dropout_ratio > 0:
            lstm2_out = Dropout(dropout_ratio)(lstm2_out)

    lstm3_input = Reshape((-1, frame_sizes[1]*2*hyper_sizes[1]))(lstm2_out)
    lstm3_out = LSTM(hyper_sizes[2], stateful=stateful)(lstm3_input)
    out = Dense(output_dim, activation='softmax')(lstm3_out)

    return Model(inputs=lstm1_input, outputs=out)


class ResetStatesCallback(keras.callbacks.Callback):
    def __init__(self, dataset):
        self.dataset = dataset

    def on_batch_begin(self, batch, logs={}):
        if self.dataset.generator['train'].get('sequence_end') is True:
            self.model.reset_states()


class TestCallback(keras.callbacks.Callback):
    def __init__(self, history, dataset):
        self.history = history
        self.dataset = dataset

    def on_epoch_end(self, epoch, logs):
        test_loss = 0.
        generator = self.dataset.generator['test']
        self.model.reset_states()
        for i in range(len(generator)):
            sys.stdout.write("\rEvaluating on the test set... {}/{}".format(
                i+1, len(generator)))
            result = self.model.evaluate_generator(generator, steps=1)[1]
            test_loss += result / len(generator)
            if generator.get('sequence_end') is True:
                self.model.reset_states()
        self.history.add(test_loss)

    def on_epoch_begin(self, epoch, logs):
        if epoch > 0:
            print("Accuracy on the test set: {}".format(self.history.peek()))


class TestEarlyStopping(keras.callbacks.Callback):
    def __init__(self, history):
        self.history = history

    def on_epoch_end(self, epoch, logs):
        if self.history.get_min_ix() == 0 and self.history.full():
            self.model.stop_training = True
            print("Early stopping on epoch {}.".format(epoch))


class TestCheckpoint(keras.callbacks.ModelCheckpoint):
    def __init__(self, history, filepath, monitor='test_loss', verbose=0,
                 save_best_only=False, save_weights_only=False,
                 mode='auto', period=1):
        self.history = history
        super(TestCheckpoint, self).__init__(filepath,
                                             monitor=monitor,
                                             verbose=verbose,
                                             save_best_only=save_best_only,
                                             save_weights_only=save_weights_only,
                                             mode=mode,
                                             period=period)

    def on_epoch_end(self, epoch, logs={}):
        logs['test_loss'] = self.history.peek()
        super(TestCheckpoint, self).on_epoch_end(epoch, logs)


class SimpleHistory:
    def __init__(self, max_size):
        self.list = []
        self.max_size = max_size

    def add(self, item):
        if self.full():
            self.list.pop(0)
        self.list.append(item)

    def get_min_ix(self):
        return numpy.argmin(self.list)

    def full(self):
        return len(self.list) == self.max_size

    def peek(self):
        return self.list[-1]


def train_edulstm(network,
                  dataset,
                  truncate=True,
                  epochs=3,
                  early_stopping=True,
                  patience=3,
                  #monitor='val_loss',
                  saves_dir=None,
                  saves_format="weights.{epoch:02d}-{val_loss:.2f}.h5",
                  save_period=1,
                  verbose=False):

    history = SimpleHistory(max_size=patience)

    callbacks = []

    if truncate is True:
        callbacks.append(ResetStatesCallback(dataset))

    callbacks.append(TestCallback(history, dataset))

    if saves_dir is not None:
        filepath = saves_dir + '/' + saves_format
        callbacks.append(TestCheckpoint(history, filepath,
                                        monitor='test_loss',
                                        period=save_period))

    if early_stopping is True:
        callbacks.append(TestEarlyStopping(history))

    network.fit_generator(generator=dataset.generator['train'],
                          steps_per_epoch=len(dataset.generator['train']),
                          epochs=epochs,
                          callbacks=callbacks,
                          shuffle=False)
