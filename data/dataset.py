from abc import ABCMeta, abstractmethod, abstractproperty
import pickle
import numpy
from dolsemlib.decorators import classproperty


class DataGenerator:
    """
    Wrapper class used to construct generators with set length
    and specified parameters.
    """
    def __init__(self, generator_func, length, generator_params=[], flags={}):
        self._generator = generator_func(*generator_params)
        self._length = length
        self._flags = flags

    def __call__(self):
        return self.__next__()

    def __next__(self):
        return next(self._generator)

    def __len__(self):
        return self._length

    def set(self, flag, value):
        self._flags[flag] = value

    def get(self, flag):
        return self._flags[flag]


class Dataset(metaclass=ABCMeta):
    """
    Dataset abstract class that has properties X_train, Y_train, X_test,
    Y_test, and generator - generator['train'] and generator['test'] can be
    used to generate train and test data accordingly.

    A subclass must define _preprocess method, which sets _training_set
    and _test_set (they should be lists of indices), and stores the dataset in
    _data. It must also store lengths of dataset and its partitions in
    _trainlen, _testlen, and _datalen.
    Define _load to change the dataset loading procedure, if necessary.
    Define _on_batch_complete and _on_dataset_complete to
    perform additional preprocessing on these events, if necessary.
    """
    _training_set = abstractproperty()
    _test_set = abstractproperty()
    _data = abstractproperty()
    _batch_size = 1

    def __len__(self):
        return self._datalen

    def __getitem__(self, key):
        return self._data[key]

    def _load(self, path):
        return pickle.load(path)

    @abstractmethod
    def _preprocess(self, raw_data):
        pass

    def __init__(self, load_params={}, preprocess_params={}):
        db_raw = self._load(**load_params)
        self._preprocess(db_raw, **preprocess_params)
        self._generator = {'full': DataGenerator(
            self._generate_data, self._datalen),
                           'train': DataGenerator(
            self._generate_data, self._trainlen, ['train']),
                           'test': DataGenerator(
            self._generate_data, self._testlen, ['test'])}

    def _generate_data(self, partition=None):
        data_ixs = (range(len(self._data)) if partition is None else
                    self._training_set if partition is 'train' else
                    self._test_set if partition is 'test' else None)
        X = []
        Y = []

        while True:
            for ix in data_ixs:
                item, flags = self._get_item(ix)
                for k in flags.keys():
                    self._generator[partition].set(k, flags[k])
                X.append(item[0])
                Y.append(item[1])
                if len(X) >= self._batch_size:
                    yield numpy.array(X), numpy.array(Y)
                    X = []
                    Y = []
            self._on_dataset_complete()

    """def _generate_data(self, partition=None):
        X = []
        Y = []
        if partition is None:
            while True:
                for example in self._data:
                    if self.batch_size is not None:
                        X.append(example['x'])
                        Y.append(example['y'])
                        if len(X) >= self.batch_size:
                            yield numpy.array(X), numpy.array(Y)
                            self._on_batch_complete((X, Y))
                            X = []
                            Y = []
                    else:
                        self._on_batch_begin()
                        for subseq in range(example['x'].shape[0]):
                            yield example['x'][subseq], example['y'][subseq]
                        self._on_batch_complete((example['x'], example['y']))
                self._on_dataset_complete()
        elif partition == 'train':
            while True:
                for ix in self._training_set:
                    if self.batch_size is not None:
                        X.append(self._data[ix]['x'])
                        Y.append(self._data[ix]['y'])
                        if len(X) >= self.batch_size:
                            yield numpy.array(X), numpy.array(Y)
                            self._on_batch_complete((X, Y))
                            X = []
                            Y = []
                    else:
                        print("ix {}".format(ix))
                        self._on_batch_begin()
                        for subseq in range(self._data[ix]['x'].shape[0]):
                            yield self._data[ix]['x'][subseq], self._data[ix]['y'][subseq]
                        self._on_batch_complete((self._data[ix]['x'],
                                                 self._data[ix]['y']))
                self._on_dataset_complete()
        elif partition == 'test':
            while True:
                for ix in self._test_set:
                    if self.batch_size is not None:
                        X.append(self._data[ix]['x'])
                        Y.append(self._data[ix]['y'])
                        if len(X) >= self.batch_size:
                            yield numpy.array(X), numpy.array(Y)
                            self._on_batch_complete((X, Y))
                            X = []
                            Y = []
                    else:
                        self._on_batch_begin()
                        for subseq in range(self._data[ix]['x'].shape[0]):
                            yield self._data[ix]['x'][subseq], self._data[ix]['y'][subseq]
                        self._on_batch_complete((self._data[ix]['x'],
                                                self._data[ix]['y']))
                self._on_dataset_complete()"""

    def _get_item(self, index):
        return (self._data[index]['x'], self._data[index]['y']), {}

    def _on_dataset_complete(self):
        pass

    _getters = {'X_train': lambda self: self._training_set['X'],
                'Y_train': lambda self: self._training_set['Y'],
                'X_test': lambda self: self._test_set['X'],
                'Y_test': lambda self: self._test_set['Y'],
                'generator': lambda self: self._generator,
                'batch_size': lambda self: self._batch_size}

    def set_batch_size(self, val):
        if type(val) is int and val > 0:
            self._batch_size = val

    X_train = property(_getters['X_train'])
    Y_train = property(_getters['Y_train'])
    X_test = property(_getters['X_test'])
    Y_test = property(_getters['Y_test'])
    generator = property(_getters['generator'])
    batch_size = property(_getters['batch_size'], set_batch_size)

    @classproperty
    def _docstring(cls):
        docstring = "Load params:\n"
        docstring += "\n".join(["-> {}: {}".format(
            key, cls._load.__annotations__[key]) for key in
            cls._load.__annotations__.keys()])
        docstring += "\nPreprocess params:\n"
        docstring += "\n".join(["-> {}: {}".format(
            key, cls._preprocess.__annotations__[key]) for key in
            cls._preprocess.__annotations__.keys()])
        return docstring
