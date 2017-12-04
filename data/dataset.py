from abc import ABCMeta, abstractmethod, abstractproperty
import pickle
import numpy
from dolsemlib.decorators import classproperty


class DataGenerator:
    """
    Wrapper class used to construct generators with set length
    and specified parameters.
    """
    def __init__(self, generator_func, length, generator_params=[]):
        self.generator = generator_func
        self.length = length
        self.generator_params = generator_params

    def __call__(self):
        return self.__next__()

    def __next__(self):
        return next(self.generator(*self.generator_params))

    def __len__(self):
        return self.length


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
    _trainlen = None
    _testlen = None
    _datalen = None
    _batch_size = 1

    def __len__(self):
        return self._datalen

    def _load(self, path):
        return pickle.load(path)

    @abstractmethod
    def _preprocess(self, raw_data):
        pass

    def __init__(self, load_params={}, preprocess_params={}):
        db_raw = self._load(**load_params)
        self._preprocess(db_raw, **preprocess_params)

    def _generate_data(self, partition=None):
        batch = []
        if partition is None:
            while True:
                for example in self._data:
                    batch.append(example)
                    if len(batch) >= self.batch_size:
                        yield numpy.array(batch)
                        self._on_batch_complete(batch)
                        batch = []
                self._on_dataset_complete()
        elif partition == 'train':
            while True:
                for example_ix in self._train_set:
                    batch.append(self._data[example_ix])
                    if len(batch) == self._batch_size:
                        yield numpy.array(batch)
                        self._on_batch_complete(batch)
                        batch = []
                self._on_dataset_complete()
        elif partition == 'test':
            while True:
                for example_ix in self._test_set:
                    batch.append(self._data[example_ix])
                    if len(batch) == self._batch_size:
                        yield numpy.array(batch)
                        self._on_batch_complete(batch)
                        batch = []
                self._on_dataset_complete()

    def _on_batch_complete(self, batch): pass

    def _on_dataset_complete(self): pass

    _generator = {'full': DataGenerator(
        _generate_data, _datalen),
                  'train': DataGenerator(
        _generate_data, _trainlen, ['train']),
                  'test': DataGenerator(
        _generate_data, _testlen, ['test'])}

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
