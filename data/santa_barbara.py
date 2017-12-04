import numpy
import pickle
import os
import glob
from dolsemlib.decorators import classproperty

from .dataset import Dataset


class SantaBarbaraDataset(Dataset):

    @classproperty
    def __doc__(cls):
        return cls._docstring

    _data = None
    _test_set = None
    _training_set = None

    def __str__(self):
        return """Santa Barbara Corpus of Spoken American English.
-> http://www.linguistics.ucsb.edu/research/santa-barbara-corpus"""

    def _load(self,
              path: "Path to dataset" = './speechintelrnn/db/SantaBarbara',
              scenes_num: "Number of scenes to load. If set to 0, loads every\
              file on the path." = 0,
              verbose: "Prints log to stdout" = False):
        db = []
        filenames = glob.glob(os.path.abspath(path)+"/sb*.p")
        if verbose is True:
            out = (", ".join([os.path.basename(item) for item in filenames]) if
                   len(filenames) < 10 else "{} files.".format(len(filenames)))
            print("Found: {}".format(out))
        count = 1
        for path_to_file in filenames:
            with open(path_to_file, 'rb') as _file:
                db += pickle.load(_file)
            if verbose is True:
                print("Loaded {}".format(os.path.basename(filenames[count-1])))
            count += 1
            if scenes_num > 0 and count > scenes_num:
                break
        if verbose is True:
            print("\n")
        return db

    def _preprocess(self,
                    raw_data,
                    test_ratio: "Ratio of test set size to dataset size" = 0.3,
                    separate_speakers: "If set to True, training set and test\
                    set will not share examples with the same speaker, which\
                    will make it impossible for the network to 'cheat' by\
                    associating education level with speaker id." = True,
                    depth: "Audio depth in bits" = 8,
                    one_hot: "Whether to use one-hot encodings for\
                    labels" = True,
                    seqlen: "Number of frames per example. Shorter examples\
                    will be zero-padded, longer examples will be cropped.\
                    If set to zero, original number of frames will be\
                    preserved (and example length will possibly vary.)" = 0,
                    noise_ratio: 'Ratio of examples with random noise to\
                    generate for training. Can be used to train the\
                    network to distinguish between human speech and other\
                    audio input. Will create additional label class with value\
                    -1.' = .0,
                    verbose: "Prints log to stdout" = False):
        freq = raw_data[0]['freq']
        data = []

        if verbose is True:
            print("Procesing audio. {} sequences to process...".format(
                len(raw_data)))

        edu_level_types = []
        for ix in range(len(raw_data)):
            example = raw_data[ix]
            if example is not None:
                edu_level = example['education_years']
                if edu_level is not None:
                    if edu_level not in edu_level_types:
                        edu_level_types.append(edu_level)
                    # Bring examples to the same length (if seqlen is
                    # non-zero), convert stereo to mono and normalize.
                    wav = (wave2numpy(raw_data[ix]['wav']['left'], seqlen) +
                           wave2numpy(raw_data[ix]['wav']['right'], seqlen) /
                           (2**depth))
                    data.append({'x': wav,
                                 'y': edu_level,
                                 'scene_id': example['conv_id']})  # Change back to scene_id

        if verbose is True:
            print("Done.")

        self.categories = edu_level_types
        self._datalen = len(data)
        num_noise = int(noise_ratio * self._datalen)
        if num_noise > 0:
            if verbose is True:
                print("\nGenerating {} noise sequences...".format(num_noise))
            self.categories.append(-1)
            self._datalen += num_noise
            for i in range(num_noise):
                # Choose sequence length between 1 and 3 seconds.
                length = numpy.random.randint(freq, 3*freq)
                example = (numpy.random.choice([1, -1], length) *
                           numpy.random.rand(length))
                data.append({'x': example, 'y': -1, 'scene_id': -1})
            if verbose is True:
                print("Done.")

        # Convert labels to one hot representation.
        if one_hot is True:
            if verbose is True:
                print("\nConverting labals to one hot representations...")
            edu_levels = numpy.zeros((self._datalen, len(self.categories)))
            edu_levels[numpy.arange(self._datalen),
                       [item['y'] for item in data]] = 1
            for ix in range(self._datalen):
                data[ix]['y'] = edu_levels[ix]
            if verbose is True:
                print("Done.")

        # Partition the dataset
        self._training_set = []
        self._test_set = []
        if verbose is True:
            print("\nPartitioning the dataset...")
        if separate_speakers is True:
            test_scenes_num = int(self.scenes_num * test_ratio)
            test_scenes = range(test_scenes_num)
            train_scenes = range(test_scenes_num, self.scenes_num)
            for ix in range(self._datalen):
                scene_id = data[ix]['scene_id']
                if scene_id in train_scenes:
                    self._training_set.append(ix)
                elif scene_id in test_scenes:
                    self._test_set.append(ix)
                else:  # One of the generated noise samples.
                    if numpy.random.rand() > test_ratio:
                        self._training_set.append(ix)
                    else:
                        self._test_set.append(ix)
        else:
            raise NotImplementedError()

        self._trainlen = len(self._training_set)
        self._testlen = len(self._test_set)
        self._data = data
        if verbose is True:
            print("Done.\n")
            print("Loaded {} train and {} test examples.".format(
                self._trainlen, self._testlen))

    def decode(self, label_vector):
        """ Maps max value of vector to corresponding category."""
        return self.categories[numpy.argmax(label_vector)]


def wave2numpy(wave, seqlen):
    if seqlen == 0:
        return numpy.array(wave)
    else:
        length = len(wave)
        if length < seqlen:
            return numpy.pad(wave, (0, seqlen-length), 'constant',
                             constant_values=0)
        else:
            return numpy.array(wave[:seqlen])

