import numpy as np
import os


def files_exist(path, files):
    """
    Returns True if all files exist within path
    """
    for f in files:
        if not os.path.isfile(os.path.join(path, f)):
            return False
    return True


def multidimensional_shifting(num_samples, sample_size, elements,
                              probabilities=None):
    """
    Magic sampling technique I found online
    https://medium.com/ibm-watson/incredibly-fast-random-sampling-in-python-baf154bd836a

    Args:
    num_samples (int): number of random samples to create
    sample_size (int): length of each sample - how many items to choose
                       from <elements> WITHOUT REPLACEMENT
    elements (list): list of elements to choose from

    KArgs:
    probabilites (list): list of probabilites for selecting each element
                         (Default: None = uniform probability)
    """
    # default to uniform probabilities
    if probabilities is None:
        probabilities = [1 / float(len(elements))] * len(elements)

    # replicate probabilities as many times as `num_samples`

    replicated_probabilities = np.tile(probabilities, (num_samples, 1))

    # get random shifting numbers & scale them correctly
    random_shifts = np.random.random(replicated_probabilities.shape)
    random_shifts /= random_shifts.sum(axis=1)[:, np.newaxis]

    # shift by numbers & find largest (by finding the smallest of the negative)
    shifted_probabilities = random_shifts - replicated_probabilities
    return np.argpartition(shifted_probabilities,
                           sample_size, axis=1)[:, :sample_size]


if __name__ == '__main__':
    # create 5 samples of 3 items selected from range(20)
    print(multidimensional_shifting(num_samples=5, sample_size=3,
                                    elements=range(20)))
