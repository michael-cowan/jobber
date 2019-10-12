import numpy as np


def multidimensional_shifting(num_samples, sample_size, elements,
                              probabilities=None):
    """
    Magic sampling technique I found online
    https://medium.com/ibm-watson/incredibly-fast-random-sampling-in-python-baf154bd836a
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
