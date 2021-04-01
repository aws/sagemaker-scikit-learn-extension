import numpy as np


class StarOversampler:
    """
    Implementation of the oversampler proposed in [1] using the `star` topology. The implementation is based on the
    implementation of https://github.com/analyticalmindsltd/smote_variants

    Parameters
    ----------
    proportion: float (default = 1)
        proportion of the difference of n_maj and n_min to sample e.g. 1.0 means that after sampling the number of
        minority samples will be equal to the number of majority samples

    References
    ----------
    .. [1] Gazzah, S. and Amara, N. E. B. "New Oversampling Approaches Based on Polynomial Fitting for Imbalanced Data
    Sets" The Eighth IAPR International Workshop on Document Analysis Systems
    """

    def __init__(self, proportion=1.0):
        self.proportion = proportion

    def fit(self, X, y=None):
        pass

    def resample(self, X, y, verbose=False):
        """
        Generate synthetic minority samples
        """
        unique, counts = np.unique(y, return_counts=True)
        class_stats = dict(zip(unique, counts))
        min_label = unique[0] if counts[0] < counts[1] else unique[1]
        maj_label = unique[1] if counts[0] < counts[1] else unique[0]

        # determine the number of samples to generate
        n_to_sample = self.det_n_to_sample(self.proportion, class_stats[maj_label], class_stats[min_label])

        if n_to_sample == 0:
            if verbose:
                print("StarOversampler: Sampling is not needed")
            return X.copy(), y.copy()

        samples = []
        # Implementation of the star topology
        X_min = X[y == min_label]
        X_mean = np.mean(X_min, axis=0)
        k = max([1, int(np.rint(n_to_sample / len(X_min)))])
        for x in X_min:
            diff = X_mean - x
            for i in range(1, k + 1):
                samples.append(x + float(i) / (k + 1) * diff)
        return np.vstack([X, np.vstack(samples)]), np.hstack([y, np.repeat(min_label, len(samples))])

    def det_n_to_sample(self, proportion, n_maj, n_min):
        """
        Determines the number of samples to generate

        Parameters
        ----------
        proportion: float
            proportion of the difference of n_maj and n_min to sample e.g. 1.0 means that after sampling the number of
            minority samples will be equal to the number of majority samples
        n_maj: int
            number of majority samples
        n_min: int
            number of minority samples
        """
        return max([0, int((n_maj - n_min) * proportion)])
