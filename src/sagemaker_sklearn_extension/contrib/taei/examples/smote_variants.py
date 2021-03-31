# taken from  https://github.com/analyticalmindsltd/smote_variants/blob/master/smote_variants/_smote_variants.py
# commit - e41348b
# flake8: noqa

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 15 11:15:24 2018
@author: gykovacs
"""

# import system packages
import itertools
import logging
import time

# used to parallelize evaluation

# numerical methods and arrays
import numpy as np


# setting the _logger format
_logger = logging.getLogger("smote_variants")
_logger.setLevel(logging.DEBUG)
_logger_ch = logging.StreamHandler()
_logger_ch.setFormatter(logging.Formatter("%(asctime)s:%(levelname)s:%(message)s"))
_logger.addHandler(_logger_ch)


class StatisticsMixin:
    """
    Mixin to compute class statistics and determine minority/majority labels
    """

    def class_label_statistics(self, X, y):
        """
        determines class sizes and minority and majority labels
        Args:
            X (np.array): features
            y (np.array): target labels
        """
        unique, counts = np.unique(y, return_counts=True)
        self.class_stats = dict(zip(unique, counts))
        self.min_label = unique[0] if counts[0] < counts[1] else unique[1]
        self.maj_label = unique[1] if counts[0] < counts[1] else unique[0]
        # shorthands
        self.min_label = self.min_label
        self.maj_label = self.maj_label

    def check_enough_min_samples_for_sampling(self, threshold=2):
        if self.class_stats[self.min_label] < threshold:
            m = "The number of minority samples (%d) is not enough " "for sampling"
            m = m % self.class_stats[self.min_label]
            _logger.warning(self.__class__.__name__ + ": " + m)
            return False
        return True


class RandomStateMixin:
    """
    Mixin to set random state
    """

    def set_random_state(self, random_state):
        """
        sets the random_state member of the object
        Args:
            random_state (int/np.random.RandomState/None): the random state
                                                                initializer
        """

        self._random_state_init = random_state

        if random_state is None:
            self.random_state = np.random
        elif isinstance(random_state, int):
            self.random_state = np.random.RandomState(random_state)
        elif isinstance(random_state, np.random.RandomState):
            self.random_state = random_state
        elif random_state is np.random:
            self.random_state = random_state
        else:
            raise ValueError("random state cannot be initialized by " + str(random_state))


class ParameterCheckingMixin:
    """
    Mixin to check if parameters come from a valid range
    """

    def check_in_range(self, x, name, r):
        """
        Check if parameter is in range
        Args:
            x (numeric): the parameter value
            name (str): the parameter name
            r (list-like(2)): the lower and upper bound of a range
        Throws:
            ValueError
        """
        if x < r[0] or x > r[1]:
            m = "Value for parameter %s outside the range [%f,%f] not" " allowed: %f"
            m = m % (name, r[0], r[1], x)

            raise ValueError(self.__class__.__name__ + ": " + m)

    def check_out_range(self, x, name, r):
        """
        Check if parameter is outside of range
        Args:
            x (numeric): the parameter value
            name (str): the parameter name
            r (list-like(2)): the lower and upper bound of a range
        Throws:
            ValueError
        """
        if x >= r[0] and x <= r[1]:
            m = "Value for parameter %s in the range [%f,%f] not allowed: %f"
            m = m % (name, r[0], r[1], x)

            raise ValueError(self.__class__.__name__ + ": " + m)

    def check_less_or_equal(self, x, name, val):
        """
        Check if parameter is less than or equal to value
        Args:
            x (numeric): the parameter value
            name (str): the parameter name
            val (numeric): value to compare to
        Throws:
            ValueError
        """
        if x > val:
            m = "Value for parameter %s greater than %f not allowed: %f > %f"
            m = m % (name, val, x, val)

            raise ValueError(self.__class__.__name__ + ": " + m)

    def check_less_or_equal_par(self, x, name_x, y, name_y):
        """
        Check if parameter is less than or equal to another parameter
        Args:
            x (numeric): the parameter value
            name_x (str): the parameter name
            y (numeric): the other parameter value
            name_y (str): the other parameter name
        Throws:
            ValueError
        """
        if x > y:
            m = "Value for parameter %s greater than parameter %s not" " allowed: %f > %f"
            m = m % (name_x, name_y, x, y)

            raise ValueError(self.__class__.__name__ + ": " + m)

    def check_less(self, x, name, val):
        """
        Check if parameter is less than value
        Args:
            x (numeric): the parameter value
            name (str): the parameter name
            val (numeric): value to compare to
        Throws:
            ValueError
        """
        if x >= val:
            m = "Value for parameter %s greater than or equal to %f" " not allowed: %f >= %f"
            m = m % (name, val, x, val)

            raise ValueError(self.__class__.__name__ + ": " + m)

    def check_less_par(self, x, name_x, y, name_y):
        """
        Check if parameter is less than another parameter
        Args:
            x (numeric): the parameter value
            name_x (str): the parameter name
            y (numeric): the other parameter value
            name_y (str): the other parameter name
        Throws:
            ValueError
        """
        if x >= y:
            m = "Value for parameter %s greater than or equal to parameter" " %s not allowed: %f >= %f"
            m = m % (name_x, name_y, x, y)

            raise ValueError(self.__class__.__name__ + ": " + m)

    def check_greater_or_equal(self, x, name, val):
        """
        Check if parameter is greater than or equal to value
        Args:
            x (numeric): the parameter value
            name (str): the parameter name
            val (numeric): value to compare to
        Throws:
            ValueError
        """
        if x < val:
            m = "Value for parameter %s less than %f is not allowed: %f < %f"
            m = m % (name, val, x, val)

            raise ValueError(self.__class__.__name__ + ": " + m)

    def check_greater_or_equal_par(self, x, name_x, y, name_y):
        """
        Check if parameter is less than or equal to another parameter
        Args:
            x (numeric): the parameter value
            name_x (str): the parameter name
            y (numeric): the other parameter value
            name_y (str): the other parameter name
        Throws:
            ValueError
        """
        if x < y:
            m = "Value for parameter %s less than parameter %s is not" " allowed: %f < %f"
            m = m % (name_x, name_y, x, y)

            raise ValueError(self.__class__.__name__ + ": " + m)

    def check_greater(self, x, name, val):
        """
        Check if parameter is greater than value
        Args:
            x (numeric): the parameter value
            name (str): the parameter name
            val (numeric): value to compare to
        Throws:
            ValueError
        """
        if x <= val:
            m = "Value for parameter %s less than or equal to %f not allowed" " %f < %f"
            m = m % (name, val, x, val)

            raise ValueError(self.__class__.__name__ + ": " + m)

    def check_greater_par(self, x, name_x, y, name_y):
        """
        Check if parameter is greater than or equal to another parameter
        Args:
            x (numeric): the parameter value
            name_x (str): the parameter name
            y (numeric): the other parameter value
            name_y (str): the other parameter name
        Throws:
            ValueError
        """
        if x <= y:
            m = "Value for parameter %s less than or equal to parameter %s" " not allowed: %f <= %f"
            m = m % (name_x, name_y, x, y)

            raise ValueError(self.__class__.__name__ + ": " + m)

    def check_equal(self, x, name, val):
        """
        Check if parameter is equal to value
        Args:
            x (numeric): the parameter value
            name (str): the parameter name
            val (numeric): value to compare to
        Throws:
            ValueError
        """
        if x == val:
            m = "Value for parameter %s equal to parameter %f is not allowed:" " %f == %f"
            m = m % (name, val, x, val)
            raise ValueError(self.__class__.__name__ + ": " + m)

    def check_equal_par(self, x, name_x, y, name_y):
        """
        Check if parameter is equal to another parameter
        Args:
            x (numeric): the parameter value
            name_x (str): the parameter name
            y (numeric): the other parameter value
            name_y (str): the other parameter name
        Throws:
            ValueError
        """
        if x == y:
            m = "Value for parameter %s equal to parameter %s is not " "allowed: %f == %f"
            m = m % (name_x, name_y, x, y)
            raise ValueError(self.__class__.__name__ + ": " + m)

    def check_isin(self, x, name, li):
        """
        Check if parameter is in list
        Args:
            x (numeric): the parameter value
            name (str): the parameter name
            li (list): list to check if parameter is in it
        Throws:
            ValueError
        """
        if x not in li:
            m = "Value for parameter %s not in list %s is not allowed: %s"
            m = m % (name, str(li), str(x))
            raise ValueError(self.__class__.__name__ + ": " + m)

    def check_n_jobs(self, x, name):
        """
        Check n_jobs parameter
        Args:
            x (int/None): number of jobs
            name (str): the parameter name
        Throws:
            ValueError
        """
        if not ((x is None) or (x is not None and isinstance(x, int) and not x == 0)):
            m = "Value for parameter n_jobs is not allowed: %s" % str(x)
            raise ValueError(self.__class__.__name__ + ": " + m)


class ParameterCombinationsMixin:
    """
    Mixin to generate parameter combinations
    """

    @classmethod
    def generate_parameter_combinations(cls, dictionary, raw):
        """
        Generates reasonable paramter combinations
        Args:
            dictionary (dict): dictionary of paramter ranges
            num (int): maximum number of combinations to generate
        """
        if raw:
            return dictionary
        keys = sorted(list(dictionary.keys()))
        values = [dictionary[k] for k in keys]
        combinations = [dict(zip(keys, p)) for p in list(itertools.product(*values))]
        return combinations


class OverSampling(StatisticsMixin, ParameterCheckingMixin, ParameterCombinationsMixin, RandomStateMixin):
    """
    Base class of oversampling methods
    """

    categories = []

    cat_noise_removal = "NR"
    cat_dim_reduction = "DR"
    cat_uses_classifier = "Clas"
    cat_sample_componentwise = "SCmp"
    cat_sample_ordinary = "SO"
    cat_sample_copy = "SCpy"
    cat_memetic = "M"
    cat_density_estimation = "DE"
    cat_density_based = "DB"
    cat_extensive = "Ex"
    cat_changes_majority = "CM"
    cat_uses_clustering = "Clus"
    cat_borderline = "BL"
    cat_application = "A"

    def __init__(self):
        pass

    def det_n_to_sample(self, strategy, n_maj, n_min):
        """
        Determines the number of samples to generate
        Args:
            strategy (str/float): if float, the fraction of the difference
                                    of the minority and majority numbers to
                                    generate, like 0.1 means that 10% of the
                                    difference will be generated if str,
                                    like 'min2maj', the minority class will
                                    be upsampled to match the cardinality
                                    of the majority class
        """
        if isinstance(strategy, float) or isinstance(strategy, int):
            return max([0, int((n_maj - n_min) * strategy)])
        else:
            m = "Value %s for parameter strategy is not supported" % strategy
            raise ValueError(self.__class__.__name__ + ": " + m)

    def sample_between_points(self, x, y):
        """
        Sample randomly along the line between two points.
        Args:
            x (np.array): point 1
            y (np.array): point 2
        Returns:
            np.array: the new sample
        """
        return x + (y - x) * self.random_state.random_sample()

    def sample_between_points_componentwise(self, x, y, mask=None):
        """
        Sample each dimension separately between the two points.
        Args:
            x (np.array): point 1
            y (np.array): point 2
            mask (np.array): array of 0,1s - specifies which dimensions
                                to sample
        Returns:
            np.array: the new sample being generated
        """
        if mask is None:
            return x + (y - x) * self.random_state.random_sample()
        else:
            return x + (y - x) * self.random_state.random_sample() * mask

    def sample_by_jittering(self, x, std):
        """
        Sample by jittering.
        Args:
            x (np.array): base point
            std (float): standard deviation
        Returns:
            np.array: the new sample
        """
        return x + (self.random_state.random_sample() - 0.5) * 2.0 * std

    def sample_by_jittering_componentwise(self, x, std):
        """
        Sample by jittering componentwise.
        Args:
            x (np.array): base point
            std (np.array): standard deviation
        Returns:
            np.array: the new sample
        """
        return x + (self.random_state.random_sample(len(x)) - 0.5) * 2.0 * std

    def sample_by_gaussian_jittering(self, x, std):
        """
        Sample by Gaussian jittering
        Args:
            x (np.array): base point
            std (np.array): standard deviation
        Returns:
            np.array: the new sample
        """
        return self.random_state.normal(x, std)

    def sample(self, X, y):
        """
        The samplig function reimplemented in child classes
        Args:
            X (np.matrix): features
            y (np.array): labels
        Returns:
            np.matrix, np.array: sampled X and y
        """
        return X, y

    def fit_resample(self, X, y):
        """
        Alias of the function "sample" for compatibility with imbalanced-learn
        pipelines
        """
        return self.sample(X, y)

    def sample_with_timing(self, X, y):
        begin = time.time()
        X_samp, y_samp = self.sample(X, y)
        _logger.info(self.__class__.__name__ + ": " + ("runtime: %f" % (time.time() - begin)))
        return X_samp, y_samp

    def preprocessing_transform(self, X):
        """
        Transforms new data according to the possible transformation
        implemented by the function "sample".
        Args:
            X (np.matrix): features
        Returns:
            np.matrix: transformed features
        """
        return X

    def get_params(self, deep=False):
        """
        Returns the parameters of the object as a dictionary.
        Returns:
            dict: the parameters of the object
        """
        pass

    def set_params(self, **params):
        """
        Set parameters
        Args:
            params (dict): dictionary of parameters
        """

        for key, value in params.items():
            setattr(self, key, value)

        return self

    def descriptor(self):
        """
        Returns:
            str: JSON description of the current sampling object
        """
        return str((self.__class__.__name__, str(self.get_params())))

    def __str__(self):
        return self.descriptor()


class polynom_fit_SMOTE(OverSampling):
    """
    References:
        * BibTex::
            @INPROCEEDINGS{polynomial_fit_smote,
                            author={Gazzah, S. and Amara, N. E. B.},
                            booktitle={2008 The Eighth IAPR International
                                        Workshop on Document Analysis Systems},
                            title={New Oversampling Approaches Based on
                                    Polynomial Fitting for Imbalanced Data
                                    Sets},
                            year={2008},
                            volume={},
                            number={},
                            pages={677-684},
                            keywords={curve fitting;learning (artificial
                                        intelligence);mesh generation;pattern
                                        classification;polynomials;sampling
                                        methods;support vector machines;
                                        oversampling approach;polynomial
                                        fitting function;imbalanced data
                                        set;pattern classification task;
                                        class-modular strategy;support
                                        vector machine;true negative rate;
                                        true positive rate;star topology;
                                        bus topology;polynomial curve
                                        topology;mesh topology;Polynomials;
                                        Topology;Support vector machines;
                                        Support vector machine classification;
                                        Pattern classification;Performance
                                        evaluation;Training data;Text
                                        analysis;Data engineering;Convergence;
                                        writer identification system;majority
                                        class;minority class;imbalanced data
                                        sets;polynomial fitting functions;
                                        class-modular strategy},
                            doi={10.1109/DAS.2008.74},
                            ISSN={},
                            month={Sept},}
    """

    categories = [OverSampling.cat_extensive]

    def __init__(self, proportion=1.0, topology="star", random_state=None):
        """
        Constructor of the sampling object
        Args:
            proportion (float): proportion of the difference of n_maj and n_min
                                to sample e.g. 1.0 means that after sampling
                                the number of minority samples will be equal
                                to the number of majority samples
            topoplogy (str): 'star'/'bus'/'mesh'
            random_state (int/RandomState/None): initializer of random_state,
                                                    like in sklearn
        """
        super().__init__()
        self.check_greater_or_equal(proportion, "proportion", 0.0)
        if topology.startswith("poly"):
            self.check_greater_or_equal(int(topology.split("_")[-1]), "topology", 1)
        else:
            self.check_isin(topology, "topology", ["star", "bus", "mesh"])

        self.proportion = proportion
        self.topology = topology

        self.set_random_state(random_state)

    @classmethod
    def parameter_combinations(cls, raw=False):
        """
        Generates reasonable paramter combinations.
        Returns:
            list(dict): a list of meaningful paramter combinations
        """
        parameter_combinations = {
            "proportion": [0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0],
            "topology": ["star", "bus", "mesh", "poly_1", "poly_2", "poly_3"],
        }
        return cls.generate_parameter_combinations(parameter_combinations, raw)

    def sample(self, X, y):
        """
        Does the sample generation according to the class paramters.
        Args:
            X (np.ndarray): training set
            y (np.array): target labels
        Returns:
            (np.ndarray, np.array): the extended training set and target labels
        """
        _logger.info(self.__class__.__name__ + ": " + "Running sampling via %s" % self.descriptor())

        self.class_label_statistics(X, y)

        # extracting minority samples
        X_min = X[y == self.min_label]

        # determine the number of samples to generate
        n_to_sample = self.det_n_to_sample(
            self.proportion, self.class_stats[self.maj_label], self.class_stats[self.min_label]
        )

        if n_to_sample == 0:
            _logger.warning(self.__class__.__name__ + ": " + "Sampling is not needed")
            return X.copy(), y.copy()

        samples = []
        if self.topology == "star":
            # Implementation of the star topology
            X_mean = np.mean(X_min, axis=0)
            k = max([1, int(np.rint(n_to_sample / len(X_min)))])
            for x in X_min:
                diff = X_mean - x
                for i in range(1, k + 1):
                    samples.append(x + float(i) / (k + 1) * diff)
        elif self.topology == "bus":
            # Implementation of the bus topology
            k = max([1, int(np.rint(n_to_sample / len(X_min)))])
            for i in range(1, len(X_min)):
                diff = X_min[i - 1] - X_min[i]
                for j in range(1, k + 1):
                    samples.append(X_min[i] + float(j) / (k + 1) * diff)
        elif self.topology == "mesh":
            # Implementation of the mesh topology
            if len(X_min) ** 2 > n_to_sample:
                while len(samples) < n_to_sample:
                    random_i = self.random_state.randint(len(X_min))
                    random_j = self.random_state.randint(len(X_min))
                    diff = X_min[random_i] - X_min[random_j]
                    samples.append(X_min[random_i] + 0.5 * diff)
            else:
                n_combs = len(X_min) * (len(X_min) - 1) / 2
                k = max([1, int(np.rint(n_to_sample / n_combs))])
                for i in range(len(X_min)):
                    for j in range(len(X_min)):
                        diff = X_min[i] - X_min[j]
                        for li in range(1, k + 1):
                            samples.append(X_min[j] + float(li) / (k + 1) * diff)
        elif self.topology.startswith("poly"):
            # Implementation of the polynomial topology
            deg = int(self.topology.split("_")[1])
            dim = len(X_min[0])

            def fit_poly(d):
                return np.poly1d(np.polyfit(np.arange(len(X_min)), X_min[:, d], deg))

            polys = [fit_poly(d) for d in range(dim)]

            for d in range(dim):
                random_sample = self.random_state.random_sample() * len(X_min)
                samples_gen = [polys[d](random_sample) for _ in range(n_to_sample)]
                samples.append(np.array(samples_gen))
            samples = np.vstack(samples).T

        return (np.vstack([X, np.vstack(samples)]), np.hstack([y, np.repeat(self.min_label, len(samples))]))

    def get_params(self, deep=False):
        """
        Returns:
            dict: the parameters of the current sampling object
        """
        return {"proportion": self.proportion, "topology": self.topology, "random_state": self._random_state_init}
