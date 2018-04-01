from __future__ import division
import warnings
import numpy as np
import scipy as sp
from matplotlib import image
from random import randint
from scipy.misc import logsumexp
from helper_functions import image_to_matrix, matrix_to_image, \
                             flatten_image_matrix, unflatten_image_matrix, \
                             image_difference

warnings.simplefilter(action="ignore", category=FutureWarning)


def k_means_cluster(image_values, k=3, initial_means=None):
    """
    Separate the provided RGB values into
    k separate clusters using the k-means algorithm,
    then return an updated version of the image
    with the original values replaced with
    the corresponding cluster values.

    params:
    image_values = numpy.ndarray[numpy.ndarray[numpy.ndarray[float]]]
    k = int
    initial_means = numpy.ndarray[numpy.ndarray[float]] or None

    returns:
    updated_image_values = numpy.ndarray[numpy.ndarray[numpy.ndarray[float]]]
    """

    image_shape = image_values.shape
    X = np.copy(image_values.reshape(-1, 3))
    if (initial_means is None):
        initial_means = np.random.random((k, 3))


    while True:
        distance = np.linalg.norm(initial_means[:, np.newaxis] - X, axis=2)
        k_map = np.argmin(distance, axis=0)

        ref_means = np.copy(initial_means)
        for i in range(0, k):
            initial_means[i] = np.mean(X[np.where(k_map==i)], axis=0)

        if(np.allclose(ref_means, initial_means, rtol=1.e-10)):
            break

    for i in range(0,k):
        X[np.where(k_map==i)] = initial_means[i]

    return X.reshape(image_shape)


def default_convergence(prev_likelihood, new_likelihood, conv_ctr,
                        conv_ctr_cap=10):
    """
    Default condition for increasing
    convergence counter:
    new likelihood deviates less than 10%
    from previous likelihood.

    params:
    prev_likelihood = float
    new_likelihood = float
    conv_ctr = int
    conv_ctr_cap = int

    returns:
    conv_ctr = int
    converged = boolean
    """
    increase_convergence_ctr = (abs(prev_likelihood) * 0.9 <
                                abs(new_likelihood) <
                                abs(prev_likelihood) * 1.1)

    if increase_convergence_ctr:
        conv_ctr += 1
    else:
        conv_ctr = 0

    return conv_ctr, conv_ctr > conv_ctr_cap


class GaussianMixtureModel:
    """
    A Gaussian mixture model
    to represent a provided
    grayscale image.
    """

    def __init__(self, image_matrix, num_components, means=None):
        """
        Initialize a Gaussian mixture model.

        params:
        image_matrix = (grayscale) numpy.nparray[numpy.nparray[float]]
        num_components = int
        """
        self.shape = image_matrix.shape
        self.image_matrix = image_matrix.reshape(-1)
        self.num_components = num_components
        if(means is None):
            self.means = np.zeros(num_components)
        else:
            self.means = means
        self.variances = np.zeros(num_components)
        self.mixing_coefficients = np.zeros(num_components)
        self.image_pixels = self.image_matrix.size

    def joint_prob(self, val):
        """Calculate the joint
        log probability of a greyscale
        value within the image.

        params:
        val = float

        returns:
        joint_prob = float
        """
        logNormal = -0.5 * np.log(2 * np.pi * self.variances) - (np.square(val - self.means) / (2 * self.variances))

        return logsumexp(a=logNormal, b=self.mixing_coefficients)

    def initialize_training(self):
        """
        Initialize the training
        process by setting each
        component mean to a random
        pixel's value (without replacement),
        each component variance to 1, and
        each component mixing coefficient
        to a uniform value
        (e.g. 4 components -> [0.25,0.25,0.25,0.25]).

        NOTE: this should be called before
        train_model() in order for tests
        to execute correctly.
        """
        self.mixing_coefficients.fill(1.0/self.num_components)
        self.variances.fill(1.0)
        if not self.means.all():
            self.means = np.random.choice(self.image_matrix, self.num_components, replace=False)

    def train_model(self, convergence_function=default_convergence):
        """
        Train the mixture model
        using the expectation-maximization
        algorithm. Since each Gaussian is
        a combination of mean and variance,
        this will fill self.means and
        self.variances, plus
        self.mixing_coefficients, with
        the values that maximize
        the overall model likelihood.

        params:
        convergence_function = function, returns True if convergence is reached
        """

        counter = 0
        converged = False
        prev_likelihood = self.likelihood()

        while not converged:
            # E step Evaluate responsibilities using current parameter values
            # Evaluate gaussians for all components multiplied by mixing coeff
            weighted_normal = self.mixing_coefficients * (1 / np.sqrt(2 * np.pi * self.variances)) * np.exp(-1 * np.square(self.image_matrix[:, np.newaxis] - self.means) / (2 * self.variances))
            # weighted_normal = self.mixing_coefficients * -0.5 * np.log(2 * np.pi * self.variances) - (np.square(self.image_matrix[:, np.newaxis] - self.means) / (2 * self.variances))
            # Get sum of the guassians
            denom = np.sum(weighted_normal, axis=1)
            # Calculate posterior probability posterior = guassian_k / sum(gaussians)
            posterior = np.divide(weighted_normal, denom[:, np.newaxis])

            # M step: Re-estimate params using current responsibilities (posterior)
            # Calculate sum_denom vector
            sum_denom = np.sum(posterior, axis=0)
            # Calculate new means
            self.means = np.sum(posterior * self.image_matrix[:, np.newaxis], axis=0) / sum_denom
            # Calculate new variances
            self.variances = np.sum(posterior * np.square(self.image_matrix[:, np.newaxis] - self.means), axis=0) / sum_denom
            # Calculate new mixture coeffs
            self.mixing_coefficients = sum_denom / self.image_pixels
            
            cur_likelihood = self.likelihood()

            counter, converged = convergence_function(prev_likelihood, cur_likelihood, counter)

            prev_likelihood = cur_likelihood

    def segment(self):
        """
        Using the trained model,
        segment the image matrix into
        the pre-specified number of
        components. Returns the original
        image matrix with the each
        pixel's intensity replaced
        with its max-likelihood
        component mean.

        returns:
        segment = numpy.ndarray[numpy.ndarray[float]]
        """
        weighted_normal = self.mixing_coefficients * (1 / np.sqrt(2 * np.pi * self.variances)) * np.exp(-1 * np.square(self.image_matrix[:, np.newaxis] - self.means) / (2 * self.variances))
        # weighted_normal = self.mixing_coefficients * -0.5 * np.log(2 * np.pi * self.variances) - (np.square(self.image_matrix[:, np.newaxis] - self.means) / (2 * self.variances))
        # Get sum of the guassians
        denom = np.sum(weighted_normal, axis=1)
        # Calculate posterior probability posterior = guassian_k / sum(gaussians)
        posterior = np.divide(weighted_normal, denom[:, np.newaxis])

        cluster_map = np.argmax(posterior, axis=1)

        for i in range(0, self.num_components):
            self.image_matrix[np.where(cluster_map == i)] = self.means[i]

        return self.image_matrix.reshape(self.shape)

    def likelihood(self):
        """Assign a log
        likelihood to the trained
        model based on the following
        formula for posterior probability:
        ln(Pr(X | mixing, mean, stdev)) = sum((n=1 to N), ln(sum((k=1 to K),
                                          mixing_k * N(x_n | mean_k,stdev_k))))

        returns:
        log_likelihood = float [0,1]
        """
        self.variances = np.asarray(self.variances)
        # Get the probability for all pixels in the image
        logNormal = -0.5 * np.log(2 * np.pi * self.variances) \
              - (np.square(self.image_matrix[:, np.newaxis] - self.means)\
              / (2 * self.variances))

        # Sum up the log exponent of all pixels in image
        return np.sum(logsumexp(a=logNormal, b=self.mixing_coefficients, keepdims=True, axis=1))

    def best_segment(self, iters):
        """Determine the best segmentation
        of the image by repeatedly
        training the model and
        calculating its likelihood.
        Return the segment with the
        highest likelihood.

        params:
        iters = int

        returns:
        segment = numpy.ndarray[numpy.ndarray[float]]
        """
        max_likelihood = float("-inf")
        best_means = None
        best_variances = None
        best_mixing_coeffs = None

        self.variances = np.asarray(self.variances)

        for i in range(0, iters):
            # Re-initialize model
            self.initialize_training()
            # Train model
            self.train_model()
            # Get Likelihood
            cur_likelihood = self.likelihood()
            # If greater than previous save
            if cur_likelihood > max_likelihood:
                max_likelihood = cur_likelihood
                best_means = np.copy(self.means)
                best_variances = np.copy(self.variances)
                best_mixing_coeffs = np.copy(self.mixing_coefficients)
            # If less than previous, discard

        self.means = best_means
        self.variances = best_variances
        self.mixing_coefficients = best_mixing_coeffs

        return self.segment()


class GaussianMixtureModelImproved(GaussianMixtureModel):
    """A Gaussian mixture model
    for a provided grayscale image,
    with improved training
    performance."""

    def initialize_training(self):
        """
        Initialize the training
        process by setting each
        component mean using some algorithm that
        you think might give better means to start with,
        each component variance to 1, and
        each component mixing coefficient
        to a uniform value
        (e.g. 4 components -> [0.25,0.25,0.25,0.25]).
        [You can feel free to modify the variance and mixing coefficient
         initializations too if that works well.]
        """
        # Initialize mixing coefficients with uniform distribution
        self.mixing_coefficients.fill(float(1)/self.num_components)
        # Initialize variances to 1
        self.variances.fill(1.0)
        # If no means were passed at object creation, choose a uniform dist between max and min
        if not self.means.all():
            max_img = np.max(self.image_matrix)
            min_img = np.min(self.image_matrix)
            step = float(max_img - min_img) / self.num_components
            self.means = np.arange(min_img, max_img, step)

def new_convergence_function(previous_variables, new_variables, conv_ctr,
                             conv_ctr_cap=10):
    """
    Convergence function
    based on parameters:
    when all variables vary by
    less than 10% from the previous
    iteration's variables, increase
    the convergence counter.

    params:

    previous_variables = [numpy.ndarray[float]]
                         containing [means, variances, mixing_coefficients]
    new_variables = [numpy.ndarray[float]]
                    containing [means, variances, mixing_coefficients]
    conv_ctr = int
    conv_ctr_cap = int

    return:
    conv_ctr = int
    converged = boolean
    """
    mean_converge = ((np.abs(previous_variables[0]) * 0.9 < np.abs(new_variables[0])) &
                               (np.abs(new_variables[0]) < np.abs(previous_variables[0]) * 1.1)).all()

    variance_converge = ((np.abs(previous_variables[1]) * 0.9 < np.abs(new_variables[1])) &
                               (np.abs(new_variables[1]) < np.abs(previous_variables[1]) * 1.1)).all()

    mixing_converge = ((np.abs(previous_variables[2]) * 0.9 < np.abs(new_variables[2])) &
                               (np.abs(new_variables[2]) < np.abs(previous_variables[2]) * 1.1)).all()

    if mean_converge and variance_converge and mixing_converge:
        conv_ctr += 1
    else:
        conv_ctr = 0

    del mean_converge, variance_converge, mixing_converge

    return conv_ctr, conv_ctr > conv_ctr_cap


class GaussianMixtureModelConvergence(GaussianMixtureModel):
    """
    Class to test the
    new convergence function
    in the same GMM model as
    before.
    """

    def train_model(self, convergence_function=new_convergence_function):
        counter = 0
        converged = False


        while not converged:
            prev_vars = [np.copy(self.means), np.copy(self.variances), np.copy(self.mixing_coefficients)]
            weighted_normal = self.mixing_coefficients * (1 / np.sqrt(2 * np.pi * self.variances)) * np.exp(-1 * np.square(self.image_matrix[:, np.newaxis] - self.means) / (2 * self.variances))
            denom = np.sum(weighted_normal, axis=1)
            posterior = np.divide(weighted_normal, denom[:, np.newaxis])

            sum_denom = np.sum(posterior, axis=0)
            self.means = np.sum(posterior * self.image_matrix[:, np.newaxis], axis=0) / sum_denom
            self.variances = np.sum(posterior * np.square(self.image_matrix[:, np.newaxis] - self.means), axis=0) / sum_denom
            self.mixing_coefficients = sum_denom / self.image_pixels
            
            del sum_denom, posterior, weighted_normal, denom
            counter, converged = convergence_function(prev_vars, [self.means, self.variances, self.mixing_coefficients], counter)

def bayes_info_criterion(gmm):
    return np.log(gmm.image_pixels) * 3 * gmm.num_components - 2 * gmm.likelihood()


def BIC_likelihood_model_test():
    """Test to compare the
    models with the lowest BIC
    and the highest likelihood.

    returns:
    min_BIC_model = GaussianMixtureModel
    max_likelihood_model = GaussianMixtureModel
    """
    comp_means = [
        np.array([0.023529412, 0.1254902]),
        np.array([0.023529412, 0.1254902, 0.20392157]),
        np.array([0.023529412, 0.1254902, 0.20392157, 0.36078432]),
        np.array([0.023529412, 0.1254902, 0.20392157, 0.36078432, 0.59215689]),
        np.array([0.023529412, 0.1254902, 0.20392157, 0.36078432, 0.59215689,
         0.71372563]),
        np.array([0.023529412, 0.1254902, 0.20392157, 0.36078432, 0.59215689,
         0.71372563, 0.964706])
    ]

    img = image_to_matrix('images/party_spock.png')

    max_likelihood = float("-inf")
    min_BIC = float("inf")
    max_likelihood_model = None
    min_BIC_model = None

    k = 2
    for means in comp_means:
        gmm = GaussianMixtureModel(np.copy(img), k)
        gmm.initialize_training()
        gmm.means = means
        # gmm.train_model()
        cur_likelihood = gmm.likelihood()
        cur_BIC = bayes_info_criterion(gmm)
        if max_likelihood < cur_likelihood:
            max_likelihood = cur_likelihood
            max_likelihood_model = gmm
        if min_BIC > cur_BIC:
            min_BIC = cur_BIC
            min_BIC_model = gmm
        k += 1

    del comp_means

    return min_BIC_model, max_likelihood_model


def BIC_likelihood_question():
    """
    Choose the best number of
    components for each metric
    (min BIC and maximum likelihood).

    returns:
    pairs = dict
    """
    bic = 7
    likelihood = 7
    pairs = {
        'BIC': bic,
        'likelihood': likelihood
    }
    return pairs

def return_your_name():
    # return your name
    return "Bian Du"

def bonus(points_array, means_array):
    """
    Return the distance from every point in points_array
    to every point in means_array.

    returns:
    dists = numpy array of float
    """
    end = means_array.shape[0]
    middle = int(end / 2)
    first = points_array[:, np.newaxis] - means_array[0:middle]
    first = np.linalg.norm(first, axis=2)
    second = points_array[:, np.newaxis] - means_array[middle:end]
    second = np.linalg.norm(second, axis=2)
    dists = np.append(first, second, axis=1)
    del first, second
    return dists
