from abc import ABC, abstractmethod
from inspect import signature
import keras.backend as K
import numpy as np


class Prob(ABC):
    @property
    @abstractmethod
    def name(self):
        pass

    @abstractmethod
    def prob(self, h0, *args):
        pass

    @abstractmethod
    def cum(self, h0, *args):
        pass

    def normalized_hindex(self, h0, *args):
        # 1/P(h>=h0) = 1/(1-P(h<h0)) = 1/(1-P(h<=h0-1))
        # For this to work for h0=0, we must make sure that self.cum(-1) = 0
        return 1/(1-self.cum(h0-1, *args))

    @property
    def num_params(self):
        # How many parameters does our probability distribution have?
        return len(signature(self.prob).parameters)-1


class ProbGammaPoisson(Prob):
    @property
    def name(self):
        return 'gamma-poisson'

    def gamma(self, a):
        tf = K.tensorflow_backend.tf
        return tf.exp(tf.lgamma(a))

    def cum(self, h0, alpha, beta):
        raise Exception("Not implemented yet")

    def prob(self, h0, alpha, beta):
        # http://www.math.wm.edu/~leemis/chart/UDR/PDFs/Gammapoisson.pdf
        return (self.gamma(h0 + beta) * alpha**h0) / \
            (self.gamma(beta) * (1+alpha)**(beta+h0) * self.gamma(1+h0))


class ProbLognormal(Prob):
    @property
    def name(self):
        return 'lognormal'

    def _lognormal_cum(self, x, mu, sigma):
        erfc = K.tensorflow_backend.tf.math.erfc
        # https://en.wikipedia.org/wiki/Log-normal_distribution#Cumulative_distribution_function
        return (1/2) * erfc(-(K.log(x) - mu)/(np.sqrt(2) * sigma))

    def cum(self, h0, mu, sigma):
        return self._lognormal_cum(h0+1, mu, sigma) -\
            self._lognormal_cum(0., mu, sigma)

    def prob(self, h0, mu, sigma):
        # Log-normal probability mass function
        # NOTE: We add .001 to h0, since keras/tf cannot handle our
        # prob_lognormal for h0 -> 0 very well. If we don't do this, we tend to
        # always get loss=nan
        return self._lognormal_cum(h0+1+.001, mu, sigma) -\
            self._lognormal_cum(h0+.001, mu, sigma)


class ProbGamma(Prob):
    @property
    def name(self):
        return 'gamma'

    def _gamma_cum(self, x, alpha, beta):
        # https://www.tensorflow.org/api_docs/python/tf/math/igamma
        # igamma(a, x) = 1/Gamma(a) * int_0^x (t^(a-1) exp(-t))
        igamma = K.tensorflow_backend.tf.math.igamma
        # https://en.wikipedia.org/wiki/Gamma_distribution
        return igamma(alpha, beta*x)

    def cum(self, h0, alpha, beta):
        return self._gamma_cum(h0+1, alpha, beta) -\
            self._gamma_cum( 0., alpha, beta)

    def prob(self, h0, alpha, beta):
        # Gamma probability mass function
        # NOTE: We add .001 to h0, since keras/tf cannot handle our prob_gamma
        # for h0 -> 0 very well. If we don't do this, we tend to always get
        # loss=nan Gamma probability mass function
        return self._gamma_cum(h0+1+.001, alpha, beta) -\
            self._gamma_cum(h0+.001, alpha, beta)

