#import re
import abc
from scipy.linalg import cholesky
from scipy.stats import norm, multivariate_normal
import scipy.integrate as integrate
from scipy.optimize import minimize #, LinearConstraint
import numpy as np

class LatentVariableModel(abc.ABC):
    def __init__(self, thresholds: np.ndarray, notionals: np.ndarray=None, lgds: np.ndarray=None):
        if len(thresholds.shape)>2:
            raise Exception('Thresholds must either be a one dimensional vector to model two states (default and non-default) or a matrix defining more states (ratings).')
        self._thresholds = thresholds
        self._notionals = notionals
        self._lgds = lgds
        self._simulated = None

    @abc.abstractmethod
    def random_variates_shape(self):
        """Return shape of uniform random variates that are needed to simulate the respective defaults/payoffs/ratings
        """
        pass

    def n_loans(self):
        """Return number of loans

        Returns:
           int: Number of loans in Model
        """
        return self._thresholds.shape[0]

    def simulate(self, X: np.ndarray=None):
        if X is None:
            X = np.random.uniform(low=0.0, high=1.0, size=self.random_variates_shape)
        X_sim = self._simulate(X)
        if len(self._thresholds.shape) == 1:
            self._simulated = (X_sim > self._thresholds).astype(float)
        else:
            Y = np.zeros(X_sim.shape[0])
            for i in range(self._thresholds.shape[1]):
                Y[X_sim>self._thresholds[i]] = float(i)
            self._simulated = Y

    def compute_default_loss(self):
        if self._simulated is None:
            raise Exception('No simulation run. Please first call simulate.')
        if self._notionals is None:
            raise Exception('No notionals provided, please first set notionals.')
        if self._lgds is None:
            raise Exception('No lgds provided, please first set lgds.')
        defaults = self._simulated==0.0
        return self._lgds[defaults].dot(self._notionals[defaults])

    @abc.abstractmethod
    def _simulate(self, X: np.ndarray):
        pass

class VasicekModel(LatentVariableModel):
    def __init__(self, betas: np.array, alphas: np.array=None, thresholds: np.ndarray=None, pds: np.ndarray=None, notionals: np.ndarray=None, lgds: np.ndarray=None):
        if (thresholds is None) and (pds is None):
            raise Exception('Either a threshold or default probabilities (pds) must be specified')
        if (thresholds is not None) and (pds is not None):
            raise Exception('A threshold and default probabilities cannot be specified at the same time')
        if pds is not None:
            thresholds = norm.ppf(pds)
        if (alphas is not None) and (len(alphas.shape)!=2):
            raise Exception('alphas must be either None or a two dimensional matrix.')
        if (alphas is not None) and (alphas.shape[0] != thresholds.shape[0]):
            raise Exception('Number of alphas rows must equal number of loans')
        super(VasicekModel, self).__init__(thresholds, notionals, lgds)
        self._betas = betas
        self._sqrt_betas = np.sqrt(1-self._betas**2)
        self._alphas = alphas

    def random_variates_shape(self):
        if self._alphas is None: # only one systematic facor
            return (self._thresholds.shape[0]+1, )
        return (self._thresholds.shape[0]+self._alphas.shape[1],)

    def _simulate(self, X: np.ndarray):
        X_normal = norm.ppf(X)
        if self._alphas is None:
            idiosyncratic = X_normal[:-1]
            systematic = X_normal[-1:]
            Y = self._betas*systematic + self._sqrt_betas*idiosyncratic
        else:
            idiosyncratic = X_normal[:-self._alphas.shape[1]]
            systematic = self._alphas.dot(X_normal[-self._alphas.shape[1]:])
            Y = self._betas*systematic + self._sqrt_betas*idiosyncratic
        return Y


class LoanPortfolio:
    def __init__(self, lgd, r, pd, beta):
        self.lgd = lgd
        self.__pd = pd
        self.r = r
        self.__beta = beta
        self._norm_ppf = norm.ppf(self.__pd)
        self._sqrt_beta = np.sqrt(1-self.__beta**2)

    @property
    def pd(self):
        return self.__pd

    @pd.setter  
    def pd(self, value):
        self.__pd = value
        self._norm_ppf = norm.ppf(self.__pd)

    @property
    def beta(self):
        return  self.__beta

    @beta.setter
    def beta(self,value):
        self.__beta = value
        self._sqrt_beta = np.sqrt(1-self.__beta**2)

    def __setattr__(self, a, v):
        propobj = getattr(self.__class__, a, None)
        if isinstance(propobj, property):
            if propobj.fset is None:
                raise AttributeError("can't set attribute")
            propobj.fset(self, v)
        else:
            super(LoanPortfolio, self).__setattr__(a, v)

    def compute_expected_return(self):
        """Compute expected value of return

        Returns:
            [type]: [description]
        """
        return (1+self.r)*(1-self.pd) + (1-self.lgd)*self.pd-1.0
    
    def compute_loss(self, X)->np.array:
        """Compute portfolio loss for realizations X of systemic factor


        Args:
            X (np.array): Array containing realizations of normally distributed systemic factor.

            This method computes losses according to
            :math: LGD\cdot \phi\left( \frac{\phi^{-1}(pd)+\beta X}{\sqrt{1-\beta^2}}\right)


        Returns:
            np.array: Array of losses
        """
        return self.lgd*norm.cdf((self._norm_ppf+self.beta*X)/self._sqrt_beta, loc=0, scale=1.0)

    def compute_return(self, X: np.ndarray)->np.array:
        """Compute portfolio return for realizations X of systemic factor

        Args:
            X (np.array): Array containing realizations of normally distributed systemic factor.

        Returns:
            np.array: Array of return
        """
        #raise NotImplementedError()
        return -(self.lgd+self.r)*norm.cdf((self._norm_ppf+self.beta*X)/self._sqrt_beta, loc=0, scale=1.0) + (1.0+self.r) -1
        
  