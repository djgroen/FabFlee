import autograd.numpy as anp
import numpy as np
from pymoo.core.decomposition import Decomposition

class Tchebicheff2(Decomposition):
    def _do(self, F, weights, **kwargs):
        weights = np.where(weights == 0, 0.00001, weights)
        v = anp.abs(F - self.utopian_point) / weights
        tchebi = v.max(axis=1)
        return tchebi

