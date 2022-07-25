from pymoo.factory import get_problem
from pymoo.optimize import minimize

from pymoo.algorithms.moo.nsga2 import NSGA2

problem = get_problem("zdt2")
algorithm = NSGA2(pop_size=10)
res = minimize(problem,
               algorithm,
               ('n_gen', 100),
               verbose=True,
               return_least_infeasible=False,
               seed=1)

# all decision variables
print('\nall decision variables:')
print(res.pop.get("X"))
print('\n')

# all objective variables
print('\nall objective variables:')
print(res.pop.get("F"))
print('\n')

# non-dominated set
print('\nnon-dominated set - decision variables:')
print(res.X)
print('\n')
print('\nnon-dominated set - objective variables:')
print(res.F)


