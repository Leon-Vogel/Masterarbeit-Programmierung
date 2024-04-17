from pymoo.algorithms.moo.ctaea import CTAEA
from pymoo.algorithms.moo.moead import MOEAD
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.rvea import RVEA
from pymoo.algorithms.moo.unsga3 import UNSGA3
from pymoo.algorithms.soo.nonconvex.ga import GA
#from pymoo.core.repair import Repair
from pymoo.util.reference_direction import default_ref_dirs


def get_alg(name, n_objectives, crossover, mutation, sampling, duplicate_elimination, pressure=2):
    assert n_objectives <= 3, 'currently, max. 3 objectives are allowd. please configure custom ref_dirs (see code below), to enable more objectives'
    
    # wichtig für many objectives verfahren. es können max. n=3 ziele definiert sein.
    # darüber hinaus kann die default funktion nicht verwendet werden. die ref_dirs müssen selbst definiert werden.
    # https://github.com/anyoptimization/pymoo/blob/main/pymoo/util/reference_direction.py
    ref_dirs = default_ref_dirs(n_objectives)  
    
    #repair = Repair()
    

    if name == 'moead': # Well-known multi-objective optimization algorithm based on decomposition.
        return MOEAD( 
            ref_dirs=ref_dirs, # defines pop_size via ref_dirs
            sampling=sampling,
            crossover=crossover,
            mutation=mutation
        )
    if name == 'nsga2': # Well-known multi-objective optimization algorithm based on non-dominated sorting and crowding.
        return NSGA2(
            sampling=sampling,
            crossover=crossover,
            mutation=mutation,
            eliminate_duplicates=duplicate_elimination,
        )
    if name == 'rvea': # A reference direction based algorithm used an angle-penalized metric.
        return RVEA( 
            ref_dirs=ref_dirs,
            sampling=sampling,
            crossover=crossover,
            mutation=mutation,
            eliminate_duplicates=duplicate_elimination,
        )
    if name == 'ctaea': # An algorithm with a more sophisticated constraint-handling for many-objective optimization algoritms.
        return CTAEA( 
            ref_dirs=ref_dirs,
            sampling=sampling,
            crossover=crossover,
            mutation=mutation,
            eliminate_duplicates=duplicate_elimination
        )
    if name == 'unsga3': # A generalization of NSGA-III to be more efficient for single and bi-objective optimization problems.
        return UNSGA3( 
            ref_dirs=ref_dirs,
            sampling=sampling,
            crossover=crossover,
            mutation=mutation,
            eliminate_duplicates=duplicate_elimination,
        )
    
    if name == 'GA':
        algo = GA(
            sampling=sampling,
            crossover=crossover,
            mutation=mutation
        )
        algo.mating.selection.pressure = pressure
        return algo
    raise Exception(f'{name} is not a defined algorithm')
    # wenn erfordert, können auch weitere verfahren hinzugefügt werden...