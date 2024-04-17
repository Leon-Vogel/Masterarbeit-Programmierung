import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from pymoo.core.crossover import Crossover
from pymoo.core.mutation import Mutation
import random
import numpy as np
from scipy.stats import kendalltau
from copy import deepcopy


def genome_distance(par1, par2):
    """Kendall Tau Distanz oder Spearman's footrule distance:
    Diese sind spezifisch für die Messung der Distanz zwischen zwei Permutationen.
    Kendall Tau misst die Anzahl der inversen Paare zwischen zwei Permutationen,
    während Spearman's footrule die Summe der absoluten Differenzen zwischen
    den Positionen in zwei Permutationen ist."""
    tau, _ = kendalltau(par1, par2)
    normalized_distance = (1 - tau) / 2  # Kendall tau is between -1 and 1.
    return normalized_distance  # 0: equal vectors, 1: completely different vectors


def get_num_genes_to_transfer_from_par1(par1, par2):
    """ziel: exploitation beim crossover sicherstellen, da exploitation der mutation vorbehalten werden soll.
    demnach: bei sehr unterschiedlichen parents dürfen nur wenig gene vermischt werden.
    bei sehr ähnlichen parents hingegen dürfen mehr gene vermischt werden.
    """
    norm_kendalltau_dist = genome_distance(par1, par2)
    mix_probability = 0.6 - (norm_kendalltau_dist / 2)
    min_genes = 2
    max_genes = max(3, int(round(len(par1) * mix_probability)))
    return min_genes, max_genes


def create_child_genome(par1, par2):
    jobs = list(set([g for g in par1]))
    offspr = [None] * len(par1)
    min_jobs_from_par1, max_jobs_from_par1 = get_num_genes_to_transfer_from_par1(par1, par2)
    jobs_from_par1 = random.sample(jobs, k=random.randint(min_jobs_from_par1, max_jobs_from_par1))
    for idx, j in enumerate(par1):
        if j in jobs_from_par1:
            offspr[idx] = par1[idx]
    par2_remaining_jobs = [j for j in par2 if j not in jobs_from_par1]
    for i in range(len(offspr)):
        if offspr[i] is None:
            offspr[i] = par2_remaining_jobs.pop(0)
    return offspr


class FlipMutation(Mutation):
    def _do(self, problem, X, **kwargs):
        Y = np.copy(X)
        for y in Y:
            y = self._swap_items(y)
        return Y
    
    @staticmethod
    def _swap_items(ind):
        idx_1, idx_2 = np.random.choice(len(ind), 2)
        ind_new = deepcopy(ind)
        ind_new[idx_1] = ind[idx_2]
        ind_new[idx_2] = ind[idx_1]
        return ind_new


class JobOrderCrossoverByKendallDistance(Crossover):
    def __init__(self, prob):
        super().__init__(prob=prob, n_parents=2, n_offsprings=2)

    def _do(self, problem, all_parents, **kwargs):
        n_matings = len(all_parents[0])
        Y = [[], []]
        for i in range(n_matings):
            parent1 = list(all_parents[0][i])
            parent2 = list(all_parents[1][i])
            Y[0].append(create_child_genome(parent1, parent2))
            Y[1].append(create_child_genome(parent2, parent1))

        z = np.array(Y)
        return z


def swap_multiple_with_logarithm(ind):
    """tauscht zwei zufaellige gene (n mal).
    wie oft gene getauscht werden, hängt mit der groesse des individuums zusammen.
    je mehr gene, desto mehr tauschvorgaenge. die anzahl vorgaenge nimmt jedoch nur logarithmisch mit der genanzahl zu.
    sonst hat man bei sehr großen individuen nur noch einen shuffling-effekt, was kein wirklicher nachbar mehr im loesungsraum ist"""
    n_swaps = int(round(len(ind)**0.5))
    positions = list(range(len(ind)))
    for _ in range(n_swaps):
        pos1, pos2 = random.sample(positions, k=2)
        ind[pos1], ind[pos2] = ind[pos2], ind[pos1]
    return ind


class SwapMultipleJobsMutationByLogarithm(Mutation):
    def _do(self, problem, X, **kwargs):
        Y = np.copy(X)
        for y in Y:
            y = swap_multiple_with_logarithm(y)
        return Y


if __name__ == "__main__":
    a = [1,2,3,4,5,6,7,8,9,0]
    b = [5,4,7,1,2,9,6,3,0,8]
    b = [1,2,3,4,5,6,7,8,9,0]
    print(get_num_genes_to_transfer_from_par1(a, b))
