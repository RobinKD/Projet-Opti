import numpy as np
import MatingVariation as mv

def eps_indic(i, j, fun):
    fun_a = fun[i]
    fun_b = fun[j]
    epsilon = fun_a - fun_b
    # if epsilon[0] < 0 and epsilon[1] < 0:
    #     return max(epsilon)
    # elif epsilon[0] > 0 and epsilon[1] < 0:
    #     return epsilon[0]
    # elif epsilon[0] < 0 and epsilon[1] > 0:
    #     return epsilon[1]
    # else:
    #     return max(epsilon)
    return max(epsilon)


#import cProfile, pstats
#from pstats import SortKey
def myIBEA(fun, pop_size, num_max_gen, fit_scale_fact):
    ibea = IBEA(pop_size, num_max_gen, fit_scale_fact, fun)
    #pr = cProfile.Profile()
    #pr.enable()
    ibea.run()
    pr.disable()
    #sortby = SortKey.CUMULATIVE
    #ps = pstats.Stats(pr).sort_stats(sortby)
    #ps.print_stats()

def unique_pop(pop, eps):
    unique = np.unique(pop, axis=0)
    same = True
    for x in unique:
        dist = np.linalg.norm(unique[0] - x)
        if dist > eps:
            same = False
            continue
    return same


class IBEA :
    def __init__(self, alpha, max_gen, fit_scale_factor, fun):
        self.alpha  = alpha
        self.max_gen = max_gen
        self.fit_scale = fit_scale_factor
        # self.cfit =0
        # self.cur_gen =0
        self.fun = fun
        self.obj = []
        # self.cur_objective
        # self.curI
        self.P = np.zeros(alpha)
        self.F = np.zeros(alpha)
        self.indic = np.zeros((alpha, alpha))
        self.max_indic = 0


    def adaptive_fit(self):
        taille_pop = len(self.P)
        self.F = np.zeros(taille_pop)
        self.indic = np.zeros((taille_pop, taille_pop))
        if self.fun.number_of_constraints > 0:
            print("Has constraints")
            C = [self.fun.constraint(x) for x in self.P]  # call constraints
            scaled_f = np.array([self.fun(x) for i, x in enumerate(self.P) if np.all(C[i] <= 0)])
        else:
            scaled_f = np.array([self.fun(x) for x in self.P])
        scaled_f_transp = scaled_f.T
        lbound = np.array([min(scaled_f_transp[0]), min(scaled_f_transp[1])])
        #print("lbound", lbound)
        ubound = np.array([max(scaled_f_transp[0]), max(scaled_f_transp[1])])
        #print("ubound", ubound)
        div_bound = ubound - lbound
        if div_bound[0] == 0 or div_bound[1] == 0:
            print("div by zero will happen", ubound, lbound, unique_pop(self.P))
            # print("Scaled f is", scaled_f_transp)
        # for i, k in enumerate(scaled_f):
        #     # print("Op : ", scaled_f[i], "-", lbound, "/(", ubound, " - ", lbound, ")")
        #     scaled_f[i] = (np.array(scaled_f[i]) - lbound)/(ubound - lbound)
        scaled_f -= lbound
        scaled_f *= np.array([1,1])/(ubound - lbound)
        # print("scaled_f", scaled_f)
        for i, x1 in enumerate(self.P):
            for j, x2 in enumerate(self.P):
                if i != j:
                    self.indic[i][j] = eps_indic(i, j, scaled_f)
                else:
                    self.indic[i][j] = 0 # None quand indicateur avec lui meme
        # print("\nself.indic", self.indic)
        self.max_indic = np.amax(np.absolute(self.indic))
        #print("max_indic", max_indic)
        for i, x1 in enumerate(self.P):
            indic_x1 = np.array([self.indic[j][i] for j, y in enumerate(self.P) if i != j])
            if (self.max_indic * self.fit_scale) == 0:
                print("Div by zero will happen", self.max_indic, self.fit_scale)
                print("Indicator values", self.indic)
            diviseur = 1 / (self.max_indic * self.fit_scale)
            self.F[i] = np.sum(-np.exp(-indic_x1 * diviseur))
        # print("\nEnd adaptive fit")
        # print("\nFitness values" , self.F)

    def updateF(self, ind, tmp_indic):
        for i, fitness in enumerate(self.F):
            self.F[i] = fitness + np.exp(-tmp_indic[ind][i]/(self.max_indic * self.fit_scale))

    def environmental_selection(self):
        # print("\nNew environmental selection")
        # print(self.P.shape, self.indic.shape, self.F.shape)
        while len(self.P) > self.alpha:
            # print("Size pop reduced to", len(self.P))
            x = 0
            min = self.F[0]
            for i, val in enumerate(self.F):
                if val < min:
                    min = val
                    x = i
            tmp_indic = self.indic
            self.P = np.delete(self.P, x, axis=0)
            self.indic = np.delete(self.indic, x, axis=0)
            self.F = np.delete(self.F, x, axis=0)
            self.updateF(x, tmp_indic)

    def run(self):
        self.P = mv.generateInitialPopulation(self.alpha, self.fun.dimension, 5, -5)
        # ubound = self.fun.upper_bounds
        # lbound = self.fun.lower_bounds
        # self.P = mv.generateInitialPopulation(self.alpha, self.fun.dimension, ubound, lbound)
        gen_counter = 0
        while gen_counter < self.max_gen and not unique_pop(self.P, 0.000001):
            self.adaptive_fit()
            self.environmental_selection()
            matingPool = mv.binary_tour_sel(self.P, self.F)
            matingPool = mv.recombination(matingPool)
            matingPool = mv.mutation(matingPool)
            self.P = np.concatenate((self.P, matingPool), axis=0)
            gen_counter += 1
            # print("Generation ", gen_counter)
