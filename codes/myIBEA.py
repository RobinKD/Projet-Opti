import numpy as np

def eps_indic(A, B, fun):
    fun_a = fun[str(A)]
    fun_b = fun[str(B)]
    epsilon = fun_a - fun_b
    if epsilon[0] < 0 and epsilon[1] < 0:
        return max(epsilon)
    else:
        return min(epsilon)

def myIBEA(fun, pop_size, num_max_gen, fit_scale_fact):
    ibea = IBEA(pop_size, num_max_gen, fit_scale_fact, fun)
    ibea.run()


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
        self.P = np.zeros(1)
        self.F = dict()
        self.indic = dict()

	# def fit(self):
	# 	self.F= dict()
	# 	vI= (lambda x: self.vectorize(self.curI)(P,{x}))
	# 	for x in P:
	# 		self.F[x]= np.sum(np.exp(-vI(x)/(self.fit*self.cfit)))-1

    def adaptive_fit(self):
        if self.fun.number_of_constraints > 0:
            print("Has constraints")
            C = [self.fun.constraint(x) for x in self.P]  # call constraints
            scaled_f = {str(x): self.fun(x) for i, x in enumerate(self.P) if np.all(C[i] <= 0)}
            scaled_f_values = np.array([self.fun(x) for i, x in enumerate(self.P) if np.all(C[i] <= 0)]).T
        else:
            scaled_f = {str(x): self.fun(x) for x in self.P}
            scaled_f_values = np.array([self.fun(x) for x in self.P]).T
        lbound = np.array([min(scaled_f_values[0]), min(scaled_f_values[1])])
        print("\nlbound", lbound)
        ubound = np.array([max(scaled_f_values[0]), max(scaled_f_values[1])])
        print("\nubound", ubound)
        for k in scaled_f.keys():
            print("Op : ", scaled_f[k], "-", lbound, "/(", ubound, " - ", lbound, ")")
            scaled_f[k] = (np.array(scaled_f[k]) - lbound)/(ubound - lbound)
        print("scaled_f", scaled_f)
        for x1 in self.P:
            for x2 in self.P:
                if not np.array_equal(x1, x2):
                    self.indic[str(x1) + str(x2)] = eps_indic(x1, x2, scaled_f)
        # print("\nself.indic", self.indic)
        max_indic = max(self.indic.values())
        print("\nmax_indic", max_indic)
        min_indic = min(self.indic.values())
        print("\nmin_indic", min_indic)
        for x1 in self.P:
            self.F[str(x1)] = 0
            indic_x1 = np.array([self.indic[str(y) + str(x1)] for y in self.P if not np.array_equal(y, x1)])
            self.F[str(x1)] = np.sum(-np.exp(-indic_x1/(max_indic * self.fit_scale)))
        print("\nEnd adaptive fit")
        print("\nFitness values" , self.F)


	# def environemental_selection(self):
	# 	while len(P) > self.alpha:
	# 		x_0 = argmin(F)
	# 		del P[x]
	# 		updateF(x)


	# def updateF(self,x):
	# 	for y in self.P:
	# 		self.F[y]+= np.exp(-I({x},{y})/(self.fit*self.cfit))


	# def terminaision(self): 
	# 	""" TODO"""	
	# 	return self.gen > self.cur_gen

	# def matingsel(self):
	# 	retP= set()
	# 	for x in sefl.P:
	# 		for y in self.P:
	# 			if self.better(x, y):
	# 				retP.add(x)
	# 	return retP
	
    #     def better(self, x, y): # used in matingsel method
    #         # inequality <= for all objective functions
    #         # and at least one strict inequality for one objective function

    #         outputObjSpace_X = problem(x)
    #         outputObjSpace_Y = problem(y)

    #         """UNFINISHED (lol)"""

    #         return true

	# def variation(self):

	# 	return P

    def generate_pop(self):
        return [np.random.rand(self.fun.dimension) for i in range(self.alpha)]

    def run(self):
        self.P = self.generate_pop()
        gen_counter = 0
        while gen_counter < self.max_gen:
            self.adaptive_fit()
            self.environemental_selection()
            mat = self.matingsel()
            self.variation()
            gen_counter += 1
