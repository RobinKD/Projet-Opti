import numpy as np

def updateF(F,x):
	pass



#suppose I(x,x)=0 



def basicIBEA():
	P= generate_pop()
	while (m< N):
		F=fitness_adjutement(P)
		
		if arret():
			return non_dominated(P)
		P = bin_tournament(P)
		P= mutations(P)
		m+=1

def adaptativeIBEA():
	P= generate_pop()
	for i in range(dim):
		mi, ma = get_bounds
	F,P= environemental_selection


class IBEA :

	def __init__(self, alpha,gen, fit, inidc, objective):
		self.alpha  = alpha
		self.gen = gen
		self.fit = fit
		self.cfit =0
		self.I = inidc
		self.cur_gen =0
		self.objective = objective
		self.cur_objective
		self.curI
		self.P = set()
		self.F = dict()

	
	def fit(self):
		self.F= dict()
		vI= (lambda x: self.vectorize(self.curI)(P,{x}))
		for x in P:
			self.F[x]= np.sum(np.exp(-vI(x)/(self.fit*self.cfit)))-1

	def adaptative_fit():
		self.cur_objective=list()
		for f in self.pbjective:
			mi = min(P,f)
<<<<<<< HEAD
			ma = max(P,fcurI)
			self.cur_objective.append(lambda x: = f[x]-mi/(ma-mi))
=======
			ma = max(P,f)
			self.cur_objective.append(lambda x: f[x]-mi/(ma-mi))
>>>>>>> 14c8a2f3abca0b9651428db474a5767b1c0b167c
		self.cur_indic= self.indic(self.cur_objective)

		self.cfit= max(self.cur_indic(x,y) for x in P and y in P)
		self.fit()
		return 


	def environemental_selection(self):
		while len(P) > self.alpha:
			x_0 = argmin(F)
			del P[x]
			updateF(x)


	def updateF(self,x):
		for y in self.P:
			self.F[y]+= np.exp(-I({x},{y})/(self.fit*self.cfit))


	def terminaision(self): 
		""" TODO"""	
		return self.gen > self.cur_gen

	def matingsel(self):
		retP= set()
		for x in sefl.P:
			for y in self.P:
				if self.better(x, y):
					retP.add(x)
		return retP
	
        def better(self, x, y): # used in matingsel method
            # inequality <= for all objective functions
            # and at least one strict inequality for one objective function

            outputObjSpace_X = problem(x)
            outputObjSpace_Y = problem(y)

            """UNFINISHED (lol)"""

            return true

	def variation(self):

		return P

	def generate_pop(self):
		return set()

	def run(self):
		self.generate_pop()
		while not terminaision():
			self.fitness_adjutement()
			self.environemental_selection()
			mat = self.matingsel()
			self.variation()
			self.cur_gen+=1
