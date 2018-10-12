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

	def __init__(self, alpha,gen, fit, indic, objective):
		self.alpha  = alpha
		self.gen = gen
		self.fit = fit
		self.cfit =0
		self.I = indic
		self.cur_gen =0
		self.objective = objective
		self.cur_objective
		self.cur_indic
		self.P = set()
		self.F = dict()

	
	def fit(self):
		"""
			Basic IBEA step 1, map x in P to F(x)
		"""
		self.F= dict()
		vI= (lambda x: self.vectorize(self.cur_indic)(P,{x}))
		for x in P:
			self.F[x]= np.sum(np.exp(-vI(x)/(self.fit*self.cfit)))-1

	def addaptive_fit(self):
		"""
			Adaptive IBEA, rescale les fonctions objectif dans cur_objective construit cur_indic  en fonctions de ces nouvelles fonctions.
			Pas fini

		"""
		self.cur_objective=list()
		for f in self.pbjective:
			mi = min(P,f)
			ma = max(P,fcurI)
			self.cur_objective.append(lambda x: = f[x]-mi/(ma-mi))
		self.cur_indic= self.indic(self.cur_objective)

		self.cfit= max(self.cur_indic(x,y) for x in P and y in P)
		self.fit()
		return 


	def environemental_selection(self):
		"""
			step 3
		"""
		while len(P) > self.alpha:
			x_0 = argmin(F)
			del P[x]
			updateF(x)


	def updateF(self,x):
		for y in self.P:
			self.F[y]+= np.exp(-I({x},{y})/(self.fit*self.cfit))


	def terminaision(self): 
		""" 
			Sep4
		"""	
		return self.gen > self.cur_gen

	def matingsel(self):
		"""
			step 5
		"""
		retP= set()
		for x in sefl.P:
			for y in self.P:
				if self.better(x, y):
					retP.add(x)
		return retP
	
	def better(self, x, y): 
		"""
			
		"""

		outputObjSpace_X = problem(x)
		outputObjSpace_Y = problem(y)

		"""UNFINISHED (lol)"""

				return true

	def variation(self):
		"""
			step 6
		"""
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
