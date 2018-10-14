import numpy as np

def updateF(F,x):
	pass



#suppose I(x,x)=0 


 

class IBEA :
	def __init__(self, alpha,gen, fit, indic, objective):
		self.alpha  = alpha 
		self.gen = gen# nbde génération max
		self.fit = fit 
		self.cfit =0
		self.I = indic
		self.cur_gen =0 # compteur de génération
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
		vI= (lambda x: np.vectorize(self.cur_indic)(self.P,{x}))
		for x in self.P:
			self.F[x]= np.sum(np.exp(-vI(x)/(self.fit*self.cfit)))-1

	def addaptive_fit(self):
		"""
			Adaptive IBEA, rescale les fonctions objectif dans cur_objective construit cur_indic  en fonctions de ces nouvelles fonctions.
			Pas fini

		"""
		self.cur_objective=list()
		for f in self.objective:
			mi = min([(f(x),x) for x in self.P])[1]
			ma = max([(f(x),x) for x in self.P])[1]
			self.cur_objective.append(lambda x:  f[x]-mi/(ma-mi))
		self.cur_indic= self.I(self.cur_objective)

		self.cfit= np.max(np.vectorize(self.cur_indic)(self.P, self.P))
		#self.cfit= max([self.cur_indic(x,y) for x in self.Pand y in self.P])
		self.fit()
		return 


	def environemental_selection(self):
		"""
			step 3
		"""
		while len(self.P) > self.alpha:
			x= np.argmin(self.F)
			del self.P[x]
			updateF(x)


	def updateF(self, el):
		for y in self.P:
			self.F[y]+= np.exp(-self.I({el},{y})/(self.fit*self.cfit))


	def terminaison(self): 
		""" 
			Sep4
		"""	
		return self.gen > self.cur_gen

	def matingsel(self):
		"""
			step 5
		"""
		retP= set()
		for x in self.P:
			for y in self.P:
				if self.better(x, y):
					retP.add(x)
		return retP
	
	def better(self, x, y): 
		"""
			
		"""
		x=None 
		y=None

		#outputObjSpace_X = problem(x)
		#outputObjSpace_Y = problem(y)

		"""UNFINISHED (lol)"""

		return True

	def variation(self):
		"""
			step 6
		"""
		return self.P

	def generate_pop(self):
		return set()

	def run(self):
		self.generate_pop()
		while not self.terminaison():
			self.addaptive_fit()
			self.environemental_selection()
			mat = self.matingsel()
			self.variation()
			self.cur_gen+=1
