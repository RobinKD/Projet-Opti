import numpy as np
import random 

import math



#suppose I(x,x)=0 


 

class IBEA :
	def __init__(self, alpha, dim, gen, fit, objective,indic):
		self.alpha  = alpha 
		self.gen = gen# nbde génération max
		self.dim = dim
		self.outdim = objective.number_of_objectives
		self.fitval = fit 
		self.cfit =0
		self.I = indic
		self.cur_gen =0 # compteur de génération
		self.objective = objective
		self.cur_objective =dict()
		self.cur_indic = dict()
		self.P = self.generate_pop(alpha, objective)
		self.F = dict()

		self.static_objective= dict()

	def fit(self):
		"""
			Basic IBEA step 1, map x in P to F(x)
			calculle pour tout x_1 de p : $\sum_{x_2\in P\{x1\}}-e^{-I({x_2,x_1})/k}$
		"""
		#self.F.clear()
		#vI= (lambda x: np.vectorize(self.cur_indic)(np.array(list(self.P)),x))
		scale = self.fitval*self.cfit
		#print(self.cfit)
		if scale == 0:
			print("div par zero")
		for x in self.P:
			self.F[x]=-np.sum(np.exp(np.array((list(self.cur_indic[x].values())))/scale))+1

	def addaptive_fit(self):
		"""
			Adaptive IBEA, rescale les fonctions objectif dans cur_objective construit cur_indic  en fonctions de ces nouvelles fonctions.
			
		"""
		
		# rescale objective
		#fx= [self.objective(x) for x in self.P]

		for x in self.P:
			if x not in self.static_objective:
				self.static_objective[x]= self.objective(x)
		fx= list(self.static_objective.values())
		
		mi = np.min(fx,axis= 0)
		ma = np.max(fx,axis= 0)
		self.cur_objective.clear()
		for x in self.P:
			self.cur_objective[x]= (self.static_objective[x]-mi)/(ma-mi)
		#self.cur_indic= (lambda a,b :self.I(self.cur_objective, a,b))

		self.cur_indic.clear()
		self.cfit= 0


		for x, obx in self.cur_objective.items():#la partie la plus lente (encore a cause de I_epsilon)
			self.cur_indic[x]= dict()
			for y, oby in self.cur_objective.items():
				cur = -self.I(oby, obx)
				self.cur_indic[x][y]= cur#on inverse x y pour avoir une opti dans fit
				self.cfit=max(self.cfit, abs(cur))
				#print(self.static_objective[x], self.static_objective[y])
		#print(self.cfit)

		self.fit()
		return 


	def environemental_selection(self):
		"""
			step 3
		"""
		l=len(self.P)
		while l > self.alpha:
			x= min(self.F.items(), key= (lambda x: x[1]))[0]
			self.P.remove(x)
			l-=1
			del self.F[x]
			self.updateF(x)


	def updateF(self, el):
		"""
			recompute F without el :
			F_1(x)= F(x)+ e^{-I(\{el\},\{x\}/(k*c)}
		"""
		scale = self.fitval*self.cfit
		for y in self.P:
			self.F[y]+= math.exp(self.cur_indic[y][el]/(scale))


	def terminaison(self): 
		""" 
			Sep4
		"""	
		return self.gen <= self.cur_gen

	
	
	def mating_selection(self):
		"""Mating selection aims at picking promising solutions for variation and
		usually is performed in a randomized fashion. Perform binary tournament 
		selection with replacement on P in order to fill the temporary mating pool P_	
		Params:
			P ---> pool of the population
		"""
		P_ = list()
		i=0
		for i in range(len(self.P)): #We want to have the same size as the original population FAUX
			a,b =random.sample(self.P, 2)
			P_.append(max(a,b, key= (lambda x : self.F[x]))) #argmax{F[x], F[y]}
			i+=1
		return P_ 


	

	def variation(self, P_,mut_rate=0.05,mu=2.5):
		"""the mutation operator modifies individuals by changing small 
		parts in the associated vectors according to a given mutation rate.
		
		Params:
			P_ ---> mating pool
			mut_rate ---> mutation rate by default is 1.0
			mu ---> 25
		"""
		size_mutation = len(P_)*mut_rate
		Uppers = self.most_vect(P_, max)
		Lowers = self.most_vect(P_, min)
		sample = random.sample(P_, int(size_mutation))
		for ind in sample :
			pMut = list(ind)
			for j in range(self.dim):
				#Step 1: Choose a random number u in [0; 1).
				u = random.random()
				#Step 2: Calculate the two parameters (δL or δR)
				#Step 3: the mutated solution p_mut for a particular variable j is created
				if u <= 0.5:
					sigma_L = math.pow(2*u,1/(mu+1))-1
					pMut[j] += sigma_L*(ind[j]-Lowers[j])
				else:
					sigma_R = 1 - math.pow(2*(1-u),1/(mu+1))
					pMut[j] += sigma_R*(Uppers[j]-ind[j])
			self.P.add(tuple(pMut)) #np.append(P_,pMut, axis = 0)
		self.P.update(P_)


	def most_vect(self,P,f):
		if len(P)!=0:
			most  = list(P[0])
			for x in P:
				i=0
				for i in range(self.dim) :#for i in range(len(most)):
					most[i]= f(x[i], most[i])
					i+=1
		return tuple(most)

	def recombination(self, P_,recom_rate=1.0,mu=2.5):
		"""The recombination operator takes a certain number of parents and creates a 
		predefined number of children by combining parts of the parents. To mimic the 
		stochastic nature of evolution, a crossover probability is associated with this
		operator.
		
		Params:
			P_ ---> mating pool
			recom_rate ---> recombination rate by default is 1.0
		"""
		size_recom = len(P_)*recom_rate

		sample = random.sample(P_,int(size_recom)) #Permutation
		for i in range(0,len(sample)-1,2):
			#Pick two individuals to recombine to generate offspring
			parent0 = sample[i]
			parent1 = sample[i+1]
			#Step 1: Choose a random number u 2 [0; 1).
			u = random.random()
			#Step 2: Calculate  beta(q) using equation
			if u <= 0.5:
				beta_q = math.pow(2*u,1/(mu+1))
			else : 
				beta_q = math.pow(1/(2*(1-u)),1/(mu+1))
			child0 = [0]* self.dim
			child1 = [0]* self.dim 
			#Step 3: Compute children solutions using equations for all variables
			for j in range(self.dim):#SBX algo to recombine cho
				child0[j] = 0.5*((1+beta_q)*parent0[j]+(1-beta_q)*parent1[j])
				child1[j] = 0.5*((1-beta_q)*parent0[j]+(1+beta_q)*parent1[j])
			P_.append(tuple(child0))  
			P_.append(tuple(child1))
		return P_  
	


	def generate_pop(self, pop_size, fun):
		"""
			generate a pop_size set of random vectors in [-5, 5]^n
		"""
		out = set()
		mod_affine= (lambda x, u,l: x*(u-l)-u)
		for i in range(pop_size):
			cur = list()
			for j in range(self.dim):
				u= fun.upper_bounds[j]
				l= fun.lower_bounds[j]
				cur.append(mod_affine(random.random(),u,l))
			out.add(tuple(cur))
		return out

	def run(self):
		while not self.terminaison():
			self.addaptive_fit()
			self.environemental_selection()
			mat = self.mating_selection()
			mat= self.recombination(mat)
			self.variation(mat)
			self.cur_gen+=1




def I_epsilon(l, A, B):#gerer les positifs négatifs
	m=-100
	i=0
	while i < l:
		c= A[i]-B[i]
		if m < c:
			m=c
		i+=1
	return m
def bin_epsilon(A, B):#gerer les positifs négatifs
	x= A[0]-B[0]
	y= A[1]-B[1]
	if x >y:
		return x
	else :
		return y

#import cProfile, pstats
#from pstats import SortKey

def myIBEA(fun, pop_size, num_max_gen, fit_scale_fact):
	ibea = IBEA(pop_size, fun.dimension, num_max_gen, fit_scale_fact, fun, I_epsilon)
	if fun.number_of_objectives == 2:
		I_eps = bin_epsilon
	else:
		I_eps =(lambda x,y : I_epsilon(fun.number_of_objectives, x,y))
	ibea = IBEA(pop_size, fun.dimension, num_max_gen, fit_scale_fact, fun, I_eps)
	#pr = cProfile.Profile()
	#pr.enable()
	ibea.run()
	#pr.disable()
	#sortby = SortKey.CUMULATIVE
	#ps = pstats.Stats(pr).sort_stats(sortby)
	#ps.print_stats()
