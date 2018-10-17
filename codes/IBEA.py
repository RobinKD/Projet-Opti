import numpy as np
import random 

import math



#suppose I(x,x)=0 


 

class IBEA :
	def __init__(self, alpha, dim, gen, fit, objective,indic):
		self.alpha  = alpha 
		self.gen = gen# nbde génération max
		self.dim = dim
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
		self.F= dict()
		#vI= (lambda x: np.vectorize(self.cur_indic)(np.array(list(self.P)),x))
		for x in self.P:
			self.F[x]= -np.sum(np.exp(np.array([-self.cur_indic[(y,x)] for y in self.P])/(self.fitval*self.cfit)))+1

	def addaptive_fit(self):
		"""
			Adaptive IBEA, rescale les fonctions objectif dans cur_objective construit cur_indic  en fonctions de ces nouvelles fonctions.
			
		"""
		
		# rescale objective
		#fx= [self.objective(x) for x in self.P]
		for x in self.P:
			self.static_objective[x]= self.objective(x)
		fx= list(self.static_objective.values())
		mi = np.min(fx,axis= 0)
		ma = np.max(fx,axis= 0)
		self.cur_objective.clear()
		
		for x in self.P:
			self.cur_objective[x]= self.static_objective[x]*(ma-mi)-mi
		#self.cur_indic= (lambda a,b :self.I(self.cur_objective, a,b))

		self.cur_indic.clear()
		self.cfit= 0

		for x in self.P:
			for y in self.P:
				cur= self.I(self.cur_objective[x],self.cur_objective[y])
				self.cur_indic[(x,y)]= cur
				self.cfit+=cur
		#self.cfit = max(self.cur_indic(x,y) for x in self.P for y in self.P)
		self.fit()
		return 


	def environemental_selection(self):
		"""
			step 3
		"""
		while len(self.P) > self.alpha:
			x= min(self.F.items(), key= (lambda x: x[1]))[0]
			self.P.remove(x)
			self.updateF(x)


	def updateF(self, el):
		"""
			recompute F without el :
			F_1(x)= F(x)+ e^{-I(\{el\},\{x\}/(k*c)}
		"""
		scale = self.fitval*self.cfit
		for y in self.P:
			self.F[y]+= np.exp(-self.cur_indic[(el,y)]/(scale))+1


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

		P_ = set()
		for i in range(len(self.P)): #We want to have the same size as the original population FAUX
			a,b =random.sample(self.P, 2)

			P_.add(max(a,b, key= (lambda x : self.F[x]))) #argmax{F[x], F[y]}
		return P_ 


	

	def variation(self, P_,mut_rate=0.01,mu=25):
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
					pMut[j] += +sigma_L*(ind[j]-Lowers[j])
				else:
					sigma_R = 1 - math.pow(2*(1-u),1/(mu+1))
					pMut[j] += sigma_R*(Uppers[j]-ind[j])
			self.P.add(tuple(pMut)) #np.append(P_,pMut, axis = 0)
	
	def most_vect(self,P,f):
		if len(P)!=0:
			most  = list(P.pop())
			P.add(tuple(most))
			for x in P:
				for i in range(len(most)):
					most[i]= f(x[i], most[i])
		return tuple(most)
	'''def recombination(P_,recom_rate=1.0,mu=25):
    """The recombination operator takes a certain number of parents and creates a 
    predefined number of children by combining parts of the parents. To mimic the 
    stochastic nature of evolution, a crossover probability is associated with this
    operator.
    
    Params:
        P_ ---> mating pool
        recom_rate ---> recombination rate by default is 1.0
    """
    recom_Pop = np.zeros((0,P_.shape[1]))
    size_recom = P_.shape[0]*recom_rate
    sample = np.random.randint(P_.shape[0],size=int(size_recom)) #Permutation
    for i in range(0,sample.shape[0]-1,2):
        #Pick two individuals to recombine to generate offspring
        parent0 = P_[sample[i],] 
        parent1 = P_[sample[i+1],]
        #Step 1: Choose a random number u 2 [0; 1).
        u = np.random.uniform(low=0.0, high=1.0)
        #Step 2: Calculate  beta(q) using equation
        if u <= 0.5:
            beta_q = np.power(2*u,1/(mu+1))
        else : 
            beta_q = np.power(1/(2*(1-u)),1/(mu+1))
    
        child0 = np.zeros((1,parent0.shape[0]))
        child1 = np.zeros((1,parent0.shape[0]))
        #Step 3: Compute children solutions using equations for all variables
        for j in range(parent0.shape[0]):#SBX algo to recombine cho
            child0[0][j] = 0.5*((1+beta_q)*parent0[j]+(1-beta_q)*parent1[j])
            child1[0][j] = 0.5*((1-beta_q)*parent0[j]+(1+beta_q)*parent1[j])
        recom_Pop = np.append(recom_Pop, child0, axis = 0)          
        recom_Pop = np.append(recom_Pop, child1, axis = 0)
    P_ = np.append(P_,recom_Pop, axis = 0)     
    return P_  
	'''


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
			self.variation(mat)
			self.cur_gen+=1





def I_epsilon(A, B):#gerer les positifs négatifs
	return np.max(A-B)

import cProfile, pstats, io
from pstats import SortKey

def myIBEA(fun, pop_size, num_max_gen, fit_scale_fact):
	ibea = IBEA(pop_size, fun.dimension, num_max_gen, fit_scale_fact, fun, I_epsilon)
	#pr = cProfile.Profile()
	#pr.enable()
	ibea.run()
	#pr.disable()
	#sortby = SortKey.CUMULATIVE
	#ps = pstats.Stats(pr).sort_stats(sortby)
	#ps.print_stats()
