'''
The crossover probability of 1.0 and mutation probability of 1/n are used
The recombination and mutation probabilities were set to 1.0 and to 0.01 resp.
NSGA-II nd SPEA2 with a population size of 100 is run
for 300 generations using a real-parameter SBX crossover operator (ηc = 15) 
and a variablewise polynomial mutation operator (ηm = 20). 
'''

#Mating selection
def binary_tour_sel(P):
    """Mating selection aims at picking promising solutions for variation and
    usually is performed in a randomized fashion. Perform binary tournament 
    selection with replacement on P in order to fill the temporary mating pool P_
    
    Params:
        P ---> pool of the population
    """
    P_ = np.zeros((0,len(P.shape[0])))
    for i in range(P.shape[0]): #We want to have the same size as the original population
        a = np.random.randint(P.shape[0],size=1) #random pick one individual a
        b = np.random.randint(P.shape[0],size=1) #random pick one individual b
        #TODO 
        #Which one fitness value is better
        if fitness(P[a,])>fitness(P[b,]) : 
            better_one = P[a,]
        else:
        	better_one = P[b,]
            #Every loop put the better one out of two into mating pool
        P_ = np.append(P_,better_one, axis = 0)  
    return P_ # return mating pool which has the same size as P 

#Variation
'''
Deb and Agrawal suggested a polynomial mutation
operator with a user-defined index parameter (ηm).
Based on a theoretical study, they concluded that ηm
induces an effect of a perturbation of O((b−a)/ηm) in
a variable, where a and b are lower and upper bounds
of the variable. They also found that a value ηm ∈
[20, 100] is adequate in most problems that they tried.
In a mutation operator, the probability of a mutated string close to the
par ent string is higher, bu t this probability is usually constant and depend s
only on one string.
https://www.iitk.ac.in/kangal/papers/k2012016.pdf
Polynomial mutation
Gaussien mutation
'''
def mutation(P_,mut_rate=0.01):
    """the mutation operator modifies individuals by changing small 
    parts in the associated vectors according to a given mutation rate.
    
    Params:
        P_ ---> mating pool
        mut_rate ---> mutation rate by default is 1.0
    """
    size_mutation = P_.shape[0]*mut_rate
    Uppers = P_.max(axis=0)
    Lowers = P_.min(axis=0)
    sample = np.random.randint(P_.shape[0],size=int(size_mutation))

    for i in range(sample):
    	indiv_to_mut = P_[i,] #Mutation happens to this individual
    	for j in range(indiv_to_mut.shape[0]):
        	#Step 1: Choose a random number u in [0; 1).
        	u = np.random.uniform(low=0.0, high=1.0)
        	#Step 2: Calculate the two parameters (δL or δR)
        	#Step 3: the mutated solution p_mut for a particular variable j is created
        	if u <= 0.5:
        		sigma_L =  np.power(2*u,1/(mu+1))-1
        		p_mut = indiv_to_mut[j]+sigma_L(indiv_to_mut[j]-Lowers[j])
        	else:
        		sigma_R = 1 - np.power(2*(1-u),1/(mu+1))
        		p_mut = indiv_to_mut[j]+sigma_R(Uppers[j]-indiv_to_mut[j])
        	#put newly mutated individuals into mating pool and return 
        	P_ = np.append(P_,p_mut, axis = 0)
    return P_ 

'''
The crossover operator combines the genes of two or more parents to generate better
offspring. It is based on the idea that the exchange of information between good chromosomes
will generate even better offspring with the distribution index (ηc)
A large value of η gives a higher probability for creating near
parent solutions and a small value of η allows distant solutions to be selected as children solution.
SBX operator http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.85.7460&rep=rep1&type=pdf
SBX_Adap operator http://www.cs.bham.ac.uk/~wbl/biblio/gecco2007/docs/p1187.pdf
'''
def recombination(P_,recom_rate=1.0):
    """The recombination operator takes a certain number of parents and creates a 
    predefined number of children by combining parts of the parents. To mimic the 
    stochastic nature of evolution, a crossover probability is associated with this
    operator.
    
    Params:
        P_ ---> mating pool
        recom_rate ---> recombination rate by default is 1.0
    """
    recom_Pop = np.zeros((0,len(P_.shape[0])))
    size_recom = P_.shape[0]*recom_rate
    sample = np.random.randint(P_.shape[0],size=int(size_recom)) #Permutation
    for i in range(0,sample.shape[0],2):
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

    	child0 = np.zeros(len(parents0.shape[0]));
        child1 = np.zeros(len(parents0.shape[0]));
    	#Step 3: Compute children solutions using equations for all variables
    	for j in range(parent0.shape[0]):#SBX algo to recombine cho
			child0[j] = 0.5((1+beta_q)*parent0[j]+(1-beta_q)*parent1[j])
			child1[j] = 0.5((1-beta_q)*parent0[j]+(1+beta_q)*parent1[j])
		recom_Pop = np.append(recom_Pop, child0, axis = 0)          
		recom_Pop = np.append(recom_Pop, child1, axis = 0)
	#put newly generated individuals into mating pool and return
	P_ = np.append(P_,recom_Pop, axis = 0)     
    return P_  
    