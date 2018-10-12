#Mating selection
def binary_tour_sel(P):
    """Mating selection aims at picking promising solutions for variation and
    usually is performed in a randomized fashion. Perform binary tournament 
    selection with replacement on P in order to fill the temporary mating pool P_
    
    Params:
        P ---> pool of the population
    """
    for i in range(P.shape[0]) #We want to have the same size as the original population
        a = np.random.randint(P.shape[0],size=1) #random pick one individual a
        b = np.random.randint(P.shape[0],size=1) #random pick one individual b
        #TODO 
        P[a],P[b] #Which one fitness value is better
        if i == 0 : 
            P_ = P['better']
        else:
            P_ = np.vstack([P_,P['better']]) #Every loop put the better one out of two in to mating pool
    return P_ # return mating pool which has the same size as P 

#Variation
def mutation(P_,mut_rate):
    """the mutation operator modifies individuals by changing small 
    parts in the associated vectors according to a given mutation rate.
    
    Params:
        P_ ---> mating pool
        mut_rate ---> mutation rate
    """
    size_mutation = P_*mut_rate
    sample = np.random.randint(P_.shape[0],size=int(size_mutation))
    for i in range(sample):
        #TODO
        P_[i] #Mutation happens to this individual
    return P_ #put newly mutated individuals into mating pool and return
def recombination(P_,recom_rate):
    """The recombination operator takes a certain number of parents and creates a 
    predefined number of children by combining parts of the parents. To mimic the 
    stochastic nature of evolution, a crossover probability is associated with this
    operator.
    
    Params:
        P_ ---> mating pool
        recom_rate ---> recombination rate
    """
    size_recom = P_*recom_rate
    sample = np.random.randint(P_.shape[0],size=int(size_recom))
    for i in range(sample):
        #TODO
        P_[i] #Two individuals recombine to generate offspring
    return P_  #put newly generated individuals into mating pool and return