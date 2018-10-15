'''
initial population generation
'''

def generateInitialPopulation(popSize,popDim,upperBound,lowerBound):
    """
    - generate an initial random population of popSize numpy arrays of popDim dimensions which
    coordinates belong to [lowerBound,upperBound]

    - population will be returned as a 2D numpy array, a line of which being the coordinates
    of one population member
    """
    popRange = upperBound-lowerBound
    pop2DArray = np.random.rand(popSize,popDim)
    pop2DArray *= popRange
    pop2DArray -= round(popRange/2) # to work on zero centered interval - COCO works on [-5;5] ?
    return(pop2DArray)

'''
environmental selection
'''

def envirSelection(pop,matingPoolPop,fitValuePop,fitValueMatingPoolPop):
    """
    - environmental selection determines which individuals of the population and the modified
    mating pool form the new population. The simplest way is to use the latter set as the
    new population. An alternative is to combine both sets and deterministically choose
    the best individuals for survival

    - on a 2*pop.size on doit revenir à pop.size en conservant les meilleurs candidats

    - on suppose le travail en numpy array: 1 array pour les coordonnées dans le decision space,
    1 array pour les fitness values, 1 array pour les objective functions evalutions, l'indice permettant de lier
    les éléments de l'un avec les éléments des autres

    - pour ne pas confondre les indexes on incrémente les indexes associés à la mating pool de popSize
    """
    wantedPopSize = pop.shape[0]

    # whole population fit value indexes numpy array initialization
    indexArrayInitialPop = np.empty((wantedPopSize,))
    indexArrayMatingPop = np.empty((wantedPopSize,))
    for indexArrayIndex in range(wantedPopSize):
        indexArrayInitialPop[indexArrayIndex] = indexArrayIndex
        indexArrayMatingPop[indexArrayIndex] = indexArrayIndex + wantedPopSize

    # we add a dimension to put the indexes associated with the decision vectors in each population
    # NB: indexes associated with mating pool decision vectors will start at wantedPopSize
    fitValuePop_temp = np.empty((fitValuePop.shape[0],2))
    fitValueMatingPoolPop_temp = np.empty((fitValueMatingPoolPop.shape[0],2))

    # we copy the fit values in each temp vector
    # and the associated index (describing the position in the initial numpy arrays of fitness values) in
    # the second column. NB: if index belongs to mating pool value is (index + wantedPopSize)
    fitValuePop_temp[:,0] = fitValuePop[:,0]
    fitValuePop_temp[:,1] = indexArrayInitialPop[:,0]
    fitValueMatingPoolPop_temp[:,0] = fitValueMatingPoolPop[:,0]
    fitValueMatingPoolPop_temp[:,1] = indexArrayMatingPop[:,0]

    fitValueMatingAndInitial = np.concatenate((fitValuePop_temp,fitValueMatingPoolPop_temp),axis=0)

    # retrait des x* jusqu'à atteindre la taille de population initiale
    while fitValueMatingAndInitial.shape[0] > wantedPopSize:
        # regarding code line below: axis = 0 to sort according to fitness value and not index value
        fitValueMatingAndInitial = np.ndarray.sort(axis=0, kind='quicksort')

        # dans le adaptive IBEA il faut enlever le candidat du decision space avec le plus petit fitness
        # value avant de sélectionner
        fitValueMatingAndInitial = fitValueMatingAndInitial[1:,:,:]

        # we update the fitness values for the remaining population

    # renvoie des decision vectors en numpy array de taille (popSize,nbrDimDecisionSpace)
    return(newPop)

'''
The crossover probability of 1.0 and mutation probability of 1/n are used
The recombination and mutation probabilities were set to 1.0 and to 0.01 resp.
NSGA-II nd SPEA2 with a population size of 100 is run
for 300 generations using a real-parameter SBX crossover operator (ηc = 15) 
and a variablewise polynomial mutation operator (ηm = 20). 
'''

#Mating selection
def binary_tour_sel(P, fit_values):
    """Mating selection aims at picking promising solutions for variation and
    usually is performed in a randomized fashion. Perform binary tournament 
    selection with replacement on P in order to fill the temporary mating pool P_
    
    Params:
        P ---> pool of the population
    """
    P_ = np.zeros((0,P.shape[1]))
    for i in range(P.shape[0]): #We want to have the same size as the original population
        a = np.random.randint(P.shape[0],size=1) #random pick one individual a
        b = np.random.randint(P.shape[0],size=1) #random pick one individual b
        #Which one fitness value is better
        #if fitness(P[a,])>fitness(P[b,]) :
        if fit_values[a] > fit_values[b] :
            better_one = P[a,][0].reshape(1,P.shape[1])
        else:
            better_one = P[b,][0].reshape(1,P.shape[1])
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
def mutation(P_,mut_rate=0.01,mu=25):
    """the mutation operator modifies individuals by changing small 
    parts in the associated vectors according to a given mutation rate.
    
    Params:
        P_ ---> mating pool
        mut_rate ---> mutation rate by default is 1.0
        mu ---> 25
    """
    size_mutation = P_.shape[0]*mut_rate
    Uppers = P_.max(axis=0)
    Lowers = P_.min(axis=0)
    sample = np.random.randint(0,P_.shape[0],size=int(size_mutation))
    for i in sample:
        indiv_to_mut = P_[i,] #Mutation happens to this individual
        p_mut = np.zeros((1,P_.shape[1]))
        for j in range(indiv_to_mut.shape[0]):
            #Step 1: Choose a random number u in [0; 1).
            u = np.random.uniform(low=0.0, high=1.0)
            #Step 2: Calculate the two parameters (δL or δR)
            #Step 3: the mutated solution p_mut for a particular variable j is created
            if u <= 0.5:
                sigma_L =  np.power(2*u,1/(mu+1))-1
                p_mut[0][j] = indiv_to_mut[j]+sigma_L*(indiv_to_mut[j]-Lowers[j])
            else:
                sigma_R = 1 - np.power(2*(1-u),1/(mu+1))
                p_mut[0][j] = indiv_to_mut[j]+sigma_R*(Uppers[j]-indiv_to_mut[j])
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
def recombination(P_,recom_rate=1.0,mu=25):
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

import numpy as np
# #Generate a random population with size 100 and each one has 20 chromosomes
# a = np.random.random((100,20))
# P_ = binary_tour_sel(a)
# print(P_.shape)
# P_ = recombination(P_)
# print(P_.shape)
# P_ = mutation(P_)
# print(P_.shape)
