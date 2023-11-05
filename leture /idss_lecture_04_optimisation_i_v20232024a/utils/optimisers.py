from utils.history import History
import numpy as np
import itertools


def genetic_search(L, pop, guess_fn, mutation_fn, iters, keep=0.25):
    o = History()
    # create the initial population randomly
    population = np.array([guess_fn() for i in range(pop)])
    d = len(guess_fn()) # store dimensionality of problem
    loss = np.zeros(pop)
    
    for i in range(iters):                
        for j in range(pop):                        
            # could also mutate *everyone* here
            # this works better in asexual reproduction            
            loss[j] = L(population[j])
            
        # order by loss
        order = np.argsort(loss)
        loss = loss[order]
        population = population[order]
        
        # replicate top "keep" fraction of individuals
        top = int(pop * keep)
        for j in range(top, pop):
            # sexual reproduction
            mum = np.random.randint(0, top)
            dad = np.random.randint(0, top)
            chromosones = np.random.randint(0,2,d)  
            
            # select elements from each dimension randomly from mum and dad
            population[j]  = mutation_fn(np.where(chromosones==0, population[mum], population[dad]))             
            
        # track the best individual so far
        o.track(population[0], loss[0])            
    return o.finalise()


def grid_search(L, ranges, divs, maxiter=None):
    """L: loss function
    ranges: Parameter ranges for each dimension (e.g. [[0,1], [-1,1], [0,2]])
    divs: division per range
    """    
    o = History()
    divisions = [np.linspace(r[0], r[1], divs) for r in ranges]        
    i  = 0
    for theta in itertools.product(*divisions):                            
        o.track(theta, L(np.array(theta)))
        
        i+=1
        if maxiter and i>=maxiter:
            break
    return o.finalise()
    
def hill_climbing(L, guess_fn, neighbour_fn, iters):
    """
    L: loss function
    theta_0: initial guess
    neighbour_fn(theta): given a parameter vector, returns a random vector nearby
    iters: number of iterations to run the optimisation for
    """
    o = History()
    theta_0 = guess_fn()
    o.track(theta_0, L(theta_0))
    for i in range(iters):
        proposal = neighbour_fn(o.best_theta)
        o.track(proposal, L(proposal))        
    return o.finalise()
    
    
def random_search(L, sample_fn, iters):
    """L: loss function
    sample_fn: calling this should draw one random sample from the parameter space
    iters: number of iterations to run the optimisation for
    """
    o = History()
    for i in range(iters):
        theta = sample_fn()        
        o.track(theta, L(theta))    
    return o.finalise()    
    

def simulated_anneal(L, guess_fn, neighbour_fn, temperature_fn, iters):
    """
    L: loss function
    theta_0: initial guess
    neighbour_fn(theta): given a parameter vector, 
                         returns a random vector nearby
    temperature_fn(iter): given an iteration,     
                        return the temperature schedule
    iters: number of iterations to run the optimisation for
    """
    o = History()
    theta_0 = guess_fn()
    o.track(theta_0, L(theta_0))
    state = theta_0.copy()
    loss = L(theta_0)
    for i in range(iters):
        proposal = neighbour_fn(state)        
        proposal_loss = L(proposal)                
        # climb if we can
        if proposal_loss<loss:
            o.track(proposal, proposal_loss, force=True)   
            loss, state = proposal_loss, proposal
            
        else:                                   
            # check how bad this jump might be
            p = np.exp(-(proposal_loss-loss)) * temperature_fn(i)    
                                            
            # randomly accept the jump with the given probability
            if np.random.uniform(0,1)<p:                
                o.track(proposal, proposal_loss, force=True)                                    
                loss, state = proposal_loss, proposal
            else:
                o.track(proposal, proposal_loss)
    return o.finalise()
    
def gradient_descent(L, dL, theta_0, delta, tol=1e-4, maxiter=None):
    theta = np.array(theta_0) # copy theta_0    
    o = History()    
    i  = 0
    # while the loss changes
    while np.abs(o.loss_change)>tol:
        # step along the derivative        
        i+=1
        theta += -delta * dL(theta)        
        o.track(np.array(theta), L(theta))   
        if maxiter and i>=maxiter:
            break        
        
    return o.finalise()    