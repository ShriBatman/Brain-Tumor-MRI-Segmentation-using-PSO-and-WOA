import numpy as np
import math
import sys

ground = [[20,40],[60,80],[100,120],[140,170]]

def fitness(sset, noclus, dpoints, tr=1,):
    clussize = np.zeros(noclus, dtype=int)
    cluselem = np.zeros((noclus,dpoints.shape[0]), dtype=int)

    for i in range(dpoints.shape[0]):
        minval = sys.maxint
        clusindex = -1
        for j in range(noclus):
            val = abs(dpoints[i] - sset[j])
            if val < minval :
                minval = val
                clusindex = j
        cluselem[clusindex][clussize[clusindex]] = i
        clussize[clusindex] += 1;

    fitness = 0 

    for i in range(noclus):
        cluscenter = sset[i]
        val = 0.0
        for j in range(clussize[i]):
            dpoint = dpoints[cluselem[i][j]]
            val += abs(dpoint - cluscenter) 
        if clussize[i] > 0:
            val /= clussize[i]
        fitness += val
    
    if tr :
        return fitness
    else :
        return fitness, cluselem, clussize

def woa(noclus ,dpoints):
    
    ''' 
        noclus = no of clusters.
    '''
    randomcount=0
    max_iterations = 30
    noposs = 10 # no. of possible solutions
    poss_sols = np.zeros((noposs, noclus)) # whale positions
    gbest = np.zeros((noclus,)) # globally best whale postitions
    b = 2.0

    for i in range(noposs):
        for j in range(noclus):
            poss_sols[i][j] = np.random.randint(ground[j][0],ground[j][1])

    global_fitness = sys.maxint
    
    for i in range(noposs):
        cur_par_fitness = fitness(poss_sols[i], noclus, dpoints)
        if cur_par_fitness < global_fitness:
            global_fitness = cur_par_fitness
            gbest = poss_sols[i]

    print "initial gfitness=",global_fitness
    for it in range(max_iterations):
        for i in range(noposs):
            a = 2.0 - (2.0*it)/(1.0 * max_iterations)
            r = np.random.random_sample()
            A = 2.0*a*r - a
            C = 2.0*r
            l = 2.0 * np.random.random_sample() - 1.0
            p = np.random.random_sample()
            for j in range(noclus):
                lb = 0
                ub = 256

                x = poss_sols[i][j]
                if p < 0.5:
                    if abs(A) < 1:
                        _x = gbest[j]
                    else :
                        rand = np.random.randint(noposs)
                        _x = poss_sols[rand][j]

                    D = abs(C*_x - x)
                    updatedx = _x - A*D
                else :
                    _x = gbest[j]
                    D = abs(_x - x)
                    updatedx = D * math.exp(b*l) * math.cos(2.0* math.acos(-1.0) * l) + _x

                if updatedx < lb or updatedx > ub:
                    updatedx = np.random.randint(lb, high = ub+1)
                    randomcount += 1

                poss_sols[i][j] = updatedx

            fitnessi = fitness(poss_sols[i], noclus, dpoints)
            if fitnessi < global_fitness :
                global_fitness = fitnessi
                gbest = poss_sols[i]

        print "iteration",it,"=",global_fitness
                
    print "random count =",randomcount
    fitnessi, cluselem, clussize = fitness(gbest, noclus, dpoints, tr=0)
    return gbest, cluselem, clussize
