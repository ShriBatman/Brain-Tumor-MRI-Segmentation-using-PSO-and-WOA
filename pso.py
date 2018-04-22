import numpy as np
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

    fitness = 0.0

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

def pso(noclus, dpoints):
    
    ''' 
        noclus = no of clusters.
    '''
    randomcount = 0
    max_iterations = 30
    noposs = 5 # no. of possible solutions
    poss_sols = np.zeros((noposs, noclus),) # particles position
    pbest = np.zeros((noposs, noclus),) # each particle's best position
    pfit = np.zeros(noposs) # each particle's best fitness value
    gbest = np.zeros((noclus),) # globally best particle postition
    parvel = np.zeros((noposs, noclus)) # particle velocity
    c2 = 1.7 # social constant
    c1 = 1.7 # cognitive constant
    w = .5 # inertia  
    global_fitness = sys.maxint

    for i in range(noposs):
        pfit[i] = sys.maxint
        
    for i in range(noposs):
        for j in range(noclus):
                poss_sols[i][j] = np.random.randint(ground[j][0],ground[j][1])
                parvel[i][j] = np.random.randint(ground[j][0],ground[j][1])

    for it in range(max_iterations):
        for i in range(noposs):
            cur_par_fitness = fitness(poss_sols[i], noclus, dpoints)
            best_fitness = pfit[i]
            if cur_par_fitness < best_fitness:
                pfit[i] = cur_par_fitness
                pbest[i] = poss_sols[i]
     
        for i in range(noposs):
            if pfit[i] < global_fitness:
                global_fitness = pfit[i]
                gbest = poss_sols[i]

        for i in range(noposs):
            for j in range(noclus):
            
                r1 = np.random.random_sample()
                r2 = np.random.random_sample()

                lb = 0
                ub = 255
                
                inertial_vel = w * parvel[i][j] # inertia weight
                cog_vel = r1 * c1 * (pbest[i][j] - poss_sols[i][j]) # cognitive factor
                soc_vel = r2 * c2 * (gbest[j] - poss_sols[i][j]) # social factor

                vel = inertial_vel + cog_vel + soc_vel #update in vel

                # if vel < lb or vel > ub:
                #     vel = np.random.randint(lb,high = ub+1)
                #     randomcount += 1

                parvel[i][j] = vel
                position = poss_sols[i][j] + vel 

                if position < lb or position > ub :
                    position = np.random.randint(lb,high = ub+1)
                    randomcount += 1
                
                poss_sols[i][j] = position #update in position

        print "iteration",it,"=",global_fitness


    print "random count=",randomcount
    
    
    fitnessi, cluselem, clussize = fitness(gbest, noclus, dpoints, tr=0)
    return gbest, cluselem, clussize
        
