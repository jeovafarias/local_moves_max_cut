import numpy as np 

# NOTE: may not be fast - need to improve by using edge indices after
# finding the subproblem

#---------------------- General Utilities ------------------------#

# Clustering energy
def energy(V,E,w,labels):
    """
    Calculates the weight of hyperedges cut by a clustering
    Args:
        V (list of int): list of vertices
        E (list of list of int): list of hyperedges (subsets of V)
        w (list of float): corresponding weights of edges in E
        labels (dict of int: int): clustering labels for each v in V
    Returns:
        energy (float): sum of weight of edges cut by clustering labels
    """
    energy = 0.
    for i in range(len(E)):
        allSame = True
        label0 = labels[E[i][0]]
        for v in E[i][1:]:
            if labels[v] != label0:
                allSame = False
                break

        if allSame == False:
            energy += w[i]

    return energy

# Generalized submodular function that captures a-b swaps and a-exps
def submodEval(x,E0,E1,w0,w1):
    """
    Evaluates the general submodular function
    Args:
        x (dict of int: float): probability each v in V is set to 1
        E0 (list of list of int): list of hyperedges that are not cut iff all v in e are 0 or 1
        E1 (list of list of int): list of hyperedges that are not cut iff all v in e are 1
        w0 (list of float): corresponding weights of edges in E0
        w1 (list of float): corresponding weights of edges in E1
    Returns:
        sumCut (float): sum of weight of edges cut by x in expectation
    """
    sumCut = 0.

    for i in range(len(E0)):
        probAllZero = 1.0
        probAllOne = 1.0
        for v in E0[i]:
            probAllZero = probAllZero*(1-x[v])
            probAllOne = probAllOne*(x[v])
        sumCut += w0[i]*(1-probAllZero-probAllOne)

    for i in range(len(E1)):
        probAllOne = 1.0
        for v in E1[i]:
            probAllOne = probAllOne*(x[v])
        sumCut += w1[i]*(1-probAllOne)
        
    return sumCut

#--------------------- Alpha Exp --------------------------------#

# Create the general submodular problem for an alpha-expansion
def AExpSubproblem(VOrig,EOrig,wOrig,labels,alpha):
    """
    Creates the submodular problem corresponding to an alpha-expansion
    Args:
        VOrig (list of int): list of vertices
        EOrig (list of list of int): list of hyperedges (subsets of VOrig)
        wOrig (list of float): corresponding weights of edges in EOrig
        labels (dict of int: int): clustering labels for each v in VOrig
        alpha (int): cluster chosen for an alpha-expansion
    Returns:
        E0 (list of list of int): list of hyperedges that are not cut by labels and not between alphas
        w0 (list of float): corresponding weights of edges in E0
        E1 (list of list of int): list of hyperedges that are cut by labels
        w1 (list of float): corresponding weights of edges in E1
    """
    E0, w0, E1, w1 = [], [], [], []
    for i in range(len(EOrig)):
        e = EOrig[i]
        label0 = labels[e[0]]
        allSame = True
        for v in e[1:]:
            if labels[v] != label0:
                allSame = False
                break
        if allSame and label0 != alpha:
            E0.append(e)
            w0.append(wOrig[i])
        elif allSame == False:
            E1.append(e)
            w1.append(wOrig[i])
                       
    return E0, w0, E1, w1

# Update labels given submodular problem output for alpha-expansion
def AExpUpdate(labels,x,alpha):
    """
    Update labels given submodular problem output for alpha-expansion
    Args:
        labels (dict of int: int): clustering labels for each v in V
        x (dict of int: float): probability each v in V is set to 1 in the submod prob
        alpha (int): cluster chosen for an alpha-expansion
    Returns:
       newLabels (dict of int: int): updated clustering labels for each v in V
    """
    newLabels = dict()
    for v in labels.keys():
        if x[v] == 1:
            newLabels[v] = alpha
        else:
            newLabels[v] = labels[v]
    return newLabels

#--------------------- AB Swaps --------------------------------#

# Create the general submodular problem for an alpha-beta swap
def ABSubproblem(VOrig,EOrig,wOrig,labels,alpha,beta):
    """
    Creates the submodular problem corresponding to an alpha-beta swap 
    Args:
        VOrig (list of int): list of vertices
        EOrig (list of list of int): list of hyperedges (subsets of VOrig)
        wOrig (list of float): corresponding weights of edges in EOrig
        labels (dict of int: int): clustering labels for each v in VOrig
        alpha (int): first cluster chosen for the alpha-beta swap
        beta (int): second cluster chosen for the alpha-beta swap
    Returns:
        V (list of int): list of vertices labeled alpha or beta
        E0 (list of list of int): list of hyperedges that only involve V
        w0 (list of float): corresponding weights of edges in E
    """
    V = []
    for v in VOrig:
        if (labels[v] == alpha) or (labels[v] == beta):
            V.append(v)
    
    E0, w0 = [], []
    for i in range(len(EOrig)):
        e = EOrig[i]
        keepE = True
        for v in e:
            if (labels[v] != alpha) and (labels[v] != beta):
                keepE = False
                break
        if keepE:
            E0.append(e)
            w0.append(wOrig[i])

    return V, E0, w0

# Update labels given submodular problem output for alpha-beta swap
def ABUpdate(labels,x,alpha,beta):
    """
    Update labels given submodular problem output for alpha-beta swap
    Args:
        labels (dict of int: int): clustering labels for each v in V
        x (dict of int: float): probability each v in V is set to 1 in the submod prob
        alpha (int): first cluster chosen for an alpha-expansion
        beta (int): second cluster chosen for an alpha-beta swap
    Returns:
       newLabels (dict of int: int): updated clustering labels for each v in V
    """
    newLabels = dict()
    for v in labels.keys():
        if labels[v] != alpha and labels[v] != beta:
            newLabels[v] = labels[v]
        elif x[v] == 1:
            newLabels[v] = alpha
        else:
            newLabels[v] = beta
    return newLabels

#--------------------- Submod Alg --------------------------------#

# Find the best alpha-beta swap or alpha-expansion by solving the corresponding submodular
# problem using the Buchbinder et al algorithm
def submodAlg(VOrig,EOrig,wOrig,labels,moveType,alpha,beta=None):
    """
    Find the best alpha-beta swap or alpha-expansion by solving the corresponding submodular
    problem using the Buchbinder et al algorithm
    Args:
        VOrig (list of int): list of vertices
        EOrig (list of list of int): list of hyperedges (subsets of VOrig)
        wOrig (list of float): corresponding weights of edges in EOrig
        labels (dict of int: int): clustering labels for each v in VOrig
        alpha (int): first cluster chosen for the alpha-beta swap
        beta (int): second cluster chosen for the alpha-beta swap
        moveType (str): 'ab' indicates alpha-beta swap, 'aexp' indicates alpha-expansion
    Returns:
        newLabels (dict of int: int): updated clustering labels for each v in VOrig
    """
    if moveType == 'aexp':
        V = VOrig
        E0,w0,E1,w1 = AExpSubproblem(VOrig,EOrig,wOrig,labels,alpha)
    else:
        V,E0,w0 = ABSubproblem(VOrig,EOrig,wOrig,labels,alpha,beta)
        E1, w1 = [], []

    x, y = dict(), dict()
    for v in V:
        if moveType == 'aexp' and labels[v] == alpha:
            x[v] = 1
        else:
            x[v] = 0
        y[v] = 1

    # Get probabilities
    for v in V:
        # If v is already set, skip this iteration
        if x[v] == 1:
            continue
        # Calc current values
        cutX = submodEval(x,E0,E1,w0,w1)
        cutY = submodEval(y,E0,E1,w0,w1)

        # Calc difference of changing x or y's value for v
        x[v], y[v] = 1, 0
        newCutX = submodEval(x,E0,E1,w0,w1)
        newCutY = submodEval(y,E0,E1,w0,w1)
        
        tv = newCutX - cutX
        fv = newCutY - cutY

        # Update value for v
        if tv <= 0:
            pv = 0
        elif fv <= 0:
            pv = 1
        else:
            pv = tv/(tv+fv)
        x[v], y[v] = pv, pv

    # Derandomize x
    for v in V:
        if x[v] == 1:
            continue
        
        x[v] = 1
        cut1 = submodEval(x,E0,E1,w0,w1)
        x[v] = 0
        cut0 = submodEval(x,E0,E1,w0,w1)

        if cut1 >= cut0:
            x[v] = 1

    # Get new labels
    if moveType == 'aexp':
        newLabels = AExpUpdate(labels,x,alpha)
    else:
        newLabels = ABUpdate(labels,x,alpha,beta)

    return newLabels

#--------------------- Local Search --------------------------------#

# Run local search using alpha-beta swaps of alpha-expansions
def localSearchSubmod(V,E,w,k,origLabels,moveType = 'ab',moveSequence=None,maxIt=1000):
    """
    Approximately solve the Max-K-Hypercut problem represented by V,E,w using a submodular maximization

    Args:
        V (list of int): list of vertices
        E (list of list of int): list of hyperedges (subsets of V)
        w (list of float): corresponding weights of edges in E
        k (int): number of clusters
        origLabels (dict of int: int): clustering labels for each v in V
        moveType (str): 'ab' indicates alpha-beta swap, 'aexp' indicates alpha-expansion
        maxIt (int): maximum number of iterations run
    Returns:
        labels (dict of int: int): updated clustering labels for each v in V
    """
    assert moveType in ("ab", "aexp", "aebs", "aexp-ran", "aexp-2b")

    if moveSequence == None:
        if moveType == 'ab':
            moveSequence = [[i,j] for i in range(k) for j in range(i+1,k)]
        else:
            moveSequence = range(k)

    labels = origLabels.copy()
    it, maxEne = 0, energy(V,E,w,labels)
    updated = True
    while updated and it <= maxIt:
        updated = False
        for move in moveSequence:
            if moveType == 'ab':
                alpha,beta = move[0], move[1]
                #print("Swapping (%d, %d), Current Class.: %s" % (alpha, beta, labels))
                newLabels = submodAlg(V,E,w,labels,moveType,alpha,beta)
            elif moveType == 'aexp':
                alpha = move
                #print("Expanding (%d), Current Class.: %s" % (alpha, labels))
                newLabels = submodAlg(V,E,w,labels,moveType,alpha)
            elif moveType == 'aebs':
                alpha,beta = move[0],move[1]
                labelsTemp = labels.copy()
                for v in V:
                    if labelsTemp[v] == alpha:
                        labelsTemp[v] = beta
                newLabels = submodAlg(V,E,w,labelsTemp,'aexp',alpha)
            elif moveType == 'aexp-ran':
                alpha = move
                labelsTemp = labels.copy()
                # Relabel those with label alpha randomly before alpha exp
                for v in V:
                    while (labelsTemp[v] == alpha):
                        labelsTemp[v] = np.random.randint(0,k)
                newLabels = submodAlg(V,E,w,labelsTemp,'aexp',alpha)
            elif moveType == 'aexp-2b':
                alpha = move
                labelsTemp = labels.copy()
                # Relabel those with label alpha to other best option before alpha-exp
                for v in V:
                    if labelsTemp[v] == alpha:
                        best_label, best_ene = alpha, 0
                        for i in range(k):
                            if i == alpha:
                                continue
                            labelsTemp[v] = i
                            ene = energy(V,E,w,labelsTemp)
                            if ene < best_ene:
                                labelsTemp[v] = best_label
                            else:
                                best_label, best_ene = i, ene
                newLabels = submodAlg(V,E,w,labelsTemp,'aexp',alpha)
                    
            ene = energy(V,E,w,newLabels)
            if ene > maxEne:
                maxEne = ene
                labels = newLabels
                updated = True
        it += 1

    return labels, it


# Vanilla local search for clustering
# NOTE: Could be more efficient by indexing E
def localSearch(V,E,w,k,origLabels,maxIt=1000):
    """
    Simple local search algorithm which iteratively finds the best cluster
    for each point
    
    Args:
        V (list of int): list of vertices
        E (list of list of int): list of hyperedges (subsets of V)
        w (list of float): corresponding weights of edges in E
        k (int): number of clusters
        origLabels (dict of int: int): clustering labels for each v in V
        maxIt (int): maximum number of iterations run
    Returns:
        labels (dict of int: int): updated clustering labels for each v in V
    """
    
    labels = origLabels.copy()
    it, maxEne = 0, energy(V,E,w,labels)
    updated = True
    while updated and it <= maxIt:
        print(it)
        updated = False
        for v in V:
            #print("Updating (%d), Current Class.: %s" % (v, labels))
            for i in range(k):
                label0 = labels[v]
                labels[v] = i
                ene = energy(V,E,w,labels)
                if ene > maxEne:
                    updated = True
                    maxEne = ene
                else:
                    labels[v] = label0
        it += 1


    return labels, it
