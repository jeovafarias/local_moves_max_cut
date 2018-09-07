import submod_alg as sa
import unittest
import numpy as np

places = 6

#---------------------- General Utilities ------------------------#

class TestEnergy(unittest.TestCase):
    def test_empty(self):
        V = [0,1,2]
        E = []
        w = []

        labels = {1:0,2:0,0:0}
        self.assertEqual(sa.energy(V,E,w,labels),0.)
        
    def test_edges(self):
        V = [0,1,2]
        E = [[1,2],[0,1],[2,0]]
        w = [1.0, 2.0, 0.5]

        labels = {1:0,2:0,0:0}
        self.assertEqual(sa.energy(V,E,w,labels),0.)

        labels[1] = 1
        self.assertEqual(sa.energy(V,E,w,labels),3.)

        labels[2] = 2
        self.assertEqual(sa.energy(V,E,w,labels),3.5)
        
    def test_hyperedges(self):
        V = [0,1,2,3]
        E = [[0,1,2],[2,1,3]]
        w = [1.0, 2.0]

        labels = {1:0,2:0,0:0,3:0}
        self.assertEqual(sa.energy(V,E,w,labels),0.)

        labels[3] = 1
        self.assertEqual(sa.energy(V,E,w,labels),2.)

        labels[2] = 1
        self.assertEqual(sa.energy(V,E,w,labels),3.)

class TestSubmodEval(unittest.TestCase):
    def test_all0(self):
        E0 = [[0,1],[1,2],[2,1,0]]
        w0 = [1.0,2.0,0.5]
        E1 = []
        w1 = []

        x = {0:0,1:0,2:0}
        self.assertEqual(sa.submodEval(x,E0,E1,w0,w1),0.)

        x = {0:1.,1:1.,2:1.}
        self.assertEqual(sa.submodEval(x,E0,E1,w0,w1),0.)
        
        x = {0:0.9,1:0.3,2:0}
        self.assertAlmostEqual(sa.submodEval(x,E0,E1,w0,w1),1.725,places)

    def test_all1(self):
        E1 = [[0,1],[1,2],[2,1,0]]
        w1 = [1.0,2.0,0.5]
        E0 = []
        w0 = []

        x = {0:0,1:0,2:0}
        self.assertEqual(sa.submodEval(x,E0,E1,w0,w1),3.5)

        x = {0:1.,1:1.,2:1.}
        self.assertEqual(sa.submodEval(x,E0,E1,w0,w1),0.0)
        
        x = {0:0.9,1:0.3,2:0}
        self.assertAlmostEqual(sa.submodEval(x,E0,E1,w0,w1),3.23,places)

    def test_mix(self):
        E0 = [[0,1],[1,2],[2,1,0]]
        w0 = [1.0,2.0,0.5]
        E1 = [[0,1],[1,2],[2,1,0]]
        w1 = [1.0,2.0,0.5]
        x = {0:0.9,1:0.3,2:0}
        self.assertAlmostEqual(sa.submodEval(x,E0,E1,w0,w1),3.23+1.725,places)

#--------------------- Alpha Exp --------------------------------#

class TestAExpSubproblem(unittest.TestCase):
    def test_empty_alpha(self):
        VOrig = [1,2,3,4]
        EOrig = [[1,2],[3,4],[2,3,4]]
        wOrig = [1.0,2.0,3.0]
        labels = {1:1,2:1,3:2,4:2}
        alpha = 3

        E0,w0,E1,w1 = sa.AExpSubproblem(VOrig,EOrig,wOrig,labels,alpha)
        self.assertEqual(E0,[[1,2],[3,4]])
        self.assertEqual(w0,[1.0,2.0])
        self.assertEqual(E1,[[2,3,4]])
        self.assertEqual(w1,[3.0])

    def test_three_clusters(self):
        VOrig = [1,2,3,4,5,6]
        EOrig = [[1,2],[3,4],[2,3,4],[5,6],[5,1,4],[5,1,2]]
        wOrig = [1.0,2.0,3.0,4.0,5.0,6.0]
        labels = {1:1,2:1,3:2,4:2,5:3,6:3}
        alpha = 3

        E0,w0,E1,w1 = sa.AExpSubproblem(VOrig,EOrig,wOrig,labels,alpha)
        self.assertEqual(E0,[[1,2],[3,4]])
        self.assertEqual(w0,[1.0,2.0])
        self.assertEqual(E1,[[2,3,4],[5,1,4],[5,1,2]])
        self.assertEqual(w1,[3.0,5.0,6.0])

class TestAExpUpdate(unittest.TestCase):
    def test_simple(self):
        labels = {0:1,1:1,2:2,3:2,4:3}
        alpha = 3

        x = {0:0,1:0,2:0,3:0,4:0}
        newlabels = {0:1,1:1,2:2,3:2,4:3}
        self.assertEqual(sa.AExpUpdate(labels,x,alpha),newlabels)

        x[3] = 1.
        x[0] = 1.
        newlabels = {0:3,1:1,2:2,3:3,4:3}
        self.assertEqual(sa.AExpUpdate(labels,x,alpha),newlabels)

#--------------------- AB Swaps --------------------------------#

class TestABSubproblem(unittest.TestCase):
    def test_empty_alpha(self):
        VOrig = [1,2,3,4]
        EOrig = [[1,2],[3,4],[2,3,4]]
        wOrig = [1.0,2.0,3.0]
        labels = {1:1,2:1,3:2,4:2}
        alpha = 3
        beta = 1

        V,E0,w0 = sa.ABSubproblem(VOrig,EOrig,wOrig,labels,alpha,beta)
        self.assertEqual(V,[1,2])
        self.assertEqual(E0,[[1,2]])
        self.assertEqual(w0,[1.0])

    def test_larger_example(self):
        VOrig = [1,2,3,4,5,6]
        EOrig = [[1,2],[3,4],[2,3,4],[5,6],[5,1,4],[5,1,2]]
        wOrig = [1.0,2.0,3.0,4.0,5.0,6.0]
        labels = {1:1,2:1,3:2,4:2,5:3,6:3}
        alpha = 3
        beta = 1

        V,E0,w0 = sa.ABSubproblem(VOrig,EOrig,wOrig,labels,alpha,beta)
        self.assertEqual(V,[1,2,5,6])
        self.assertEqual(E0,[[1,2],[5,6],[5,1,2]])
        self.assertEqual(w0,[1.0,4.0,6.0])

class TestABUpdate(unittest.TestCase):
    def test_simple(self):
        labels = {1:1,2:1,3:2,4:2,5:3,6:3}
        alpha = 3
        beta = 1

        x = {1:0,2:0,5:0,6:0}
        newlabels = {1:1,2:1,3:2,4:2,5:1,6:1}
        self.assertEqual(sa.ABUpdate(labels,x,alpha,beta),newlabels)

        x[2] = 1.
        x[6] = 1.
        newlabels = {1:1,2:3,3:2,4:2,5:1,6:3}
        self.assertEqual(sa.ABUpdate(labels,x,alpha,beta),newlabels)

#--------------------- Submod Alg --------------------------------#

class TestSubmodAlg(unittest.TestCase):
    def test_AB_simple(self):
        VOrig = [1,2,3,4,5,6]
        EOrig = [[1,3],[1,4],[1,5],[1,6],[2,3],[2,4],[2,5],[2,6],[3,5],[3,6],
                 [4,5],[4,6],[1,3,5],[2,4,6],[1,3,6]]
        wOrig = [1.0 for _ in range(15)]
        
        labels = {1:1,2:1,3:1,4:1,5:2,6:2}
        alpha = 3
        beta = 1
        moveType = 'ab'
        newlabels = sa.submodAlg(VOrig,EOrig,wOrig,labels,moveType,alpha,beta)
        self.assertEqual(newlabels[5],2)
        self.assertEqual(newlabels[6],2)
        self.assertEqual(newlabels[1],newlabels[2])
        self.assertEqual(newlabels[3],newlabels[4])
        self.assertIn(newlabels[1],[1,3])
        self.assertIn(newlabels[3],[1,3])
        self.assertNotEqual(newlabels[1],newlabels[3])

        labels = {1:1,2:3,3:1,4:3,5:2,6:2}
        alpha = 3
        beta = 1
        newlabels = sa.submodAlg(VOrig,EOrig,wOrig,labels,moveType,alpha,beta)
        self.assertEqual(newlabels[5],2)
        self.assertEqual(newlabels[6],2)
        self.assertEqual(newlabels[1],newlabels[2])
        self.assertEqual(newlabels[3],newlabels[4])
        self.assertIn(newlabels[1],[1,3])
        self.assertIn(newlabels[3],[1,3])
        self.assertNotEqual(newlabels[1],newlabels[3])

    def test_AExp_simple(self):
        VOrig = [1,2,3,4,5,6]
        EOrig = [[1,3],[1,4],[1,5],[1,6],[2,3],[2,4],[2,5],[2,6],[3,5],[3,6],
                 [4,5],[4,6],[1,3,5],[2,4,6],[1,3,6]]
        wOrig = [1.0 for _ in range(15)]

        labels = {1:1,2:1,3:1,4:2,5:2,6:2}
        alpha = 3
        moveType = 'aexp'
        newlabels = sa.submodAlg(VOrig,EOrig,wOrig,labels,moveType,alpha)
        self.assertEqual(newlabels[5],2)
        self.assertEqual(newlabels[6],2)
        self.assertEqual(newlabels[1],1)
        self.assertEqual(newlabels[2],1)
        self.assertEqual(newlabels[3],3)
        self.assertEqual(newlabels[4],3)

        labels = {1:1,2:1,3:2,4:2,5:2,6:3}
        alpha = 3
        moveType = 'aexp'
        newlabels = sa.submodAlg(VOrig,EOrig,wOrig,labels,moveType,alpha)
        self.assertEqual(newlabels[5],3)
        self.assertEqual(newlabels[6],3)
        self.assertEqual(newlabels[1],1)
        self.assertEqual(newlabels[2],1)
        self.assertEqual(newlabels[3],2)
        self.assertEqual(newlabels[4],2)

#--------------------- Local Search --------------------------------#
class TestLocalSearchSubmod(unittest.TestCase):
    def test_AB_simple(self):
        V = [1,2,3,4,5,6]
        E = [[1,3],[1,4],[1,5],[1,6],[2,3],[2,4],[2,5],[2,6],[3,5],[3,6],
                 [4,5],[4,6],[1,3,5],[2,4,6],[1,3,6]]
        w = [1.0 for _ in range(15)]

        labels = {1:1,2:1,3:1,4:1,5:1,6:1}
        moveType = 'ab'
        k = 3
        newlabels = sa.localSearchSubmod(V,E,w,k,labels,moveType)
        self.assertEqual(newlabels[1],newlabels[2])
        self.assertEqual(newlabels[3],newlabels[4])
        self.assertEqual(newlabels[5],newlabels[6])
        self.assertNotEqual(newlabels[1],newlabels[3])
        self.assertNotEqual(newlabels[1],newlabels[5])
        self.assertNotEqual(newlabels[3],newlabels[5])

        labels = {1:1,2:2,3:0,4:1,5:1,6:2}
        newlabels = sa.localSearchSubmod(V,E,w,k,labels,moveType)
        self.assertEqual(newlabels[1],newlabels[2])
        self.assertEqual(newlabels[3],newlabels[4])
        self.assertEqual(newlabels[5],newlabels[6])
        self.assertNotEqual(newlabels[1],newlabels[3])
        self.assertNotEqual(newlabels[1],newlabels[5])
        self.assertNotEqual(newlabels[3],newlabels[5])

    def test_AExp_simple(self):
        V = [1,2,3,4,5,6]
        E = [[1,3],[1,4],[1,5],[1,6],[2,3],[2,4],[2,5],[2,6],[3,5],[3,6],
                 [4,5],[4,6],[1,3,5],[2,4,6],[1,3,6]]
        w = [1.0 for _ in range(15)]

        labels = {1:1,2:1,3:1,4:1,5:1,6:1}
        moveType = 'aexp'
        k = 3
        newlabels = sa.localSearchSubmod(V,E,w,k,labels,moveType)
        self.assertEqual(newlabels[1],newlabels[2])
        self.assertEqual(newlabels[3],newlabels[4])
        self.assertEqual(newlabels[5],newlabels[6])
        self.assertNotEqual(newlabels[1],newlabels[3])
        self.assertNotEqual(newlabels[1],newlabels[5])
        self.assertNotEqual(newlabels[3],newlabels[5])

class TestLocalSearch(unittest.TestCase):
    def test_simple(self):
        V = [1,2,3,4,5,6]
        E = [[1,3],[1,4],[1,5],[1,6],[2,3],[2,4],[2,5],[2,6],[3,5],[3,6],
                 [4,5],[4,6],[1,3,5],[2,4,6],[1,3,6]]
        w = [1.0 for _ in range(15)]

        labels = {1:1,2:1,3:1,4:1,5:1,6:1}
        k = 3
        newlabels = sa.localSearch(V,E,w,k,labels)
        self.assertEqual(newlabels[1],newlabels[2])
        self.assertEqual(newlabels[3],newlabels[4])
        self.assertEqual(newlabels[5],newlabels[6])
        self.assertNotEqual(newlabels[1],newlabels[3])
        self.assertNotEqual(newlabels[1],newlabels[5])
        self.assertNotEqual(newlabels[3],newlabels[5])

        #labels = {1:1,2:2,3:0,4:1,5:1,6:2}
        #labels = {i:np.random.randint(0, 3) for i in range(1,7)}
        #newlabels = sa.localSearch(V,E,w,k,labels)

if __name__ == '__main__':
    unittest.main()
