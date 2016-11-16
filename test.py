import unittest

import numpy
numpy.random.seed(1234)


class MadeUpTestCase(unittest.TestCase):
    
    def setUp(self):
    
        import complex_step
    
        def f(t, x, y):
            return x**2 * y

        def g(t, x, y):
            return 5*x + numpy.sin(y)

        self.integrator = complex_step.ComplexStep(f, g)
        

    def test_integrate(self):
        
        self.integrator.integrate()


    def test_jacobian(self):

        def analytical(t, v):
            
            x, y = v

            df = [2*x*y,         x**2]
            dg = [    5, numpy.cos(y)]

            return [df, dg]

        t, v, h = 1.0, numpy.random.random(2), 10e-6

        self.assertTrue(
            numpy.allclose(analytical(t, v), self.integrator.jacobian(t, v, h))
        )



class LotkaVolterraTestCase(unittest.TestCase):
    
    def setUp(self):
    
        import complex_step
    
        self.alpha, self.beta, self.delta, self.gamma = 100, 1, 2, 20

        def f(t, x, y):
            return self.alpha*x - self.beta*x*y

        def g(t, x, y):
            return self.delta*x*y - self.gamma*y

        self.integrator = complex_step.ComplexStep(f, g)
        

    def test_integrate(self):
        
        self.integrator.integrate()

        
    def test_jacobian(self):

        def analytical(t, v):
    
            x, y = v

            df = [self.alpha - self.beta*y,              -self.beta*x]
            dg = [            self.delta*y, self.delta*x - self.gamma]

            return numpy.stack((df, dg))

        t, v, h = 1.0, numpy.random.random(2), 10e-6

        self.assertTrue(
            numpy.allclose(analytical(t, v), self.integrator.jacobian(t, v, h))
        )
