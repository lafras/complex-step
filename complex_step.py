import math
import operator
Im = operator.attrgetter('imag')

import numpy
from numpy import sin, cos

import scipy.integrate

from matplotlib import pyplot


class ComplexStep:
    """
        Integrates systems with two variables and two equations. 
        The Jacobian is calculated using the complex step method.
    """

    def __init__(self, f, g):

        if (f.__code__.co_argcount != 3) or (g.__code__.co_argcount != 3):
            raise TypeError("Functions must take three arguments, f(t, x, y).")

        self.f, self.g = f, g


    def system(self, t, v):
        
        dx = numpy.zeros(2)

        x, y = v

        dx[0] = self.f(t, x, y)
        dx[1] = self.g(t, x, y)

        return dx


    def jacobian(self, t, v, h):
        
        dfdx = numpy.zeros((2, 2))
        
        x, y = v
        f, g = self.f, self.g

        dfdx[0, :] = [Im(f(t, x + 1j*h, y)) / h, Im(f(t, x, y + 1j*h)) / h]
        dfdx[1, :] = [Im(g(t, x + 1j*h, y)) / h, Im(g(t, x, y + 1j*h)) / h]

        return dfdx


    def integrate(self):
    
        r = scipy.integrate.ode(self.system, self.jacobian)
        r.set_initial_value([1, 1], 0).set_jac_params(1e-6)

        steps = 1000
        stop = 1.0
        dt = stop / steps

        self.simulation = numpy.zeros((steps, 3))

        step = 0

        while r.successful() and r.t < stop:
            self.simulation[step, 0], self.simulation[step, 1:] = r.t, r.integrate(r.t+dt)
            step += 1


    def plot(self):

        pyplot.plot(self.simulation[:,0], self.simulation[:,1:])
        pyplot.show()

