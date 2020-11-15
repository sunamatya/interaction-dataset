#!/usr/bin/env python


import matplotlib.pyplot as plt
import scipy.special as sp   
import numpy as np

def bernstein_basis(i, n, t):
        return sp.binom(n, i) * t**i * (1-t)**(n-i)

def bezier_curve(points, nTimes=1000):
    """
       Given a set of control points, return the
       bezier curve defined by the control points.

       points should be a list of lists, or list of tuples
       such as [ [1,1], 
                 [2,3], 
                 [4,5], ..[Xn, Yn] ]
        nTimes is the number of time steps, defaults to 1000

        See http://processingjs.nihongoresources.com/bezierinfo/
    """

    nPoints = len(points)
    xPoints = np.array([p[0] for p in points])
    yPoints = np.array([p[1] for p in points])

    t = np.linspace(0.0, 1.0, nTimes)

    polynomial_array = np.array([ bernstein_basis(i, nPoints-1, t) for i in range(0, nPoints)   ])

    xvals = np.dot(xPoints, polynomial_array)
    yvals = np.dot(yPoints, polynomial_array)

    return xvals, yvals


def bezier_points(xy_points_T, degree):
    '''
        Given points P, interpolate the curve with n degree bezier curve

        xy_points_T is a 2D array where xy_points_T[0] is x values 
                                    and xy_points_T[1] is y values
    '''
    assert len(xy_points_T)==2
    assert len(xy_points_T[0]) == len(xy_points_T[1])

    N = len(xy_points_T[0])
    P = np.transpose(xy_points_T)
    M = np.empty((N,N))

    for i in range(N):
        for j in range(N):
            M[i,j] = bernstein_basis(j, N-1, (i/(N-1)))


    lhs = np.matmul(np.transpose(M), M)
    rhs = np.matmul(np.transpose(M),P)
    bez = np.linalg.solve(lhs, rhs)
    
    return bez

'''
    xvals, yvals = bezier_curve(bez, nTimes=1000)
    plt.plot(xvals, yvals)
    #plt.plot(xy_points_T[0], xy_points_T[1], "ro")
    for nr in range(len(bez)):
        plt.text(bez[nr][0], bez[nr][1], nr)

    plt.show()
'''


'''
bez = np.array([[1,2], [1,3], [2,2], [2,1]])

xvals, yvals = bezier_curve(bez, nTimes=1000)
xpoints = [p[0] for p in bez]
ypoints = [p[1] for p in bez]

plt.plot(xvals, yvals)
plt.plot(xpoints, ypoints, "ro")
for nr in range(len(bez)):
    plt.text(bez[nr][0], bez[nr][1], nr)

plt.show()
'''