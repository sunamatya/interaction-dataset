#!/usr/bin/env python


import matplotlib.pyplot as plt
import scipy.special as sp   
import numpy as np

def bernstein_basis(i, n, t):
        return sp.binom(n, i) * t**i * (1-t)**(n-i)

def global_to_local(min, max, u):
        return (u-min)/(max-min)

def decasteljaus(bez, t):
    '''
        Given a set of control points and a t,
        return the point on the bezier curve 
        evaluated at t
    '''

    if t < 0 or t > 1:
        print("Error -- invalid t: %.2f" % (t))
        return []
    
    N = len(bez) - 1
    if N == 0:
        # base case: point on the line found
        return bez
    else:
        bx      = np.array([b[0] for b in bez])
        by      = np.array([b[1] for b in bez])
        xcpt    = [((1-t)*bx[i] + t*bx[i+1]) for i in range(N)]
        ycpt    = [((1-t)*by[i] + t*by[i+1]) for i in range(N)]
        new_bez = [xcpt, ycpt]
        new_bez = np.transpose(new_bez)
        return decasteljaus(new_bez, t)
    



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


def bezier_points(xy_points_T,degree=3):
    '''
        Given points P, interpolate the curve with n degree bezier curve

        xy_points_T is a 2D array where xy_points_T[0] is x values 
                                    and xy_points_T[1] is y values
    '''
    assert len(xy_points_T)==2
    assert len(xy_points_T[0]) == len(xy_points_T[1])
    err = 1
    N   = len(xy_points_T[0])
    P   = np.transpose(xy_points_T)
    M   = np.empty((N,N))

    for i in range(N):
        for j in range(N):
            M[i,j] = bernstein_basis(j, N-1, (i/(N-1)))

    try:
        lhs = np.matmul(np.transpose(M), M)
        rhs = np.matmul(np.transpose(M),P)
        bez = np.linalg.solve(lhs, rhs)
    except:
        err = -1
        bez = []

    return err, bez

def bezier_metrics(traj_dict):
    
    N = 500
    t = np.linspace(0.0, 1.0, n)
