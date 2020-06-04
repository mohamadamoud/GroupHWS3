#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 03:39:35 2020

@author: narminmammadova
"""

from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def logistic(r, x):
    return r * x * (1 - x)
x = np.linspace(0, 1)
fig, ax = plt.subplots(1, 1)
ax.plot(x, logistic(2, x), 'k')

def plot_system(r, x0, n, ax=None):
    # Plot the function and the
    # y=x diagonal line.
    t = np.linspace(0, 4)
    ax.plot(t, logistic(r, t), 'k', lw=2)
    ax.plot([0, 1], [0, 1], 'k', lw=2)

    # Recursively apply y=f(x) and plot two lines:
    # (x, x) -> (x, y)
    # (x, y) -> (y, y)
    x = x0
    for i in range(n):
        y = logistic(r, x)
        # Plot the two lines.
        ax.plot([x, x], [x, y], 'k', lw=1)
        ax.plot([x, y], [y, y], 'k', lw=1)
        # Plot the positions with increasing
        # opacity.
        ax.plot([x], [y], 'ok', ms=10,
                alpha=(i + 1) / n)
        x = y

    ax.set_xlim(0, 4)
    ax.set_ylim(0, 1)
    ax.set_title(f"$r={r:.1f}, \, x_0={x0:.1f}$")


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8),
                               sharey=True)
plot_system(2, .1, 10, ax=ax1)
plot_system(3, .1, 10, ax=ax2)

n=10000
r=np.linspace(2,4,n)
iterations=1000
last=100
x=1e-5
fig,(ax1,ax2)=plt.subplots(2,1,figsize=(8,9),sharex=True)
for i in range(iterations):
    x = logistic(r, x)
    if i>=(iterations-last):

        ax1.plot(r, x, ',k', alpha=.25)
ax1.set_xlim(2.5, 4)
ax1.set_title("Bifurcation diagram")

def lorenz(sigma, rho, beta):
    """
    Create rhs of lorenz system with given parameters.
    """
    def fun(t, x):
        return np.array([
            sigma * (x[1] - x[0]),
            x[0] * (rho - x[2]) - x[1],
            x[0] * x[1] - beta * x[2]
        ])
    return fun


def plot_trajectories(sigma, rho, beta):
    """
    Subtask 2:
    Plot 2 trajectories of Lorenz system from slightly different initial values.
    """
    fun = lorenz(sigma, rho, beta)
    fig = plt.figure()
    fig.suptitle(rf'Lorenz system with $\sigma$ = {sigma}, $\rho$ = {rho}, $\beta$ = {beta}')
    ax = fig.gca(projection='3d')
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$y$')
    ax.set_zlabel(r'$z$')
    t_span = (0, 100)

    for x0 in [np.array([10.0, 10.0, 10.0]), np.array([10.0 + 10**(-8), 10.0, 10.0])]:
        bunch = solve_ivp(fun, t_span, x0)
        ax.plot(bunch.y[0, :], bunch.y[1, :], bunch.y[2, :], label=rf'$x_0$ = ({x0[0]}, {x0[1]}, {x0[2]})')

    ax.legend()
    plt.show()


if __name__ == '__main__':

    plot_trajectories(10, 28, 8 / 3)
    plot_trajectories(10, 0.5, 8 / 3)
