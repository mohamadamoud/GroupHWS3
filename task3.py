import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

x1 = np.arange(-2,2,0.01)
x2 = np.arange(-2,2,0.01)


X1, X2 = np.meshgrid(x1, x2)


fig, ax = plt.subplots(1,3)


Alpha = [-1,0,1]


for i in range(len(Alpha)):
    alpha = Alpha[i]
    x1_dot = alpha*X1 - X2 - np.multiply(X1,np.multiply(X1,X1) + np.multiply(X2,X2))
    x2_dot = X1 + alpha*X2 - np.multiply(X2,np.multiply(X1,X1) + np.multiply(X2,X2))


    ax[i].streamplot(X1,X2,x1_dot,x2_dot, density = 1.25,linewidth= 0.5)
    ax[i].set_title(r'$ \alpha =  $' + str(alpha))
    ax[i].set_aspect('equal')








fig1 = plt.figure()
ax1 = fig1.gca(projection='3d')

dt = 0.001
alpha = 1

x10 = 2
x20 = 0

time = np.arange(0,30,dt)

x1n = np.zeros(len(time))
x2n = np.zeros(len(time))

x1n[0] = x10
x2n[0] = x20


for i in range(1,len(time)):
    x1n[i] = x1n[i-1] + dt*(x1n[i-1] - x2n[i-1] - x1n[i-1]*(x1n[i-1]*x1n[i-1] + x2n[i-1]*x2n[i-1]))
    x2n[i] = x2n[i-1] + dt*(x1n[i-1] + x2n[i-1] - x2n[i-1]*(x1n[i-1]*x1n[i-1] + x2n[i-1]*x2n[i-1]))




ax1.plot(x1n,x2n,time,label = 'Starting point 'r'$(2,0)$',color = 'blue',linestyle = 'dashed')

x10 = 0.5
x20 = 0

time = np.arange(0,30,dt)

x1n = np.zeros(len(time))
x2n = np.zeros(len(time))

x1n[0] = x10
x2n[0] = x20


for i in range(1,len(time)):
    x1n[i] = x1n[i-1] + dt*(x1n[i-1] - x2n[i-1] - x1n[i-1]*(x1n[i-1]*x1n[i-1] + x2n[i-1]*x2n[i-1]))
    x2n[i] = x2n[i-1] + dt*(x1n[i-1] + x2n[i-1] - x2n[i-1]*(x1n[i-1]*x1n[i-1] + x2n[i-1]*x2n[i-1]))




ax1.plot(x1n,x2n,time,label = 'Starting point 'r'$(0.5,0)$',color = 'red', linestyle = 'dashed')

ax1.set_title(r'$\alpha= 1$')
plt.legend()

fig2 = plt.figure()
ax2 = fig2.gca(projection='3d')



A1 = np.zeros(int(33e3))
A2 = np.zeros(int(33e3))
X  = np.zeros(int(33e3))
ctr = 0
for a1 in np.arange(-4,4,0.03):
    for a2 in np.arange(-4,4,0.03):
        rrr =np.roots([-1,0,a2,a1])
        #rrr = np.roots([a1,a2,0,-1])

        xd0 = round(rrr[0],5)
        xd1 = round(rrr[1],5)
        xd2 = round(rrr[2],5)
        if(np.isreal(xd0) and np.isreal(xd1) and np.isreal(xd2)):

            A1[ctr] = a1
            A1[ctr+1] = a1
            A1[ctr+2] = a1

            A2[ctr] = a2
            A2[ctr+1] = a2
            A2[ctr+2] = a2

            X[ctr] = xd0
            X[ctr+1] = xd1
            X[ctr+2] = xd2

            ctr +=3

print(ctr)

#X, Y = np.meshgrid(X, Y)
ax2.scatter(A1,A2,X, s= 0.1,cmap='cool',c = X)
ax2.set_xlabel(r'$\alpha_1$')
ax2.set_ylabel(r'$\alpha_2$')

ax2.set_zlabel(r'$x$')
ax2.set_title("Cusp Bifurcation")


#ax.axis([0.5,2.1,0,2])
#ax.xaxis.set_ticks([])
#ax.yaxis.set_ticks([])








plt.show()
