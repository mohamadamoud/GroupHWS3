import matplotlib.pyplot as plt
import numpy as np
import math



def roots(a,b,c):
    if(b**2 - 4*a*c < 0):
        x1 = None
        x2 = None

    else:
        x1 = (-b + math.sqrt(b**2 - 4*a*c))/(2*a)
        x2 = (-b - math.sqrt(b**2 - 4*a*c))/(2*a)

    return x1,x2

#Alpha range
alpha_min = -1
alpha_max = 5
step = 0.001

size = len(np.arange(alpha_min,alpha_max+step,step))

fig1 = plt.figure(1)

ax1 = fig1.add_subplot(1,1,1)


xa = np.zeros(size)
xb = np.zeros(size)
aa = np.zeros(size)
ctr = 0
for alpha in np.arange(alpha_min,alpha_max + step,step):
    xa[ctr],xb[ctr] = roots(-1,0,alpha)

    aa[ctr] = alpha
    ctr += 1

ax1.grid()


ax1.set_xlabel(r'$\alpha$')
ax1.set_ylabel("Roots")


ax1.plot(aa,xb,label=r'$\dot{x} = \alpha - x^{2}$',linestyle='solid',color='blue')
ax1.plot(aa,xa,linestyle='dashed',color='blue')

#ax1.set_ylim([-4,-4])
#ax1.set_xlim([-1,11])
ax1.set_xlim([-1,5])
ax1.set_ylim([-4,4])



#fig2 = plt.figure(2)

#ax2 =  fig2.add_subplot(1,1,1)


xa = np.zeros(size)
xb = np.zeros(size)
aa = np.zeros(size)
ctr = 0
for alpha in np.arange(alpha_min,alpha_max + step,step):
    xa[ctr],xb[ctr] = roots(-2,0,alpha-3)

    aa[ctr] = alpha
    ctr += 1

#ax2.grid()


#ax2.set_xlabel(r'$\alpha$')

ax1.plot(aa,xb,label=r'$\dot{x} = \alpha - 2x^{2} - 3$',linestyle='solid',color='red')
ax1.plot(aa,xa,linestyle='dashed',color='red')

ax1.set_title("Bifurcation Diagram")

plt.legend()
#ax1.set_xlim([-1,11])
#ax1.set_ylim([-4,4])

#ax2.set_ylim([-4,-4])
#ax2.set_xlim([-1,11])

#x1 = np.arange(-1,1,0.1)
#time = np.arange(-1,1,0.1)
#
#xdot = np.zeros(len(x1))
#
#X1, T = np.meshgrid(x1,time)







plt.show()
