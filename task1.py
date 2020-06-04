import matplotlib.pyplot as plt
import numpy as np

x1 = np.arange(-1,1,0.1)
x2 = np.arange(-1,1,0.1)


X1, X2 = np.meshgrid(x1, x2)


A_alpha = np.zeros([2,2])

A_alpha[1][0] = -0.25

fig, ax = plt.subplots(2,2)



Alpha = [0.10,0.5,2,10]


for i in range(len(Alpha)):

    alpha = Alpha[i]

    A_alpha[0][0] = alpha
    A_alpha[0][1] = alpha

    x1_dot = A_alpha[0][0]*X1 + A_alpha[0][1]*X2
    x2_dot = A_alpha[1][0]*X1 + A_alpha[1][1]*X2



    eigenValues = np.linalg.eig(A_alpha)
    eigenValues = eigenValues[0]

    ax[i//2][i%2].streamplot(X1,X2,x1_dot,x2_dot, density = 1,linewidth= 0.5)
    ax[i//2][i%2].set_title(r'$ \alpha =  $' + str(alpha) + r', $\lambda_{1} = $' + str(round(eigenValues[0],3))+ r', $\lambda_{2} = $' + str(round(eigenValues[1],3)))

    ax[i//2][i%2].set_ylim(-1,1)
    ax[i//2][i%2].set_xlim(-1,1)
    ax[i//2][i%2].set_aspect(aspect=1)
    ax[i//2][i%2].plot(0,0,'ro')



fig2, ax2 = plt.subplots(2,3)


Mat = np.zeros([6,2,2])

#Stable node phase portrait (negative real eigenValues)
Mat[0] = [[-2.5,0],[3,-2]]
#Stable focus phase portrait (negative complex conjugates eigenValues)
Mat[1] = [[-2,3],[-3,-2]]
#unstalbe saddle point
Mat[2] = [[2,0],[2,1]]

#unstalbe centre
Mat[3] = [[0,1],[-1,0]]

#unstable node

Mat[4] = [[-1,4],[-2,5]]

#Unstable focus
Mat[5] = [[2,3],[-3,2]]


for i in range(6):

    x1_dot = Mat[i][0][0]*X1 + Mat[i][0][1]*X2
    x2_dot = Mat[i][1][0]*X1 + Mat[i][1][1]*X2



    eigenValues = np.linalg.eig(Mat[i])
    eigenValues = eigenValues[0]

    ax2[i//3][i%3].streamplot(X1,X2,x1_dot,x2_dot, density = 1,linewidth= 0.5)
    ax2[i//3][i%3].set_title(r'$\lambda_{1} = $' + str(round(eigenValues[0],3))+ r', $\lambda_{2} = $' + str(round(eigenValues[1],3)))

    ax2[i//3][i%3].set_ylim(-1,1)
    ax2[i//3][i%3].set_xlim(-1,1)
    ax2[i//3][i%3].set_aspect(aspect=1)
    ax2[i//3][i%3].plot(0,0,'ro')











plt.show()
