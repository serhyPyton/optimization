import numpy as np 
import copy
popul = np.random.randn(10, 2).tolist()
a = np.full((10,2),4)
b = np.full((10,2),-2)
popul = popul * a-b
def func(x,y):
    return 20 + x**2+y**2-10*(np.cos(2*np.pi*x)+np.cos(2*np.pi*y))
#    return x**2-2*x*y+6*y**2+x-y
def f_min(popul):
    return np.min([func(x,y) for x, y in popul])
#print(population[0:20][:])
f_min(popul)
#plot

_x= np.arange(-10.,10.,0.5)
_y= np.arange(-10.,10.,0.5)  
_x,_y=np.meshgrid(_x,_y)
_z= func(_x, _y)

from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.view_init(62, 11)
ax.plot_surface(_x, _y, _z, cmap=cm.coolwarm,rstride=1, cstride=1, linewidth=0.5, antialiased=True, color='blue',alpha=0.8)
ax.plot_wireframe(_x, _y, _z, linewidth=0.5, antialiased=True, color='blue')

#plot
print(_x[1])
k=0
while k<100:
    order = np.argsort([func(*item) for item in popul])
    x_ord = abs(np.random.normal(0, 2, 10)).astype(int)
    y_ord = abs(np.random.normal(0, 2, 10)).astype(int)
    x_ord = x_ord%10
    y_ord = y_ord%10
    newpop = np.copy(popul)
    i=0
    for x1, y1 in zip(x_ord, y_ord):   #create new population
        newpop[i][0]=popul[order[x1]][0]
        newpop[i][1]=popul[order[y1]][1]
        i+=1
    popul = np.copy(newpop)
    for i in range(0,10):
        if np.random.rand(1,1)>0.8:
            if np.random.rand(1,1)>0.5:
                if np.random.rand(1,1)<0.5:
                    popul[i][0]+=0.05/abs(np.random.rand(1, 1))
                else:
                    popul[i][0]-=0.05/abs(np.random.rand(1, 1))
            else:
                if np.random.rand(1,1)<0.5:
                    popul[i][1]+=0.05/abs(np.random.rand(1, 1))
                else:
                    popul[i][1]-=0.05/abs(np.random.rand(1, 1))
    #draw

#    ax = fig.add_subplot(111, projection='3d')
#    ax.view_init(62, 11)
#    ax.plot_surface(_x, _y, _z, cmap=cm.coolwarm,rstride=1, cstride=1, linewidth=0.5, antialiased=True, color='blue',alpha=0.8)
#    ax.plot_wireframe(_x, _y, _z, linewidth=0.5, antialiased=True, color='blue')
    plt.plot(popul[:,0], popul[:,1], func(popul[:,0], popul[:,1]), 'ro')
    plt.draw()
    #draw  
    plt.pause(0.6)
    k+=1
    print("min ",f_min(popul))

plt.show()