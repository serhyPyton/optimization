import numpy as np
x = np.array([[-8],[8]],float)
H = np.array([[1,0],[0,1]],float)
A = np.array([[2,-2],[-2,12]],float)
b = np.array([[1],[-1]],float)
alpha = 0

def f(x):
    return ((np.dot(np.transpose(np.dot(A,x)),x)/2.) + np.dot(np.transpose(b),x) + alpha).flatten()

def f1(x,y):
    return x**2-2*x*y+6*y**2+x-y


#plot

_x= np.arange(-10.,10.,0.5)
_y= np.arange(-10.,10.,0.5)  
_x,_y=np.meshgrid(_x,_y)
_z= f1(_x, _y)

from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.view_init(62, 11)
ax.plot_surface(_x, _y, _z, cmap=cm.coolwarm,rstride=1, cstride=1, linewidth=0.5, antialiased=True, color='blue',alpha=0.8)
ax.plot_wireframe(_x, _y, _z, linewidth=0.5, antialiased=True, color='blue')

#plot
#draw
plt.plot(x[0], x[1], f(x), 'ro')
plt.draw()
plt.pause(0.2)
#draw

z= np.dot(A, x)+b

h=-np.dot(np.transpose(H),z)

rho = -np.dot(np.transpose(z),h)/np.dot(np.transpose(np.dot(A,h)),h)

x_old=x
x=x+rho*h

z= np.dot(A, x)+b


H0=H

#draw
plt.plot(x[0], x[1], f(x), 'ro')
plt.draw()
plt.pause(0.2)
#draw
time=0
while (np.linalg.norm(z)>0.0001):
    time+=1

    r=x-x_old
    g=np.dot(A,r)

    H=H-np.dot(np.dot(H0,g),np.transpose(r))/np.dot(np.transpose(r),g)

    h=-np.dot(np.transpose(H),z)

    rho = -np.dot(np.transpose(z),h)/np.dot(np.transpose(np.dot(A,h)),h)
    
    x_old=x
    x=x+rho*h
    #draw
    plt.plot(x[0], x[1], f(x), 'ro')
    plt.draw()
    plt.pause(0.2)
    #draw
    z= np.dot(A, x)+b
print( f(x) )
print("rezult", x)
print(time)
plt.show()