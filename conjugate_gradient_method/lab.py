import numpy as np

teta=3
def f(x):
#    return 20 + x[0]**2+x[1]**2-10*(np.cos(2*np.pi*x[0])+np.cos(2*np.pi*x[1]))
    return (x[0]**2+x[1]-11)**2 + (x[1]**2+x[0]-6)**2
#    return(x[0]*x[0]+2*x[1]*x[1]+np.exp(x[0]*x[0]+x[1]*x[1])-x[0]+2*x[1])
def f1(x, y):
#    return 20 + x**2+y**2-10*(np.cos(2*np.pi*x)+np.cos(2*np.pi*y))
    return (x**2+y-11)**2 + (y**2+x-6)**2
#    return(x*x+2*y*y+np.exp(x*x+y*y)-x+2*y)

def get_grad(x):
#    return np.array([2*(x[0]+10*np.pi*np.sin(2*np.pi*x[0])) ,2*(x[1]+10*np.pi*np.sin(2*np.pi*x[1])) ],float)
    return np.array([2*(2*x[0]*(x[0]**2+x[1]-11)+x[0]+x[1]**2-6),2*(2*x[1]*(x[0]+x[1]**2-6)+x[0]**2+x[1]-11)], float)
#    return np.array([2*x[0]+2*x[0]*np.exp(x[0]*x[0]+x[1]*x[1])-1,4*x[1]+2*x[1]*np.exp(x[0]*x[0]+x[1]*x[1])+2],float)   

x = np.array([[-3],[2]], dtype=np.float64)
k=0
a=0
grad_f=get_grad(x)
g=-grad_f
h=-grad_f

#draw
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
 

x1 = np.arange(-5.,5.,0.5)
y1 = np.arange(-5.,5.,0.5)
x2, y2 = np.meshgrid(x1, y1)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x2, y2, f1(x2, y2), cmap=cm.coolwarm,rstride=1, cstride=1, linewidth=0.5, antialiased=True, color='blue',alpha=0.8)
ax.plot_wireframe(x2, y2, f1(x2, y2), linewidth=0.5, antialiased=True, color='blue')
ax.set_xlabel('x', color='blue')
ax.set_ylabel('y', color='blue')
ax.set_zlabel('z', color='blue')

plt.plot(x[0], x[1], f(x), 'ro')
ax.view_init(62, 11)
plt.draw()
plt.pause(0.5)
#draw
print("f(x) = ", f(x))
for a1 in np.arange(-1.,1.,0.5):
    for a2 in np.arange(-1.,1.,0.5):
        print("start dot is: ", x.flatten())
        stop = 0
        while (stop != 1):
            rho = 1
            min = 100
            for a in np.arange(0.,1.,0.01):
                s=f(x+a*h)
                if s<min:
                    min=s
                    rho=a
            x=x+rho*h
            g_old=g
            g=-get_grad(x)
            if np.linalg.norm(g) < 0.001:
                stop = 1
            beta=np.dot(np.transpose(g-g_old),g)/np.dot(np.transpose(g_old),g_old)
            if k%(teta-1)==0:
                beta=0
            h=g+beta*h
            k=k+1
            #draw
            plt.plot(x[0], x[1], f(x), 'ro')
            plt.draw()
            plt.pause(0.2)
            #draw
        #draw
        plt.plot(x[0], x[1], f(x), 'go')
        plt.draw()
        plt.pause(0.2)
        #draw
        print("x = ", x.flatten())
        print("f(x) = ", min)

        x = np.array([[a1],[a2]], dtype=np.float64)
        k=0
        a=0
        grad_f=get_grad(x)
        g=-grad_f
        h=-grad_f

plt.show()
