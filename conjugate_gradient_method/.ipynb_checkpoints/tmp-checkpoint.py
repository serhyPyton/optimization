import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
teta=3
def f(x):
	return(x[0]*x[0]+2*x[1]*x[1]+np.exp(x[0]*x[0]+x[1]*x[1])-x[0]+2*x[1])
def f1(x, y):
	return(x*x+2*y*y+np.exp(x*x+y*y)-x+2*y)


def get_grad(x):
    grad_f = np.array([2*x[0]+2*x[0]*np.exp(x[0]*x[0]+x[1]*x[1])-1,4*x[1]+2*x[1]*np.exp(x[0]*x[0]+x[1]*x[1])+2],float)   
    return(grad_f)

print("iterr ", "1")
print("step ", "1")
x = np.array([[-1.5],[0]], dtype=np.float64)
print("x_start", x.reshape(2,))
k=0
a=0
print("step ", "2")
grad_f=get_grad(x)
print("grad = ", grad_f.reshape(2,))
g=-grad_f
h=-grad_f

print(f(x+0.86*h))
stape = 0
while (stape != 1):
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
    if np.linalg.norm(g) < 0.01:
        print("NOOOOOOOORMMMMM = ",np.linalg.norm(g))
        stape = 1
    beta=np.dot(np.transpose(g-g_old),g)/np.dot(np.transpose(g_old),g_old)
    if k%(teta-1)==0:
        beta=0

    h=g+beta*h
    k=k+1
print("x = ", x)
rez = x
from matplotlib import cm
x1 = np.arange(-1.,1.,0.1)
y1 = np.arange(-1.,1.,0.1)
x, y = np.meshgrid(x1, y1)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plotting the 3 planes
ax.plot_surface(x, y, f1(x, y), cmap=cm.coolwarm,rstride=1, cstride=1, linewidth=0, antialiased=True, color='blue')
ax.set_xlabel('x', color='blue')
ax.set_ylabel('y', color='blue')
ax.set_zlabel('z', color='blue')


plt.plot(rez[0], rez[1], 'ro')
plt.savefig("fig1.jpg")
for angle in range(0, 360):
    ax.view_init(30, angle)
    plt.draw()
    plt.pause(.01)
plt.show()

