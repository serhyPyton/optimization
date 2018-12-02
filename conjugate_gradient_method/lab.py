import numpy as np
teta=3
x = np.array([[0],[0]],float)

print("x", x)
k=0

def f(x):
	return(x[0]*x[0]+2*x[1]*x[1]+np.exp(x[0]*x[0]+x[1]*x[1])-x[0]+2*x[1])

def get_grad(x):
    grad_f = np.array([2*x[0]+2*x[0]*np.exp(x[0]*x[0]+x[1]*x[1])-1,4*x[1]+2*x[1]*np.exp(x[0]*x[0]+x[1]*x[1])+2],float)   
    return(grad_f)
    
grad_f=get_grad(x)
print(grad_f)
g=-grad_f
h=-grad_f

while 1:
	rho = 1
	max = 100
	for a in np.arange(0.,1.,0.01):
	    s=f(x+a*h)
	    if s<max:
	        max=s
	        rho=a

	x=x+rho*h

	g_old=g
	g=-get_grad(x)

	if np.linalg.norm(g) <0.001:
		break

	beta=np.dot(np.transpose(g-g_old),g)/np.dot(np.transpose(g_old),g_old)
	if k%(teta-1)==0:
	    beta=0

	h=g+beta*h
	k=k+1
print("x ", x)