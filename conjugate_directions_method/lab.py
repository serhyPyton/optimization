import numpy as np
x = np.array([[0],[0]],float)
H = np.array([[1,0],[0,1]],float)
A = np.array([[2,-2],[-2,12]],float)
b = np.array([[1],[-1]],float)

z= np.dot(A, x)+b

h=-np.dot(np.transpose(H),z)

rho = -np.dot(np.transpose(z),h)/np.dot(np.transpose(np.dot(A,h)),h)

x_old=x
x=x+rho*h

z= np.dot(A, x)+b


H0=H

while (np.linalg.norm(z)>0.0001):
	r=x-x_old
	g=np.dot(A,r)

	H=H-np.dot(np.dot(H0,g),np.transpose(r))/np.dot(np.transpose(r),g)

	h=-np.dot(np.transpose(H),z)

	rho = -np.dot(np.transpose(z),h)/np.dot(np.transpose(np.dot(A,h)),h)

	x_old=x
	x=x+rho*h

	z= np.dot(A, x)+b

print("rezult", x)