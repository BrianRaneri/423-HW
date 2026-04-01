import numpy as np
import matplotlib.pyplot as plt

def g(x):
    return (1+4*(x-0.5)**2)

def gprime(x):
    return (8*x-4)

def u_exact(x):
    c=4-2*uix**2
    return -1* (2-(c/(1+4*(x-0.5)**2)))**0.5

def update_solution(u_arr):
    u_arr_new = u_arr.copy()
    for i in range(1,len(u_arr)-1):
        neg_u_i5 = (u_arr[i-1]+u_arr[i])/2
        pos_u_i5 = (u_arr[i+1]+u_arr[i])/2

        RHS = (1-((u_arr[i]**2)/2))*gprime(x_arr[i])/g(x_arr[i])

        if (neg_u_i5 > 0 and pos_u_i5 > 0):
            # Supersonic Point
            u_arr_new[i]=dt*RHS-(neg_u_i5*dt/dx)*(u_arr[i]-u_arr[i-1])+u_arr[i]

        elif (neg_u_i5 > 0 and pos_u_i5 < 0):
            # Shock Point
            u_arr_new[i]=u_arr[i] + dt*RHS - (neg_u_i5*dt/dx)*(u_arr[i]-u_arr[i-1]) - (pos_u_i5*dt/dx)*(u_arr[i+1]-u_arr[i])

        elif (neg_u_i5 <= 0 and pos_u_i5 >= 0):
            # Sonic Point
            u_arr_new[i]= (dt*RHS+u_arr[i]) / (1+(dt/(2*dx))*(u_arr[i+1]-u_arr[i-1]))

        elif (neg_u_i5 < 0 and pos_u_i5 < 0):
            # Subsonic Point
            u_arr_new[i]=dt*RHS-(pos_u_i5*dt/dx)*(u_arr[i+1]-u_arr[i])+u_arr[i]
        else:
            pass

    return u_arr_new


ix=211
#uix=-1*1.5**0.5
uix=-1

dx=1/(ix-1)
dt=dx/(-uix)

x_arr=np.linspace(0,1,211)

u_arr = np.zeros_like(x_arr)
u_arr[0]=uix
u_arr[-1]=uix

max_iter = 2000
tol = 1e-6

for i in range(max_iter):
    u_new = update_solution(u_arr)
    print(np.max(np.abs(u_new-u_arr)))
    if np.max(np.abs(u_new-u_arr))<=tol:
        print(f'Steady State Reached at Iteration {i}')
        break

    u_arr = u_new

plt.plot(x_arr,u_exact(x_arr),label='exact')
plt.plot(x_arr,u_arr,label='Numerical')
plt.title(fr'$u_{{ix}} = {uix:.3f}$')
plt.grid(True)
plt.xlim([0,1])
plt.show()
plt.legend()
