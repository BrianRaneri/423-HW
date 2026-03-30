import matplotlib.pyplot as plt
import numpy as np


def exact_solution(x):
    return x**3-3*x**2+6*x-6+6*np.exp(-x)

def int_function(x,u):
    return x**3-u


def propogate_euler(dx_arr,u0):

    u_arr=np.zeros_like(dx_arr)
    u_arr[0] = u0

    for i in range(len(dx_arr)-1):
        u_arr[i+1] = u_arr[i] + h*int_function(dx_arr[i],u_arr[i])

    return u_arr

def propogate_improved(dx_arr,u0):
    u_arr=np.zeros_like(dx_arr)
    u_arr[0] = u0

    for i in range(len(dx_arr)-1):
        u_pred = u_arr[i] + h*int_function(dx_arr[i],u_arr[i])
        u_arr[i+1] = u_arr[i] + (h/2)*(int_function(dx_arr[i],u_arr[i])+int_function(dx_arr[i],u_pred))
    
    return u_arr

def propogate_rk4(dx_arr,u0):
    u_arr=np.zeros_like(dx_arr)
    u_arr[0] = u0    

    for i in range(len(dx_arr)-1):
        a_i=h*int_function(dx_arr[i],u_arr[i])
        b_i=h*int_function(dx_arr[i]+(h/2),u_arr[i]+a_i/2)      
        c_i=h*int_function(dx_arr[i]+(h/2),u_arr[i]+b_i/2) 
        d_i=h*int_function(dx_arr[i]+h,u_arr[i]+c_i) 

        u_arr[i+1]=u_arr[i]+ (1/6)*(a_i+2*b_i+2*c_i+d_i)

    return u_arr
  

def calc_error(u_euler,u_improved,u_rk4,dx_coarse):
    err_euler=np.zeros_like(u_euler)
    err_improved = np.zeros_like(u_improved)
    err_rk4 = np.zeros_like(u_rk4)
    for i in range(len(dx_coarse)):
        err_euler[i]=np.abs(u_euler[i]-exact_solution(dx_coarse[i]))
        err_improved[i]=np.abs(u_improved[i]-exact_solution(dx_coarse[i]))
        err_rk4[i]=np.abs(u_rk4[i]-exact_solution(dx_coarse[i]))
    
    return err_euler,err_improved,err_rk4





h=0.2
u0=0

dx_arr_coarse =np.arange(0,1+h,h)
u_euler = propogate_euler(dx_arr_coarse,0)
u_improved = propogate_improved(dx_arr_coarse,0)
u_rk4 = propogate_rk4(dx_arr_coarse,0)

# Solve exact solution
h=0.05
dx_arr_fine = np.arange(0,1+h,h)
exact_solution_arr=exact_solution(dx_arr_fine)

plt.figure()
plt.plot(dx_arr_fine,exact_solution_arr,label='Exact Solution')
plt.plot(dx_arr_coarse,u_euler,marker='o',label='Euler Cauchy')
plt.plot(dx_arr_coarse,u_improved,marker='o',label='Improved Euler Cauchy')
plt.plot(dx_arr_coarse,u_rk4,marker='o',label='rk4')
plt.xlabel('x')
plt.ylabel('u(x)')
plt.title('Comparison of Exact and Numerical Solutions')

plt.legend()
plt.grid(True)

plt.figure()
err_euler,err_improved,err_rk4 = calc_error(u_euler,u_improved,u_rk4,dx_arr_coarse)
plt.plot(dx_arr_coarse,err_euler,marker='o',label='Euler Cauchy')
plt.plot(dx_arr_coarse,err_improved,marker='o',label='Improved Euler')
plt.plot(dx_arr_coarse,err_rk4,marker='o',label='rk4')
plt.xlabel('x')
plt.ylabel('error norm')
plt.title('Comparison of Error')
plt.grid(True)
plt.legend()
#plt.show()

print(u_improved)