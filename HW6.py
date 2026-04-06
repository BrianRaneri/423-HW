import numpy as np
import matplotlib.pyplot as plt

def plot(arr,X=None,Y=None,type='resid',fig=None,ax=None,plabel=None):

    if ax is None:
        if fig is None:
            fig, ax = plt.subplots()
        else:
            ax = fig.add_subplot(111)
    else:
        fig = ax.figure

    if type == 'contour':
        contour_obj = ax.contourf(X, Y, arr, levels=20)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_aspect('equal')
        fig.colorbar(contour_obj, ax=ax)

    if type == 'split':

        arr = np.array(arr)
        j_mid = arr.shape[1] // 2
        phi_x_slice = arr[:, j_mid]
        x_vals = np.linspace(0, 1, arr.shape[0])

        ax.plot(x_vals, phi_x_slice, label=plabel)
        ax.set_xlabel('x')
        ax.set_ylabel(f'Phi(y=0.5)')
        ax.grid(True)

    if type == 'resid':
        iter_arr = [x[0] for x in arr]
        res_arr = [x[1] for x in arr]
        log_res_arr = np.log10(res_arr)

        ax.set_xlabel('Iteration')
        ax.set_ylabel('log10(Max Residual)')
        ax.grid(True)

        if plabel is not None:
            ax.plot(iter_arr, log_res_arr,label=plabel)
            ax.legend()
        else:
            ax.plot(iter_arr, log_res_arr)


    return fig
        
def create_grid(nx):
    x_arr = np.linspace(0,1,nx)
    y_arr = np.linspace(0,1,nx)
    X, Y = np.meshgrid(x_arr, y_arr, indexing='ij')
    phi_arr = np.zeros((nx,nx))

    return X,Y,phi_arr


def set_bc(arr):
    for i in range(len(arr[0])):
        arr[i,-1] = 1

    return arr

def exact_solution(nx):
    
    X,Y,phi_arr = create_grid(nx)

    dx = L/(len(phi_arr[0])-1)
    dy = W/(len(phi_arr)-1)
    sol_arr = np.copy(phi_arr)

    for i in range(len(phi_arr[0])):
        x = i*dx
        for j in range(len(phi_arr)):
            y = j*dy
            for n in range(1,51):
                sol_arr[i,j] += (((-1)**(n+1)+1)/n)*np.sin(n*np.pi*x/L)*(np.sinh(n*np.pi*y/L)/np.sinh(n*np.pi*W/L))
            sol_arr[i,j]=sol_arr[i,j]*(2/np.pi)

    return sol_arr,X,Y

def jacobi(nx):
    
    x_jacobi,y_jacobi,phi_arr = create_grid(nx)

    dx = L/(len(phi_arr[0])-1)
    dy = W/(len(phi_arr)-1)
    sol_arr = np.copy(phi_arr)
    sol_arr = set_bc(sol_arr)
    #tol = 1e-5
    tol=4

    Ny, Nx = sol_arr.shape
    dx = L / (Nx - 1)
    dy = W / (Ny - 1)

    updated_sol_arr = np.copy(sol_arr)
    resid_arr = np.zeros_like(sol_arr)
    resid_final = []
    sigma = (dx/dy)
 

    for iter in range(1,20000):
        for i in range(1,Nx-1):
            for j in range(1,Ny-1):
                updated_sol_arr[i,j] = (sigma**2*(sol_arr[i,j+1]+sol_arr[i,j-1])+sol_arr[i+1,j]+sol_arr[i-1,j])/(2*(1+sigma**2))
                
        # Calculate Residuals
        for i in range(1,len(sol_arr[0])-1):
            for j in range(1,len(sol_arr[1])-1):
                resid_arr[i,j] = (sol_arr[i+1,j]-2*sol_arr[i,j]+sol_arr[i-1,j])/(dx**2)+(sol_arr[i,j+1]-2*sol_arr[i,j]+sol_arr[i,j-1])/(dy**2)   

        
        res = np.max(resid_arr)
        resid_final.append((iter, res))
        sol_arr = np.copy(updated_sol_arr)

        if (iter % 250 == 0):
            print(f'iter {iter}: res: {res} ')

        if res<tol:
            print(f'Jacobi Method Converged at {iter} iterations')
            return sol_arr.reshape((nx, nx)),resid_final, x_jacobi, y_jacobi

    return sol_arr.reshape((nx,nx)), resid_final, x_jacobi, y_jacobi


L = 1
W = 1

exact_sol_arr, x50, y50 = exact_solution(50)
exact_fig = plot(exact_sol_arr,x50,y50,'contour')

# Jacobi Method
jacobi_iter_fig,jacobi_iter_ax =  plt.subplots()
jacobi_split_fig,jacobi_split_ax =  plt.subplots()

print('Starting Jacobi Method...')
jacobi_sol_arr_100, jacobi_res_100, x100 , y100 = jacobi(20)
jacobi_sol_arr_50, jacobi_res_50, x50, y50 = jacobi(50)

jacobi_iter_fig = plot(jacobi_res_50,fig=jacobi_iter_fig,ax=jacobi_iter_ax, plabel= 'ix = jx = 50')
jacobi_iter_fig = plot(jacobi_res_100,fig=jacobi_iter_fig,ax=jacobi_iter_ax, plabel= 'ix = jx = 100')

print(jacobi_res_50)
print(exact_sol_arr)

jacobi_split_fig = plot(jacobi_sol_arr_50,X=x50,Y=y50,fig=jacobi_split_fig,ax=jacobi_split_ax, plabel= 'ix = jx = 50',type='split')
#jacobi_split_fig = plot(jacobi_res_100,X=x100,Y=y100,fig=jacobi_split_fig,ax=jacobi_split_ax, plabel= 'ix = jx = 100',type='split')
jacobi_split_fig = plot(exact_sol_arr,X=x50,Y=y50,fig=jacobi_split_fig,ax=jacobi_split_ax, plabel= 'Exact',type='split')


#jacobi_fig = plot(jacobi_sol_arr_50,x50,y50,'contour')
plt.show()