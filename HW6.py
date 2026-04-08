import numpy as np
import matplotlib.pyplot as plt

def plot(arr,X=None,Y=None,type='resid',fig=None,ax=None,plabel=None,ptitle = None):

    if ax is None:
        if fig is None:
            fig, ax = plt.subplots()
        else:
            ax = fig.add_subplot(111)
    else:
        fig = ax.figure

    if type == 'contour':
        contour_obj = ax.contourf(X, Y, arr, levels=20, vmin=0, vmax=1.5)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_aspect('equal')
        fig.colorbar(contour_obj, ax=ax)


        ax.set_title('Exact Solution Contour Plot')

    if type == 'split':

        arr = np.array(arr)
        j_mid = arr.shape[1] // 2
        phi_x_slice = arr[:, j_mid]
        x_vals = np.linspace(0, 1, arr.shape[0])

        ax.plot(x_vals, phi_x_slice, label=plabel)
        ax.set_xlabel('x')
        ax.set_ylabel(f'Phi(y=0.5)')
        if ptitle is None:
            ax.set_title(r'$\phi(x, W/2)$')
        else:
            ax.set_title(ptitle)
        ax.grid(True)
        ax.legend()

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

        if ptitle is None:
            ax.set_title('Test')
        else:
            ax.set_title(ptitle)


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

def gauss(nx):
    x_gauss,y_gauss,phi_arr = create_grid(nx)


    sol_arr = np.copy(phi_arr)
    sol_arr = set_bc(sol_arr)
    tol = 1e-5

    Ny, Nx = sol_arr.shape
    dx = L / (Nx - 1)
    dy = W / (Ny - 1)

    resid_arr = np.zeros_like(sol_arr)
    resid_final = []

    h2=1/(2*(1/dx**2+1/dy**2))
    omega = 1

    for iter in range(1,20000):
        for i in range(1,Nx-1):
            for j in range(1,Ny-1):
                RHS = ((sol_arr[i+1,j]+sol_arr[i-1,j])/dx**2)+((sol_arr[i,j+1]+sol_arr[i,j-1])/dy**2) - ((1-(1/omega))*sol_arr[i,j]/h2)
                phi_new = RHS*h2*omega
                sol_arr[i,j] = phi_new
                
        # Calculate Residuals
        for i in range(1,len(sol_arr[0])-1):
            for j in range(1,len(sol_arr[1])-1):
                resid_arr[i,j] = (sol_arr[i+1,j]-2*sol_arr[i,j]+sol_arr[i-1,j])/(dx**2)+(sol_arr[i,j+1]-2*sol_arr[i,j]+sol_arr[i,j-1])/(dy**2)       

        res = np.max(resid_arr)
        resid_final.append((iter, res))

        if (iter % 50 == 0):
            print(f'iter {iter}: res: {res} ')

        if res<tol:
            print(f'Jacobi Method Converged at {iter} iterations')
            break
        
    return sol_arr.reshape((nx, nx)),resid_final, x_gauss, y_gauss  

def jacobi(nx):
    
    x_jacobi,y_jacobi,phi_arr = create_grid(nx)

    sol_arr = np.copy(phi_arr)
    sol_arr = set_bc(sol_arr)
    tol = 1e-5

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

        if (iter % 50 == 0):
            print(f'iter {iter}: res: {res} ')

        if res<tol:
            print(f'Jacobi Method Converged at {iter} iterations')
            break

    return sol_arr.reshape((nx,nx)), resid_final, x_jacobi, y_jacobi

def SOR(nx,omega):
    x_SOR,y_SOR,phi_arr = create_grid(nx)


    sol_arr = np.copy(phi_arr)
    sol_arr = set_bc(sol_arr)
    tol = 1e-5

    Ny, Nx = sol_arr.shape
    dx = L / (Nx - 1)
    dy = W / (Ny - 1)

    resid_arr = np.zeros_like(sol_arr)
    resid_final = []

    h2=1/(2*(1/dx**2+1/dy**2))

    for iter in range(1,20000):
        for i in range(1,Nx-1):
            for j in range(1,Ny-1):
                RHS = ((sol_arr[i+1,j]+sol_arr[i-1,j])/dx**2)+((sol_arr[i,j+1]+sol_arr[i,j-1])/dy**2) - ((1-(1/omega))*sol_arr[i,j]/h2)
                phi_new = RHS*h2*omega
                sol_arr[i,j] = phi_new
                
        # Calculate Residuals
        for i in range(1,len(sol_arr[0])-1):
            for j in range(1,len(sol_arr[1])-1):
                resid_arr[i,j] = (sol_arr[i+1,j]-2*sol_arr[i,j]+sol_arr[i-1,j])/(dx**2)+(sol_arr[i,j+1]-2*sol_arr[i,j]+sol_arr[i,j-1])/(dy**2)       

        res = np.max(resid_arr)
        resid_final.append((iter, res))

        if (iter % 250 == 0):
            print(f'iter {iter}: res: {res} ')

        if res<tol:
            print(f'Jacobi Method Converged at {iter} iterations')
            break
        
    return sol_arr.reshape((nx, nx)),resid_final, x_SOR, y_SOR

def SLOR(nx,omega):
    
    x_SLOR,y_SLOR,phi_arr = create_grid(nx)

    sol_arr = np.copy(phi_arr)
    sol_arr = set_bc(sol_arr)
    tol = 1e-5    

    Nx, Ny = sol_arr.shape
    dx = L / (Nx - 1)
    dy = W / (Ny - 1)

    resid_arr = np.zeros_like(sol_arr)
    resid_final = []

    qj = -2*(1/dx**2+1/dy**2)
    pj = 1/dy**2
    rj = pj

    sj = np.zeros(Ny)
    r_hat = np.zeros(Ny)
    s_hat = np.zeros(Ny) 


    for iter in range(1,2000):

        for i in range(1,Nx-1):

            # Reset Arrays
            sj[:] = 0
            r_hat[:] = 0
            s_hat[:] = 0
            old_col = sol_arr[i, :].copy()

            # Create Sj array
            for j in range(1,Ny-1):
                sj[j] = (-1/dx**2)*(sol_arr[i+1,j]+sol_arr[i-1,j])

            # Boundary Conditions
            sj[Ny-2] -= rj * sol_arr[(Nx // 2),-1]
            sj[1] -= pj * sol_arr[(Nx // 2),0]
            
            # Forward Sweep
            r_hat[1] = rj / qj
            s_hat[1] = sj[1] / qj
            for j in range(2, Ny-1):
                denom = qj - pj * r_hat[j-1]
                r_hat[j] = rj / denom if j < Ny-1 else 0.0
                s_hat[j] = (sj[j] - pj * s_hat[j-1]) / denom

            # Reverse Sweep
            phi_col = np.zeros(Ny)            
            phi_col[0] = 0
            phi_col[-1] = 1
            phi_col[Ny-2] = s_hat[Ny-2]

            for j in reversed(range(1,Ny-2)):
                phi_col[j] = s_hat[j] - r_hat[j] * phi_col[j+1]

            phi_col_relaxed = (1-omega)*old_col + omega*phi_col
            sol_arr[i,1:-1] = phi_col_relaxed[1:-1]


        
        # Calculate Residuals
        for i in range(1,len(sol_arr[0])-1):
            for j in range(1,len(sol_arr[1])-1):
                resid_arr[i,j] = (sol_arr[i+1,j]-2*sol_arr[i,j]+sol_arr[i-1,j])/(dx**2)+(sol_arr[i,j+1]-2*sol_arr[i,j]+sol_arr[i,j-1])/(dy**2)       

        res = np.max(resid_arr)
        resid_final.append((iter, res))

        if (iter % 250 == 0):
            print(f'iter {iter}: res: {res} ')

        if res<tol:
            print(f'Jacobi Method Converged at {iter} iterations')
            break
        


    return sol_arr.reshape((Nx, Ny)),resid_final, x_SLOR, y_SLOR


L = 1
W = 1

exact_sol_arr, x100, y100 = exact_solution(100)
exact_fig = plot(exact_sol_arr,x100,y100,'contour')

'''
# Jacobi Method
jacobi_iter_fig,jacobi_iter_ax =  plt.subplots()
jacobi_split_fig,jacobi_split_ax =  plt.subplots()

print('Starting Jacobi Method...')
jacobi_sol_arr_100, jacobi_res_100, x100 , y100 = jacobi(100)
jacobi_sol_arr_50, jacobi_res_50, x50, y50 = jacobi(50)


jacobi_iter_fig = plot(jacobi_res_50,fig=jacobi_iter_fig,ax=jacobi_iter_ax, plabel= 'ix = jx = 50')
jacobi_iter_fig = plot(jacobi_res_100,fig=jacobi_iter_fig,ax=jacobi_iter_ax, plabel= 'ix = jx = 100',ptitle = 'Jacobi Method Residuals')

jacobi_split_fig = plot(jacobi_sol_arr_50,X=x50,Y=y50,fig=jacobi_split_fig,ax=jacobi_split_ax, plabel= 'ix = jx = 50',type='split')
jacobi_split_fig = plot(jacobi_sol_arr_100,X=x100,Y=y100,fig=jacobi_split_fig,ax=jacobi_split_ax, plabel= 'ix = jx = 100',type='split')
jacobi_split_fig = plot(exact_sol_arr,X=x50,Y=y50,fig=jacobi_split_fig,ax=jacobi_split_ax,plabel='Exact Solution',type='split',ptitle = r'Jacobi Method $\phi(x, W/2)$')



# Gauss - Seidel Method
gauss_iter_fig,gauss_iter_ax =  plt.subplots()
gauss_split_fig,gauss_split_ax =  plt.subplots()

print('Starting Gauss - Seidel Method...')
gauss_sol_arr_100, gauss_res_100, x100 , y100 = gauss(100)
gauss_sol_arr_50, gauss_res_50, x50, y50 = gauss(50)

gauss_iter_fig = plot(gauss_res_50,fig=gauss_iter_fig,ax=gauss_iter_ax, plabel= 'ix = jx = 50')
gauss_iter_fig = plot(gauss_res_100,fig=gauss_iter_fig,ax=gauss_iter_ax, plabel= 'ix = jx = 100',ptitle = 'Gauss - Seidel Method Residuals')

gauss_split_fig = plot(gauss_sol_arr_50,X=x50,Y=y50,fig=gauss_split_fig,ax=gauss_split_ax, plabel= 'ix = jx = 50',type='split')
gauss_split_fig = plot(gauss_sol_arr_100,X=x100,Y=y100,fig=gauss_split_fig,ax=gauss_split_ax, plabel= 'ix = jx = 100',type='split')
gauss_split_fig = plot(exact_sol_arr,X=x50,Y=y50,fig=gauss_split_fig,ax=gauss_split_ax,plabel='Exact Solution', type='split', ptitle = r'Gauss - Seidel Method $\phi(x, W/2)$')
'''

'''
# SOR Method
SOR_iter_fig,SOR_iter_ax =  plt.subplots()
SOR_split_fig,SOR_split_ax =  plt.subplots()

print('Starting SOR Method...')
SOR_sol_arr_100, SOR_res_100, x100 , y100 = SOR(100, omega = 1.8)
SOR_sol_arr_50, SOR_res_50, x50, y50 = SOR(50, omega = 1.8)

SOR_iter_fig = plot(SOR_res_50,fig=SOR_iter_fig,ax=SOR_iter_ax, plabel= 'ix = jx = 50')
SOR_iter_fig = plot(SOR_res_100,fig=SOR_iter_fig,ax=SOR_iter_ax, plabel= 'ix = jx = 100',ptitle = 'SOR Method Residuals')

SOR_split_fig = plot(SOR_sol_arr_50,X=x50,Y=y50,fig=SOR_split_fig,ax=SOR_split_ax, plabel= 'ix = jx = 50',type='split')
SOR_split_fig = plot(SOR_sol_arr_100,X=x100,Y=y100,fig=SOR_split_fig,ax=SOR_split_ax, plabel= 'ix = jx = 100',type='split')
SOR_split_fig = plot(exact_sol_arr,X=x50,Y=y50,fig=SOR_split_fig,ax=SOR_split_ax,plabel='Exact Solution', type='split',ptitle = r'SOR Method $\phi(x, W/2)$')
'''


# SLOR Method
SLOR_iter_fig,SOR_iter_ax =  plt.subplots()
SLOR_split_fig,SOR_split_ax =  plt.subplots()

print('Starting SLOR Method...')
SLOR_sol_arr_100, SLOR_res_100, x100 , y100 = SLOR(100, omega = 1.8)
SLOR_sol_arr_50, SLOR_res_50, x50, y50 = SLOR(50, omega = 1.8)


SLOR_iter_fig = plot(SLOR_res_50,fig=SLOR_iter_fig,ax=SOR_iter_ax, plabel= 'ix = jx = 50')
SLOR_iter_fig = plot(SLOR_res_100,fig=SLOR_iter_fig,ax=SOR_iter_ax, plabel= 'ix = jx = 100',ptitle = 'SLOR Method Residuals')

SLOR_split_fig = plot(SLOR_sol_arr_50,X=x50,Y=y50,fig=SLOR_split_fig,ax=SOR_split_ax, plabel= 'ix = jx = 50',type='split')
SLOR_split_fig = plot(SLOR_sol_arr_100,X=x100,Y=y100,fig=SLOR_split_fig,ax=SOR_split_ax, plabel= 'ix = jx = 100',type='split')
SLOR_split_fig = plot(exact_sol_arr,X=x50,Y=y50,fig=SLOR_split_fig,ax=SOR_split_ax,plabel='Exact Solution', type='split',ptitle = r'SLOR Method $\phi(x, W/2)$')
SLOR_contor = plot(SLOR_sol_arr_50,x50,y50,'contour')
plt.show()
