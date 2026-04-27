import numpy as np
import matplotlib.pyplot as plt
import re

def plot(arr,X=None,Y=None,type='resid',fig=None,ax=None,plabel=None,ptitle = None,save=False):

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

    

    if save:
        if ptitle is None:
            filename = "plot.png"
        else:
            # Replace spaces/special chars with underscores
            filename = re.sub(r'[^\w\-_\. ]', '_', ptitle) + ".png"
        fig.savefig(filename, bbox_inches='tight', dpi=300)
        print(f"Saved figure as {filename}")



    return fig

def set_bc(arr):
    Nx, Ny = arr.shape

    '''
    for i in range(Nx):
        arr[i,-1] = 1
    '''
    
    for j in range(Ny):
        arr[0,j] = 1


    return arr

def create_grid(dx):
    x_arr = np.arange(-L,L+dx,dx)
    y_arr = np.arange(0,H+dx,dx)
    X, Y = np.meshgrid(x_arr, y_arr, indexing='ij')
    phi_arr = np.zeros((len(x_arr),len(y_arr)))
    return X,Y,phi_arr

def SLOR(nx,omega):
    
    x_SLOR,y_SLOR,phi_arr = create_grid(nx)

    sol_arr = np.copy(phi_arr)
    sol_arr = set_bc(sol_arr)
    tol = 1e-0    

    Nx, Ny = sol_arr.shape
    dx = 2*L / (Nx - 1)
    dy = H / (Ny - 1)

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
            sj[Ny-2] -= rj * sol_arr[i, -1]
            sj[1]    -= pj * sol_arr[i, 0]

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
        for i in range(1,Nx-1):
            for j in range(1,Ny-1):
                resid_arr[i,j] = (sol_arr[i+1,j]-2*sol_arr[i,j]+sol_arr[i-1,j])/(dx**2)+(sol_arr[i,j+1]-2*sol_arr[i,j]+sol_arr[i,j-1])/(dy**2)       

        res = np.max(resid_arr)
        resid_final.append((iter, res))
        print(f'iter {iter}: res: {res} ')

        if res<tol:
            print(f'Jacobi Method Converged at {iter} iterations')
            break
        

    return sol_arr.reshape((Nx, Ny)),resid_final, x_SLOR, y_SLOR

def SLORT(nx,omega):
    
    x_SLOR,y_SLOR,phi_arr = create_grid(nx)

    sol_arr = np.copy(phi_arr)
    sol_arr = set_bc(sol_arr)
    tol = 1e-0    

    Nx, Ny = sol_arr.shape
    dx = 2*L / (Nx - 1)
    dy = H / (Ny - 1)

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

            p = 1 / dx**2
            r = 1 / dx**2
            q = -2 * (1/dx**2 + 1/dy**2)


            sj = np.zeros(Ny)
            r_hat = np.zeros(Ny)
            s_hat = np.zeros(Ny)

            old_col = sol_arr[i, :].copy()

            phi_left  = sol_arr[i - 1, :]
            phi_right = sol_arr[i + 1, :]

            phi_bottom = sol_arr[i, 1]
            phi_top    = sol_arr[i, -1]

            for j in range(1, Ny - 1):
                non_linear_term = (sol_arr[i+1, j] - sol_arr[i-1, j]) / (2*dx)
                factor = 1 - M_inf**2 - (gamma + 1) * M_inf**2 * non_linear_term                

                sj[j] = factor * -(phi_right[j] + phi_left[j]) / dx**2

            # Boundary Conditions
            #sj[1]    -= p * phi_bottom
            sj[Ny-2] -= r * phi_top

            # Forward Sweep
            r_hat[1] = r / q
            s_hat[1] = sj[1] / q

            for j in range(2, Ny-1):
                denom = q - p * r_hat[j-1]
                r_hat[j] = r / denom
                s_hat[j] = (sj[j] - p * s_hat[j-1]) / denom

            # Reverse Sweep
            phi_col = np.zeros(Ny)      
            phi_col[0] = phi_col[1]      
            #phi_col[0] = 0
            phi_col[-1] = 1
            phi_col[Ny-2] = s_hat[Ny-2]

            for j in reversed(range(1,Ny-2)):
                phi_col[j] = s_hat[j] - r_hat[j] * phi_col[j+1]

            phi_col_relaxed = (1-omega)*old_col + omega*phi_col
            sol_arr[i,1:-1] = phi_col_relaxed[1:-1]

        # Calculate Residuals
        for i in range(1,Nx-1):
            for j in range(1,Ny-1):
                    x_term = (sol_arr[i+1, j] - 2*sol_arr[i, j] + sol_arr[i-1, j]) / dx**2
                    y_term = (sol_arr[i, j+1] - 2*sol_arr[i, j] + sol_arr[i, j-1]) / dy**2

                    # recompute factor consistently
                    phi_x_c = (sol_arr[i+1, j] - sol_arr[i-1, j]) / (2*dx)
                    factor = (1 - M_inf**2) - (gamma + 1) * M_inf**2 * phi_x_c

                    resid_arr[i, j] = factor * x_term + y_term
                #resid_arr[i,j] = (sol_arr[i+1,j]-2*sol_arr[i,j]+sol_arr[i-1,j])/(dx**2)+(sol_arr[i,j+1]-2*sol_arr[i,j]+sol_arr[i,j-1])/(dy**2)       

        res = np.max(resid_arr)
        resid_final.append((iter, res))
        print(f'iter {iter}: res: {res} ')

        if res<tol:
            print(f'Jacobi Method Converged at {iter} iterations')
            break
        

    return sol_arr.reshape((Nx, Ny)),resid_final, x_SLOR, y_SLOR

def airfoil_slope(x):
    return 0.6*(0.14845*x**-0.5-0.1260-0.7032*x+0.8529*x**2-0.406*x**3)


L = 5
H = 5
#M_inf = 0.1375
M_inf = 0.0
gamma = 1.4

# SLOR Method
SLOR_iter_fig,SLOR_iter_ax =  plt.subplots()
SLOR_split_fig,SLOR_split_ax =  plt.subplots()
SLORT_split_fig,SLORT_split_ax =  plt.subplots()

x_SLOR,y_SLOR,phi_arr = create_grid(0.025)

print('Starting SLOR Method...')
SLOR_sol_arr_100, SLOR_res_100, x100 , y100 = SLOR(0.025, omega = 1.7)
SLORT_sol_arr_100, SLOR_res_100, x100 , y100 = SLORT(0.025, omega = 1.7)

#SLOR_iter_fig = plot(SLOR_res_100,fig=SLOR_iter_fig,ax=SLOR_iter_ax, plabel= 'ix = jx = 100',ptitle = 'SLOR Method Residuals',save=False)

#SLOR_split_fig = plot(SLOR_sol_arr_100,X=x100,Y=y100,fig=SLOR_split_fig,ax=SOR_split_ax, plabel= 'ix = jx = 100',type='split')
SLOR_split_fig = plot(SLOR_sol_arr_100,X=x100,Y=y100,type = 'contour', ptitle = 'SLOR',save=True,fig = SLOR_split_fig,ax = SLOR_split_ax)
SLORT_split_fig = plot(SLORT_sol_arr_100,X=x100,Y=y100,type = 'contour', ptitle = 'SLORT',save=True,fig = SLORT_split_fig,ax = SLORT_split_ax)

plt.show()