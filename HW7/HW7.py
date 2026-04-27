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
        contour_obj = ax.contourf(X, Y, arr, levels=50)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_aspect('equal')
        fig.colorbar(contour_obj, ax=ax)
        ax.set_title(ptitle)

    if type == 'Cp':

        Nx,Ny = arr.shape
        dx = 2 * L / (Nx-1)
        x_cp = []
        cp_arr = []

        for i in range(1, Nx - 1):
            x = -L + i * dx
            
            if 0 < x <= 1:
                u = (arr[i + 1, 0] - arr[i - 1, 0]) / (2 * dx)
                x_cp.append(x)
                cp_arr.append(-2*u)

        ax.plot(x_cp, cp_arr, linewidth=2)
        ax.set_xlabel('x')
        ax.set_ylabel('$C_p$')
        ax.grid(True)
        ax.invert_yaxis()
        ax.set_title(ptitle)




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

def floating_bc(sol_arr,i):
    Nx,Ny = sol_arr.shape
    dx = 2*L / (Nx - 1)
    dy = H / (Ny - 1)
    x = -L + dx*i

    if x <= 0 or x>=1:
        p = 0 
        r =  2/dy**2
        q = -2/dx**2 - 2/dy**2
        s = -(sol_arr[i+1,0] + sol_arr[i-1,0]) / dx**2
    else:
        p = 0
        r = 1/dy**2
        q = -2/dx**2 - 1/dy**2
        s = -(sol_arr[i+1,0] + sol_arr[i-1,0]) / dx**2 + airfoil_slope(x)/dy

    return p, q, r, s

def build_tri(sol_arr,i):
    Nx,Ny = sol_arr.shape
    dx = 2*L / (Nx - 1)
    dy = H / (Ny - 1)    

    # Include bottom + interior
    n = Ny - 1

    p = np.zeros(n)
    q = np.zeros(n)
    r = np.zeros(n)
    s = np.zeros(n)

    p[0], q[0], r[0], s[0] = floating_bc(sol_arr, i)

    for j in range(1, n):

        p[j] = 1/dy**2
        r[j] = 1/dy**2
        q[j] = -2*(1/dx**2 + 1/dy**2)

        s[j] = -(1/dx**2) * (
            sol_arr[i+1, j] + sol_arr[i-1, j]
        )

    # Top BC
    q[-1] += r[-1]
    r[-1] = 0

    return p,q,r,s

def create_grid(dx):
    x_arr = np.arange(-L,L+dx,dx)
    y_arr = np.arange(0,H+dx,dx)
    X, Y = np.meshgrid(x_arr, y_arr, indexing='ij')
    phi_arr = np.zeros((len(x_arr),len(y_arr)))
    return X,Y,phi_arr

def SLOR(nx,omega,tol,M_inf=0):
    
    x_SLOR,y_SLOR,sol_arr = create_grid(nx)

    Nx, Ny = sol_arr.shape
    dx = 2*L / (Nx - 1)
    dy = H / (Ny - 1)

    resid_arr = np.zeros_like(sol_arr)
    resid_final = []

    for iter in range(1,2000):

        for i in range(1,Nx-1):

            old_col = sol_arr[i, :].copy()
            p, q, r, s = build_tri(sol_arr, i)

            n = Ny - 1

            # Forward Sweep
            r_hat = np.zeros(n)
            s_hat = np.zeros(n)
            phi = np.zeros(n)

            r_hat[0] = r[0] / q[0]
            s_hat[0] = s[0] / q[0]

            for j in range(1, n):
                denom = q[j] - p[j] * r_hat[j-1]
                r_hat[j] = r[j]/denom
                s_hat[j] = (s[j] - p[j] * s_hat[j-1]) / denom

            # Reverse Sweep
            phi = np.zeros(n)            

            for j in reversed(range(n)):
                if j == n - 1:
                    phi[j] = s_hat[j]
                else:
                    phi[j] = s_hat[j]-r_hat[j] * phi[j+1]

            sol_arr[i, 0:n] = ((1 - omega) * old_col[0:n] + omega * phi)

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
gamma = 1.4

# SLOR Method
M0_iter_fig,M0_iter_ax =  plt.subplots()
M0_contour_fig,M0_contour_ax =  plt.subplots()
Cp_fig,Cp_ax =  plt.subplots()

print('Starting SLOR Method...')
M0_sol_arr, M0_res, x , y = SLOR(0.02, omega = 1.7,tol = 1e-2,M_inf = 0)
plot(M0_sol_arr,X=x,Y=y,type = 'contour', ptitle = 'M=0.0 Contour Plot',save=False,fig = M0_contour_fig,ax = M0_contour_ax)
plot(M0_sol_arr,type = 'Cp', ptitle = 'M=0.0 Cp Plot',save=False,fig = Cp_fig,ax = Cp_ax)

plt.show()