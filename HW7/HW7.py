import numpy as np
import matplotlib.pyplot as plt
import re
import os

def plot(arr,X=None,Y=None,type='resid',fig=None,ax=None,plabel=None,ptitle = None,save=False,flip=False,label=None):

    if ax is None:
        if fig is None:
            fig, ax = plt.subplots()
        else:
            ax = fig.add_subplot(111)
    else:
        fig = ax.figure

    if type == 'contour':

        # Crop
        crop_x = (X[:,0] >= -1) & (X[:,0] <= 2)
        crop_y = (Y[0,:] >= 0) & (Y[0,:] <= 1)

        arr_crop = arr[np.ix_(crop_x, crop_y)]

        vmin = np.min(arr_crop)
        vmax = np.max(arr_crop)

        levels = np.linspace(vmin, vmax, 50)
        contour_obj = ax.contourf(X, Y, arr, levels=levels, vmin=vmin, vmax=vmax)

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_aspect('equal')
        ax.set_xlim(-1, 2)
        ax.set_ylim(0, 1)

        fig.colorbar(contour_obj, ax=ax)
        ax.set_title(ptitle)

    if type == 'Cp':

        Nx,Ny = arr.shape
        dx = 2 * L / (Nx-1)
        x_cp = []
        cp_arr = []

        for i in range(1, Nx - 1):
            x = -L + i * dx
            
            if 0 <= x <= 1:
                u = (arr[i + 1, 0] - arr[i - 1, 0]) / (2*dx)
                x_cp.append(x)
                cp_arr.append(-2*u)

        ax.plot(x_cp, cp_arr, linewidth=2,label=label,marker='o')
        ax.set_xlabel('x')
        ax.set_ylabel('$C_p$')
        ax.grid(True)

        if flip:
            ax.invert_yaxis()
        ax.set_title(ptitle)
        ax.legend()

        ax.set_ylim(0.6,-1)



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
            filename = re.sub(r'[^\w\-_\. ]', '_', ptitle) + ".png"

        os.chdir(os.path.dirname(os.path.realpath(__file__)))
        fig.savefig(filename, bbox_inches='tight', dpi=300)
        print(f"Saved figure as {filename}")

    return fig

def floating_bc(sol_arr,i,M_inf):
    Nx,Ny = sol_arr.shape
    dx = 2*L / (Nx - 1)
    dy = H / (Ny - 1)
    x = -L + dx*i

    non_linear_term = (sol_arr[i+1,0] - sol_arr[i-1,0]) / (2*dx)
    factor = (1 - M_inf**2) - (gamma + 1) * M_inf**2 * non_linear_term

    if x < 0 or x>1:
        p = 0 
        r =  2/dy**2
        q = -2*factor/dx**2 -2/dy**2
        s = -factor * (sol_arr[i+1,0] + sol_arr[i-1,0])/dx**2
        
    else:
        p = 0
        r = 1/dy**2
        q = -2*factor/dx**2 -1/dy**2
        s = -factor * (sol_arr[i+1,0] + sol_arr[i-1,0])/dx**2 + airfoil_slope(x)/dy

    return p, q, r, s

def compute_residual(sol_arr, i, j, M_inf):

    Nx,Ny = sol_arr.shape
    dx = 2*L/((Nx-1))
    dy = H/((Ny-1))

    non_linear_term = (sol_arr[i+1,j] - sol_arr[i-1,j]) / (2*dx)
    factor = (1 - M_inf**2) - (gamma + 1)*M_inf**2 * non_linear_term

    phi_xx = (sol_arr[i+1,j] - 2*sol_arr[i,j] + sol_arr[i-1,j]) / dx**2
    phi_yy = (sol_arr[i,j+1] - 2*sol_arr[i,j] + sol_arr[i,j-1]) / dy**2

    return factor*phi_xx + phi_yy

def build_tri(sol_arr,i,M_inf):
    Nx,Ny = sol_arr.shape
    dx = 2*L / (Nx - 1)
    dy = H / (Ny - 1)    

    # Include bottom and interior
    n = Ny - 1

    p = np.zeros(n)
    q = np.zeros(n)
    r = np.zeros(n)
    s = np.zeros(n)

    p[0], q[0], r[0], s[0] = floating_bc(sol_arr, i,M_inf)

    for j in range(1, n):

        non_linear_term = (sol_arr[i+1,j] - sol_arr[i-1,j]) / (2*dx)
        factor = (1 - M_inf**2) - (gamma + 1) * M_inf**2 * non_linear_term

        p[j] = 1 / dy**2
        r[j] = 1 / dy**2
        q[j] = -2*(factor / dx**2 + 1 / dy**2)
        s[j] = -factor * (sol_arr[i+1,j] + sol_arr[i-1,j]) / dx**2

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

    resid_arr = np.zeros_like(sol_arr)
    resid_final = []

    for iter in range(1,100000):

        for i in range(1,Nx-1):

            old_col = sol_arr[i, :].copy()
            p, q, r, s = build_tri(sol_arr, i,M_inf)

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
                resid_arr[i,j] =  compute_residual(sol_arr,i,j,M_inf)    

        res = np.max(resid_arr)
        resid_final.append((iter, res))
        print(f'iter {iter}: res: {res} ')

        if res<tol:
            print(f'Converged at {iter} iterations')
            break

        if len(resid_final)>50:
            compare_res = resid_final[-10:]
            if all(x[1] < y[1] for x, y in zip(compare_res, compare_res[1:])):
                print(f'Converged at {iter} iterations (increasing)' )
                break

        

    return sol_arr.reshape((Nx, Ny)),resid_final, x_SLOR, y_SLOR

def airfoil_slope(x):
    x = max(x, 1e-2)
    slope = 0.6*(0.14845*x**-0.5-0.1260-0.7032*x+0.8529*x**2-0.406*x**3)
    return slope

L = 5
H = 5
gamma = 1.4

# SLOR Method
iter_fig,iter_ax =  plt.subplots()
M0_contour_fig,M0_contour_ax =  plt.subplots()
M07_contour_fig,M07_contour_ax =  plt.subplots()
Cp_fig,Cp_ax =  plt.subplots()

M0_sol_arr, M0_res, x , y = SLOR(0.02, omega = 0.9,tol = 1e-4,M_inf = 0)
M07_sol_arr, M07_res, x , y = SLOR(0.02, omega = 0.8,tol = 1e-4,M_inf = 0.7)
plot(M0_sol_arr,X=x,Y=y,type = 'contour', ptitle = 'M = 0.0 Pressure Perturbation Contour Plot',save=True,fig = M0_contour_fig,ax = M0_contour_ax)
plot(M07_sol_arr,X=x,Y=y,type = 'contour', ptitle = 'M = 0.7 Pressure Perturbation Contour Plot',save=True,fig = M07_contour_fig,ax = M07_contour_ax)
plot(M0_sol_arr,type = 'Cp', save=False,fig = Cp_fig,ax = Cp_ax,label = 'M = 0.0')
plot(M07_sol_arr,type = 'Cp', ptitle = 'NACA 0012 - 0deg AoA',save=True,fig = Cp_fig,ax = Cp_ax,flip = True,label = 'M = 0.7')

iter_fig = plot(M0_res,fig=iter_fig,ax=iter_ax, plabel= 'M=0.0')
iter_fig = plot(M07_res,fig=iter_fig,ax=iter_ax, plabel= 'M=0.7',ptitle = 'Residuals',save=True)

plt.show()