import numpy as np
import matplotlib.pyplot as plt


# Exact Solution
def cd_exact(m0,e_m=0.1):
    beta = (m0**2-1)**0.5
    return (16/3)*(1/beta)*e_m**2

# Boundary Condition
def pert_bc(x,e_m=0.1):
    return 2*e_m*(1-2*x)


if __name__ == "__main__":
    
    em = 0.1
    m0 = np.linspace(1.5,4,6)

    jx= 51
    dy = 2/(jx-1)

    # Calculate Exact Solution
    cd_exact_sol = cd_exact(m0,em)

    cd_num = np.zeros_like(m0)

    for k,mach in enumerate(m0):
        beta=(mach**2-1)**0.5
        dx = beta * dy
        ix = int(round(1/dx))+1
        sigma = dx/(beta*dy)
        
        # Create Grid
        x_arr = np.linspace(0,1,ix)
        y_arr = np.linspace(0,2,jx)
        phi_arr = np.zeros((ix,jx))

        # Set BC
        for i in range(len(x_arr)):
            phi_arr[i,0]=pert_bc(x_arr[i])

        drag=0
        for i in range(1,ix-1):

            # Update
            for j in range(1,jx-1):
                phi_arr[i+1,j] = sigma**2*(phi_arr[i,j+1]-2*phi_arr[i,j]+phi_arr[i,j-1])+2*phi_arr[i,j]-phi_arr[i-1,j]
                
            # Apply BC
            phi_arr[i+1,0] = phi_arr[i+1,1] - dy * pert_bc(x_arr[i],em)

            drag += -4*(phi_arr[i+1,0]-phi_arr[i,0]) * pert_bc(x_arr[i]+dx/2,em)
        
        cd_num[k]= drag
    
    plt.figure()
    plt.plot(m0, cd_exact_sol, marker='o', label='Exact Solution')
    plt.plot(m0, cd_num, marker='o', label='Numerical Solution')
    plt.xlabel('Mach Number')
    plt.ylabel('Cd')
    plt.legend()
    plt.grid(True)
    plt.show()