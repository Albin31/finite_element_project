'''
NON linear BVP:
     - div( mu(u) * grad(u) ) + w * grad(u) = f  in domain
                                           u = g  on bdry dirichlet
                         - mu(u) nabla(u).n = 0 on bdry Neumann
    with w: given velocity field; mu: given diffusivity coeff. 
    
Example of basic exact solution in domain=(0,1)^2: 
        u = 'sin(x[0]) + sin(x[1])' corresponds to: 
        f = 'cos(x[0]) + cos(x[1]) + sin(x[0]) + sin(x[1])' and g = 'sin(x[0]) + sin(x[1])'
'''



#
# Main program
#
#from dolfin import *
from fenics import *
from sys import exit
import numpy as np 
import matplotlib.pyplot as plt

#
flag_display = 1
T = 1.0            # final time
dt = 0.01     # number of time steps
num_steps = int(T/dt) 
# Create mesh and function space
NP =  20; print('Number of mesh points NP = ', NP)
mesh = UnitSquareMesh(NP,NP)
k = 1 ; # k = input('Order of the Lagrange FE ? (1 or 2) '); k = int(k)
V = FunctionSpace(mesh, "CG", int(k)) # Lagrange FE, order k

# Define velocity field
V_vec = VectorFunctionSpace(mesh,"CG", k)
vel_exp = Expression(('(1.+abs(cos(2*pi*x[0])))', 'sin(2*pi/0.2*x[0])'), element = V.ufl_element())
# vel_exp = Expression(('0.', '0.'), element = V.ufl_element())
vel = interpolate(vel_exp,V_vec)

p=plot(vel,title='The velocity field')
p.set_cmap("rainbow"); plt.colorbar(p); plt.show()

# The physical RHS 
#f_exp = Expression('1.', element = V.ufl_element())
fp_exp = Expression('1e+03 * exp( -( abs(x[0]-0.5) + abs(x[1]-0.5) )/0.1)', element = V.ufl_element())
fp = interpolate(fp_exp,V)
#fp = Expression('0.', degree=u.ufl_element().degree())
p=plot(fp,title='f')
p.set_cmap("rainbow"); plt.colorbar(p); plt.show()

#
# Initialization: u0 solution of a semi-linearized BVP
#

# Diffusivity coeff. depending on the field u0 
mu0_exp = Expression('0.01', element = V.ufl_element())
mu0 = interpolate(mu0_exp,V)

#
# Boundary conditions

# Dirichlet bc
# The functions below return True for points inside the subdomain and False for the points outside.
# Because of rounding-off errors, we specify |𝑥−1|<𝜖, where 𝜖 is a small number (such as machine precision).
tol_bc = 1e-7
def u_bdry_x0(x, on_boundary):
 return bool(on_boundary and (near(x[0], 0, tol_bc)))
def u_bdry_x1(x, on_boundary):
 return bool(on_boundary and (near(x[0], 1., tol_bc)))

# The variational problem
u0 = Expression('1.', element = V.ufl_element())
u_n = interpolate(u0,V)
# Trial & Test functions
u = TrialFunction(V); v = TestFunction(V)

F = u*v*dx - u_n*v*dx + dt*(mu0*dot(grad(v), grad(u_n))*dx + dot(grad(u_n),vel)*v*dx - fp*v*dx)
# The bilinear and linear forms
a = lhs(F); L = rhs(F)

# assign the Dirichlet b.c.
u_diri0_exp = Expression('0.0', degree=u.ufl_element().degree())
u_diri1_exp = Expression('1.0', degree=u.ufl_element().degree())
bc = DirichletBC(V, u_diri0_exp, u_bdry_x0)

# Time-stepping
u = Function(V)
t = 0
fig = plt.figure()
for n in range(num_steps):
    
    # Update current time
    t += dt
    
    # Compute solution
    solve(a == L, u, bc)
    # Plot solution
    plot(mesh)
    p = plot(u)
    p.set_cmap("rainbow"); plt.colorbar(p);
    plt.draw()  
    plt.pause(0.01)
    fig.clear()
    # Update previous solution
    u_n.assign(u)
    print(n)
