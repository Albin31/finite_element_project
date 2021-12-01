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
 
# Create mesh and function space
NP =  30; print('Number of mesh points NP = ', NP)
mesh = UnitSquareMesh(NP,NP)
k = 1 ; # k = input('Order of the Lagrange FE ? (1 or 2) '); k = int(k)
V = FunctionSpace(mesh, "CG", int(k)) # Lagrange FE, order k

# Define velocity field
V_vec = VectorFunctionSpace(mesh,"CG", k)
vel_exp = Expression(('100*(1.+abs(cos(2*pi*x[0])))', '100*sin(2*pi/0.2*x[0])'), element = V.ufl_element())
# vel_exp = Expression(('0.', '0.'), element = V.ufl_element())
vel = interpolate(vel_exp,V_vec)

p=plot(vel,title='The velocity field')
p.set_cmap("rainbow"); plt.colorbar(p); plt.show()

# To transform a vector vec to a fenics object vf
#vf= Function(V); vf.vector().set_local(vec)

# The physical RHS 
#f_exp = Expression('1.', element = V.ufl_element())
fp_exp = Expression('1e+03 * exp( -( abs(x[0]-0.5) + abs(x[1]-0.5) )/0.1)', element = V.ufl_element())
fp = interpolate(fp_exp,V)
#fp = Expression('0.', degree=u.ufl_element().degree())
p=plot(fp,title='f')
p.set_cmap("rainbow"); plt.colorbar(p); plt.show()

##################################################################
#
# Newton - Raphson algorithm: Home-implemented non linear solver :)
#
##################################################################

#
# Initialization: u0 solution of a semi-linearized BVP
#

# Diffusivity coeff. depending on the field u0 
mu0_exp = Expression('10.', element = V.ufl_element())
mu0 = interpolate(mu0_exp,V)

#
# The variational problem

# Trial & Test functions
u = TrialFunction(V); v = TestFunction(V)

# A semi-linearized pb
F0 = mu0 * dot(grad(v),grad(u)) * dx + v * dot(vel, grad(u)) * dx - fp * v * dx
# Add the SUPG stabilisation terms
vnorm = sqrt( dot(vel, vel) )
h = MaxCellEdgeLength(mesh)
delta = h / (2.0*vnorm)
residual = - mu0 * div( grad(u) ) + dot(vel, grad(u)) - fp  # the residual expression
F0 += delta * residual * dot(vel, grad(v)) * dx # the enriched weak formulation

# The bilinear and linear forms
a0 = lhs(F0); L0 = rhs(F0)

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

# assign the Dirichlet b.c.
u_diri0_exp = Expression('0.0', degree=u.ufl_element().degree())
u_diri1_exp = Expression('1.0', degree=u.ufl_element().degree())             
bc = DirichletBC(V, u_diri0_exp, u_bdry_x0)

#
# Neumann bc
#
# Nothing to do since they here homogeneous !

# Solve the initial linear system
u0 = Function(V)
solve(a0 == L0, u0, bc)# , [bc0,bc1])


# Plot the solution
plot(mesh)
p=plot(u0, title='The initial solution u0')
p.set_cmap("rainbow"); plt.colorbar(p); plt.show()

#quit()

#
# Iterations
# 
i_max = 30 # max of iterations
i = 0; error = 1. # current iteration
eps_du = 1e-9 # tolerance on the relative norm 

# The FE unknowns 
du = TrialFunction(V)
un = Function(V)
un = u0
# un = project(Constant(1e3), V)

m=1
# Loop
while (error>eps_du and i<i_max): 
 i+=1 # update the current iteration number
 print("Newton-Raphson iteration #",i," begins...");
 mu = mu0*un**m
 # The derivative of mu wrt u
 dmu = mu0*m*un**(m-1)
# dmu = project(dmu_exp,V)
#a et L de Newton raphson
 # LHS of the linearized variational formulation
 a = inner(dmu*du*grad(un), grad(v))*dx + inner(mu * grad(du), grad(v))*dx + inner(vel, grad(du))*v*dx
 # RHS
 L = -(inner(mu*grad(un), grad(v))*dx+inner(vel, grad(un))*v*dx) + fp*v*dx   # RHS of the linearized eqn: TO BE COMPLETED
 

 # Add the SUPG stabilisation terms
 # the residual expression
# residual_lhs =
 residual_lhs = -  div( mu_n*grad(du) ) + dot(vel, grad(du)) - div( dmu_du_n*du * grad(un))
 residual_rhs = -  div( mu_n*grad(un) ) + dot(vel, grad(un))
# a += 
# L += 
 
 # Create bilinear and linear forms
 #a = lhs(F); L = rhs(F)
 
 # Homogeneous Dirichlet b.c. 
 u_diri0_exp = Expression('0.0', degree=du.ufl_element().degree())
 bc0 = DirichletBC(V, u_diri0_exp, u_bdry_x0)
 
 # Solve
 dun = Function(V)
 solve(a == L, dun, bc0)# , [bc0,bc1])
 un.assign(un+dun) # update the solution

 # relative diff.
 if flag_display:
  dun_np = dun.vector().get_local();   un_np = un.vector().get_local();
  #print(type(dun_np)) #print(dun_np.shape)
  error = np.linalg.norm(dun_np) / np.linalg.norm(un_np)
  print("Newton-Raphson iteration #",i,"; error = ", error)
 # test
 if (i == i_max):
  print("Warning: the algo exits because of the max number of ite ! error = ",error)



###########################
# Fenics non linear solver (blackbox)
##########################

# Compute the solution of the non linear BVP by employing the Newton-Raphson method
#problem = NonlinearVariationalProblem(F, u, bcs, J)
#solver  = NonlinearVariationalSolver(problem)
#solver.solve()

###solve(F == 0, u, bc, solver_parameters={"newton_solver":{"relative_tolerance":1e-6}})


#
# Plots
#
plot(mesh)
p=plot(un, title='The non linear solution')
p.set_cmap("rainbow"); plt.colorbar(p); plt.show()
    
