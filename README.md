In pdebase.py base classes of ANN-based solvers for elliptic, parabolic and hyperbolic PDEs with Dirichlet boudnary conditions are defined:
- NNPDE: 2-dimensional Poisson equation;
- NNPDE_transport: 2-dimensional stationary advection-diffusion equation;
- Heat_equation: 2-dimensional heat equation;
- Advection_diffusion_equation: 2-dimensional advection-diffusion equation;
- NNPDE_hyperbolic: 2-dimensional linear transport equation;
- NNPDE_ND: N-dimensional Poisson equation (modify the method self.loss_funcion() for extension to other PDEs).

In problems.py the problem data are specified.
For every type of PDE four problems with different levels of regularity are introduced:
- Smooth: u(x,y)=sin(pi x) sin(pi y);
- Peak: u(x,y)=exp(-1000(x-0.5)^2-1000(x-0.5)^2);
- Singularity: u(x,y)=y^0.6;
- Singularity2: (expressed in polar coordinates) u(rho, theta)=rho^(3/2) sin(3 theta/2).

In training.py an example of application of the method on the 2-dimensional smooth Poisson problem.
Download this repository and, before running training.py, create a foalder called 'p1' in the same path where training.py is saved.
In the foalder 'p1' the plots of the loss functions of the interior and boundary network and the global approximation errors as functions of the iteration counts will be saved, together with the plots of the exact and approximate solutions every 100 training iterations.

The only python libraries needed are TensorFlow and NumPy.

The approximate solution is given as the output of a coupling between two artificial dense feedforward neural networks, one taking approximating the boundary condition and the other approximating the result of the PDE in the interior of the domain. This is based on the work of K. Xu et al., 'Deep learning for Partial Differential Equations (PDEs)', 2018.
