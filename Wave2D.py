import numpy as np
import sympy as sp
import scipy.sparse as sparse
import pytest
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import cm

x, y, t = sp.symbols('x,y,t')
L = 1

class Wave2D:

    def create_mesh(self, N, sparse=False):
        """Create 2D mesh and store in self.xij and self.yij"""
        self.N = N
        self.h = 1/N

        self.x = np.linspace(0, L, self.N+1)

        self.xij, self.yij = np.meshgrid(self.x, self.x, indexing='ij')
        return self.xij, self.yij

    def D2(self, N):
        """Return second order differentiation matrix"""
        D = sparse.diags([1, -2, 1], [-1, 0, 1], (N+1, N+1), 'lil')
        D[0] = 0
        D[-1] = 0
        return D

    @property
    def w(self):
        """Return the dispersion coefficient"""
        k_x = self.mx * sp.pi
        k_y = self.my * sp.pi
        w = self.c * sp.sqrt(k_x**2 + k_y**2)
        return w

    def ue(self, mx, my):
        """Return the exact standing wave"""
        self.mx = mx
        self.my = my
        return sp.sin(mx*sp.pi*x)*sp.sin(my*sp.pi*y)*sp.cos(self.w*t)

    def initialize(self, N, mx, my):
        r"""Initialize the solution at $U^{n}$ and $U^{n-1}$

        Parameters
        ----------
        N : int
            The number of uniform intervals in each direction
        mx, my : int
            Parameters for the standing wave
        """
        self.mesh = self.create_mesh(N)
        self.u_exact = self.ue(mx, my)

        D = self.D2(N) / self.h**2

        u0 = sp.lambdify([x, y, t], self.u_exact)(self.xij, self.yij, 0)
        u1 = u0[:] + 0.5*(self.c*self.dt)**2*(D @ u0 + u0 @ D.T)
        return u0, u1

    @property
    def dt(self):
        """Return the time step"""
        return self.cfl * self.h / self.c

    def l2_error(self, u, t0):
        """Return l2-error norm

        Parameters
        ----------
        u : array
            The solution mesh function
        t0 : number
            The time of the comparison
        """
        e_n = u - sp.lambdify([x, y, t], self.u_exact)(self.xij, self.yij, t0)

        return np.sqrt(self.h**2 * np.sum(e_n**2))

    def apply_bcs(self):
        self.Un[0] = 0
        self.Un[-1] = 0
        self.Un[:, -1] = 0
        self.Un[:, 0] = 0

    def __call__(self, N, Nt, cfl=0.5, c=1.0, mx=3, my=3, store_data=-1):
        """Solve the wave equation

        Parameters
        ----------
        N : int
            The number of uniform intervals in each direction
        Nt : int
            Number of time steps
        cfl : number
            The CFL number
        c : number
            The wave speed
        mx, my : int
            Parameters for the standing wave
        store_data : int
            Store the solution every store_data time step
            Note that if store_data is -1 then you should return the l2-error
            instead of data for plotting. This is used in `convergence_rates`.

        Returns
        -------
        If store_data > 0, then return a dictionary with key, value = timestep, solution
        If store_data == -1, then return the two-tuple (h, l2-error)
        """
        self.cfl = cfl
        self.c = c

        Unp1, Un, Unm1 = np.zeros((3, N+1, N+1))
        Unm1[:], Un[:] = self.initialize(N, mx, my)

        D = self.D2(self.N)/self.h**2

        plotdata = {0: Unm1.copy()}
        l2_errors = np.zeros(Nt+1)

        for n in range(1, Nt):
            Unp1[:] = 2*Un - Unm1 + (c*self.dt)**2*(D @ Un + Un @ D.T)

            # Swap solutions
            Unm1[:] = Un
            Un[:] = Unp1

            self.Un = Un
            self.apply_bcs()

            if n % store_data == 0:
                plotdata[n] = Unm1.copy() # Unm1 is now swapped to Un

            l2_errors[n] = self.l2_error(self.Un, (n+1)*self.dt)

        l2_errors[Nt] = self.l2_error(self.Un, Nt*self.dt)

        if store_data > 0:
            return plotdata
        
        else: 
            return (self.h, l2_errors)

    def convergence_rates(self, m=4, cfl=0.1, Nt=10, mx=3, my=3):
        """Compute convergence rates for a range of discretizations

        Parameters
        ----------
        m : int
            The number of discretizations to use
        cfl : number
            The CFL number
        Nt : int
            The number of time steps to take
        mx, my : int
            Parameters for the standing wave

        Returns
        -------
        3-tuple of arrays. The arrays represent:
            0: the orders
            1: the l2-errors
            2: the mesh sizes
        """
        E = []
        h = []
        N0 = 8
        for i in range(m):
            dx, err = self(N0, Nt, cfl=cfl, mx=mx, my=my, store_data=-1)
            E.append(err[-1])
            h.append(dx)
            N0 *= 2
            Nt *= 2

        r = [np.log(E[i-1]/E[i])/np.log(h[i-1]/h[i]) for i in range(1, m, 1)]
        return r, np.array(E), np.array(h)

class Wave2D_Neumann(Wave2D):

    def D2(self, N):
        D = sparse.diags([1, -2, 1], [-1, 0, 1], (self.N + 1, self.N + 1), "lil")

        D[0, 1] = 2
        D[-1, -2]  = 2

        return D

    def ue(self, mx, my):
        self.mx = mx
        self.my = my
        return sp.cos(mx*sp.pi*x)*sp.cos(my*sp.pi*y)*sp.cos(self.w*t)


    def apply_bcs(self):
        pass

def test_convergence_wave2d():
    sol = Wave2D()
    r, E, h = sol.convergence_rates(mx=2, my=3)
    assert abs(r[-1]-2) < 1e-2

def test_convergence_wave2d_neumann():
    solN = Wave2D_Neumann()
    r, E, h = solN.convergence_rates(mx=2, my=3)
    assert abs(r[-1]-2) < 0.05

@pytest.mark.parametrize("m", np.arange(1, 50))
def test_exact_wave2d(m):
    CFL = 1 / np.sqrt(2)
    N = 10; Nt = 10

    sol = Wave2D()
    h, E = sol(N, Nt, mx=m, my=m, cfl=CFL, store_data=-1)
    assert np.allclose(E, np.zeros_like(E), atol=1e-12)

    solN = Wave2D_Neumann()
    hN, EN = solN(N, Nt, mx=m, my=m, cfl=CFL, store_data=-1)
    assert np.allclose(EN, np.zeros_like(EN), atol=1e-12)

def create_gif():
    wave = Wave2D_Neumann()
    data = wave(256, 256, mx=2, my=2, cfl=1/np.sqrt(2), store_data=2)

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    frames = []
    for n, val in data.items():
        frame = ax.plot_surface(wave.xij, wave.yij, val, vmin=-0.5*data[0].max(),
                           vmax=data[0].max(), cmap=cm.hot,
                           linewidth=0, antialiased=False)
        frames.append([frame])

    ani = animation.ArtistAnimation(fig, frames, interval=400, blit=True,
                                repeat_delay=1000)
    ani.save("report/neumann.gif", writer="pillow", fps=10) 

if __name__ == "__main__":
    create_gif()