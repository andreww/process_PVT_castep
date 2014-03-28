#!/usr/bin/env python
"""Tools for fitting 3rd order Birch-Murnaghan EOS"""
import numpy as np
import scipy.optimize as spopt


def fit_BM3_EOS(V, F, verbose=False):
    """Fit parameters of a 3rd order BM EOS"""
    popt, pconv = spopt.curve_fit(BM3_EOS_energy, V, F, 
                   p0=[np.mean(V), np.mean(F), 150.0, 4.0])
    V0 = popt[0]
    E0 = popt[1]
    K0 = popt[2]
    Kp0 = popt[3]
    if verbose:
        print "Fitted 3rd order Birch-Murnaghan EOS parameters:"
        print " E0  = {:7g} eV".format(E0)
        print " V0  = {:7g} A**3".format(V0)
        print " K0  = {:7g} eV.A**-3 ( = {:7g} GPa)".format(K0, K0*160.218)
        print " K0' = {:7g}".format(Kp0)
    return V0, E0, K0, Kp0


def BM3_EOS_energy (V, V0, E0, K0, Kp0):
    """Calculate the energy from a 3rd order BM EOS"""

    E = E0 + ((9.0*V0*K0)/16.0) * ( (((V0/V)**(2.0/3.0)-1.0)**3.0)*Kp0 +
             (((V0/V)**(2.0/3.0) - 1.0)**2.0 * (6.0-4.0*(V0/V)**(2.0/3.0))))
    return E

def BM3_EOS_pressure(V, V0, K0, Kp0):
    """Calculate the pressure from a 3rd order BM EOS"""

    P = (3.0*K0/2.0) * ((V0/V)**(7.0/3.0)-(V0/V)**(5.0/3.0)) * \
                      (1.0+(3.0/4.0)*(Kp0-4.0)*((V0/V)**(2.0/3.0)-1))
    return P 

def fit_parameters_quad(Ts, V0s, E0s, K0s, Kp0s, plot=False, filename=None):

    poptv, pconv = spopt.curve_fit(_quint_func, np.array(Ts), 
                      np.array(V0s), p0=[0.0, 0.0, 0.0, 
                      0.0, 0.0, np.mean(V0s)])
    fV0 = lambda t: _quint_func(t, poptv[0], poptv[1], poptv[2],
                                       poptv[3], poptv[4], poptv[5])

    popte, pconv = spopt.curve_fit(_quint_func, np.array(Ts), 
                      np.array(E0s), p0=[0.0, 0.0, 0.0,
                      0.0, 0.0, np.mean(E0s)])
    fE0 = lambda t: _quint_func(t, popte[0], popte[1], popte[2],
                                       popte[3], popte[4], popte[5])

    poptk, pconv = spopt.curve_fit(_quint_func, np.array(Ts), 
                      np.array(K0s), p0=[0.0, 0.0, 0.0,
                      0.0, 0.0, np.mean(K0s)])
    fK0 = lambda t: _quint_func(t, poptk[0], poptk[1], poptk[2],
                                       poptk[3], poptk[4], poptk[5])

    poptkp, pconv = spopt.curve_fit(_quint_func, np.array(Ts), 
                      np.array(Kp0s), p0=[0.0, 0.0, 0.0,
                      0.0, 0.0, np.mean(Kp0s)])
    fKp0 = lambda t: _quint_func(t, poptkp[0], poptkp[1], poptkp[2],
                                        poptkp[3], poptkp[4], poptkp[5])

    if plot:
        import matplotlib
        if filename is not None:
            matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fTs = np.linspace(300, 4000, 100)
        fig = plt.figure()
        ax = fig.add_subplot(221)
        ax.plot(Ts, V0s, 'ko')
        ax.plot(fTs, fV0(fTs), 'k-')
        ax.set_xlabel('T (K)')
        ax.set_ylabel('V0 (A**3)')
        ax = fig.add_subplot(222)
        ax.plot(Ts, K0s, 'ko')
        ax.plot(fTs, fK0(fTs), 'k-')
        ax.set_xlabel('T (K)')
        ax.set_ylabel('K0 (eV.A**-3)')
        ax = fig.add_subplot(223)
        ax.plot(Ts, Kp0s, "ko")
        ax.plot(fTs, fKp0(fTs), 'k-')
        ax.set_xlabel('T (K)')
        ax.set_ylabel("K'0" )
        ax = fig.add_subplot(224)
        ax.plot(Ts, E0s, 'ko')
        ax.plot(fTs, fE0(fTs), 'k-')
        ax.set_xlabel('T (K)')
        ax.set_ylabel("E0 (eV)" )
        if filename is not None:
            plt.savefig(filename)
        else:
            plt.show()

    return fV0, fE0, fK0, fKp0

def _quint_func(x, a, b, c, d, e, f):
    return a*x**5.0 + b*x**4.0 + c*x**3.0 + d*x**2.0 + e*x + f

def BM3_EOS_energy_plot(V, F, V0, E0, K0, Kp0, filename=None, Ts=None):
    import matplotlib
    if filename is not None:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)

    if isinstance(V, np.ndarray):
        ax.scatter(V, F)
        fine_vs = np.linspace(np.min(V), np.max(V), 100)
        fine_fs = BM3_EOS_energy(fine_vs, V0, E0, K0, Kp0)
        ax.plot(fine_vs, fine_fs, 'r-')
    else:
        # Assume we can iteratte on T
        for i in range(len(Ts)):
            fine_vs = np.linspace(np.min(V[i]), np.max(V[i]), 100)
            fine_fs = BM3_EOS_energy(fine_vs, V0[i], E0[i], K0[i], Kp0[i])
            ax.plot(fine_vs, fine_fs, 'k--')
            ax.plot(V[i], F[i], 'o', label='{:5g}'.format(Ts[i]))
        ax.legend(title="Temperature (K)")

    ax.set_xlabel('V (A**3)')
    ax.set_ylabel('F (eV)')
    if filename is not None:
        plt.savefig(filename)
    else:
        plt.show()

    
def BM3_EOS_pressure_plot(Vmin, Vmax, V0, K0, Kp0, 
                                 filename=None, Ts=None):
    import matplotlib
    if filename is not None:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)

    if isinstance(V0, np.ndarray):
        fine_vs = np.linspace(Vmin, Vmax, 100)
        fine_ps = BM3_EOS_pressure(fine_vs, V0, K0, Kp0)
        fine_ps = fine_ps * 160.218
        ax.plot(fine_ps, fine_vs, 'r-')
    else:
        # Assume we can iteratte on T
        for i in range(len(Ts)):
            fine_vs = np.linspace(Vmin, Vmax, 100)
            fine_ps = BM3_EOS_pressure(fine_vs, V0[i], K0[i], Kp0[i])
            # Put in GPa
            fine_ps = fine_ps * 160.218
            ax.plot(fine_ps, fine_vs, '-', label='{:5g}'.format(Ts[i]))
        ax.legend(title="Temperature (K)")

    ax.set_xlabel('P (GPa)')
    ax.set_ylabel('V (A**3)')
    if filename is not None:
        plt.savefig(filename)
    else:
        plt.show()

def get_V(P, T, fV0, fK0, fKp0):

    # Put P into eV/A**3
    P = P / 160.218
    V0 = fV0(T)
    K0 = fK0(T)
    Kp0 = fKp0(T)
    p_err_func = lambda v : BM3_EOS_pressure(v, V0, K0, Kp0) - P
    V = spopt.brentq(p_err_func, 0.1*V0, 2.0*V0)
    return V
