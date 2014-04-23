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

def fit_parameters_quad(Ts, V0s, E0s, K0s, Kp0s, 
        plot=False, filename=None, table=None):

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

    if table is not None:
        # Write out (LaTeX) table of EOS fitting functions
        fh = open(table, 'w')
        fh.write('$F_0(T) = '+ _f_2_latex(popte[0],noplus=True) +'T^5'
                             + _f_2_latex(popte[1]) +'T^4'
                             + _f_2_latex(popte[2]) +'T^3'
                             + _f_2_latex(popte[3]) +'T^2'
                             + _f_2_latex(popte[4]) +'T'
                             + _f_2_latex(popte[5]) +'$\n')
        fh.write('$V_0(T) = '+ _f_2_latex(poptv[0],noplus=True) +'T^5'
                             + _f_2_latex(poptv[1]) +'T^4'
                             + _f_2_latex(poptv[2]) +'T^3'
                             + _f_2_latex(poptv[3]) +'T^2'
                             + _f_2_latex(poptv[4]) +'T'
                             + _f_2_latex(poptv[5]) +'$\n')
        fh.write('$K_0(T) = '+ _f_2_latex(poptk[0]*160.218,noplus=True) +'T^5'
                             + _f_2_latex(poptk[1]*160.218) +'T^4'
                             + _f_2_latex(poptk[2]*160.218) +'T^3'
                             + _f_2_latex(poptk[3]*160.218) +'T^2'
                             + _f_2_latex(poptk[4]*160.218) +'T'
                             + _f_2_latex(poptk[5]*160.218) +'$\n')
        fh.write('$K^{\prime}_0(T) = '+ _f_2_latex(poptkp[0],noplus=True) +'T^5'
                             + _f_2_latex(poptkp[1]) +'T^4'
                             + _f_2_latex(poptkp[2]) +'T^3'
                             + _f_2_latex(poptkp[3]) +'T^2'
                             + _f_2_latex(poptkp[4]) +'T'
                             + _f_2_latex(poptkp[5]) +'$\n')
        fh.close

    if plot:
        import matplotlib
        if filename is not None:
            matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fTs = np.linspace(0, np.max(Ts), 100)
        if filename is not None:
            fig = plt.figure(figsize=(14.0,14.0), dpi=150)
            #fig.subplots_adjust(left=0.2, right=0.8, top=0.8, bottom=0.2)
        else:
            fig = plt.figure()
        ax = fig.add_subplot(221)
        ax.plot(Ts, V0s, 'ko')
        ax.plot(fTs, fV0(fTs), 'k-')
        ax.set_xlabel('Temperature (K)')
        ax.set_ylabel('V$_0$ (A$^3$)')
        ax = fig.add_subplot(222)
        ax.plot(Ts, np.array(K0s)*160.218, 'ko')
        ax.plot(fTs, fK0(fTs)*160.218, 'k-')
        ax.set_xlabel('T (K)')
        ax.set_ylabel('K$_0$ (GPa)')
        ax = fig.add_subplot(223)
        ax.plot(Ts, Kp0s, "ko")
        ax.plot(fTs, fKp0(fTs), 'k-')
        ax.set_xlabel('Temperature (K)')
        ax.set_ylabel(r'K$^{\prime}$$_0$' ) # Ugly hack to work around 
                                             # matplotlib LaTeX bug.
        ax = fig.add_subplot(224)
        ax.plot(Ts, E0s, 'ko')
        ax.plot(fTs, fE0(fTs), 'k-')
        ax.set_xlabel('Temperature (K)')
        ax.set_ylabel("F$_0$ (eV)" )
        if filename is not None:
            plt.savefig(filename)
        else:
            plt.show()

    return fV0, fE0, fK0, fKp0


def _quint_func(x, a, b, c, d, e, f):
    return a*x**5.0 + b*x**4.0 + c*x**3.0 + d*x**2.0 + e*x + f


def _f_2_latex(value, prec=2, mathmode=True, noplus=False):
    if noplus:
        fmt = '{:.'+str(prec)+'e}'
    else:
        fmt = '{:+.'+str(prec)+'e}'
    basestr = fmt.format(value).split('e')[0]
    expnstr = fmt.format(value).split('e')[1].lstrip('+').lstrip('0').replace('-0', '-', 1)
    latex = basestr+r'\times 10^{'+expnstr+'}'
    if not mathmode:
        latex = '$'+latex+'$'
    return latex


def BM3_EOS_energy_plot(V, F, V0, E0, K0, Kp0, filename=None, Ts=None,
        staticV=None, staticF=None, staticV0=None, staticE0=None,
        staticK0=None, staticKp0=None, ax=None):
    import matplotlib
    if filename is not None:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    doplot = False
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        doplot = True

    if isinstance(V, np.ndarray):
        ax.scatter(V, F)
        fine_vs = np.linspace(np.min(V), np.max(V), 100)
        fine_fs = BM3_EOS_energy(fine_vs, V0, E0, K0, Kp0)
        ax.plot(fine_vs, fine_fs, 'r-')
    else:
        # Assume we can iteratte on T
        cmap=matplotlib.cm.ScalarMappable(cmap='hot')
        cmap.set_clim(vmin=0, vmax=max(Ts)*1.5)
        for i in range(len(Ts)):
            fine_vs = np.linspace(np.min(V[i]), np.max(V[i]), 100)
            fine_fs = BM3_EOS_energy(fine_vs, V0[i], E0[i], K0[i], Kp0[i])
            c = cmap.to_rgba(Ts[i])
            ax.plot(fine_vs, fine_fs, '--', color=c)
            ax.plot(V[i], F[i], 'o', color=c, label='{:5g} K'.format(Ts[i]))
        if staticV is not None:
            # Add the static line
            fine_vs = np.linspace(np.min(staticV[i]), 
                np.max(staticV[i]), 100)
            fine_fs = BM3_EOS_energy(fine_vs, staticV0, staticE0, 
                staticK0, staticKp0)
            ax.plot(fine_vs, fine_fs, '-k', color=c)
            ax.plot(staticV, staticF, 'sk', label='static')
        ax.legend(ncol=3, bbox_to_anchor=(0.  , 0.96, 1., .102), loc=3,
                   mode="expand", borderaxespad=0., numpoints=1)

    ax.set_xlabel('Volume (A$^3$)')
    ax.set_ylabel('Helmholtz free energy (eV)')
    if doplot:
        if filename is not None:
            plt.savefig(filename)
        else:
            plt.show()

    
def BM3_EOS_pressure_plot(Vmin, Vmax, V0, K0, Kp0, ax=None,
                                 filename=None, Ts=None, leg=True):
    import matplotlib
    if filename is not None:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    doplot = False
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        doplot = True

    if isinstance(V0, np.ndarray):
        fine_vs = np.linspace(Vmin, Vmax, 100)
        fine_ps = BM3_EOS_pressure(fine_vs, V0, K0, Kp0)
        fine_ps = fine_ps * 160.218
        ax.plot(fine_ps, fine_vs, 'r-')
    else:
        # Assume we can iteratte on T
        cmap=matplotlib.cm.ScalarMappable(cmap='hot')
        cmap.set_clim(vmin=0, vmax=max(Ts)*1.5)
        for i in range(len(Ts)):
            fine_vs = np.linspace(Vmin, Vmax, 100)
            fine_ps = BM3_EOS_pressure(fine_vs, V0[i], K0[i], Kp0[i])
            # Put in GPa
            fine_ps = fine_ps * 160.218
            c = cmap.to_rgba(Ts[i])
            ax.plot(fine_ps, fine_vs, '-', color=c, 
                label='{:5g} K'.format(Ts[i]))
        if leg: ax.legend()

    ax.set_xlabel('Pressure (GPa)')
    ax.set_ylabel('Volume (A$^3$)')
    if doplot:
        if filename is not None:
            plt.savefig(filename)
        else:
            plt.show()

def BM3_EOS_twoplots(minV, maxV, Vs, Fs, V0s, E0s, K0s, 
        Kp0s, Ts, filename=None):
    import matplotlib
    if filename is not None:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(5.83,8.27), dpi=150)
    fig.subplots_adjust(left=0.2, right=0.9, top=0.9, bottom=0.1)
    ax1 = fig.add_subplot(211)
    BM3_EOS_energy_plot(Vs, Fs, V0s, E0s, K0s, Kp0s, Ts=Ts, ax=ax1)
    ax2 = fig.add_subplot(212)
    BM3_EOS_pressure_plot(minV, maxV, V0s, K0s, 
        Kp0s, Ts=Ts, ax=ax2, leg=False)
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

