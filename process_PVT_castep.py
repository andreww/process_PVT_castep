#!/usr/bin/env python
"""Fit F(V,T) data to find EOS parameters from Castep

   A Castep 'thermodynamcs' run produces a table of 
   thermodynamic data from a lattice dynamics calculation.
   If a serise of such calculations are done for stuctures
   optimised at different hydrostatic pressures, this information
   can be used extract the free energy and volume as a function
   of pressure and temperature in the 'statically constrained'
   quasi-harmonic approximation (SC QHA). That is, we assume that
   the phonon freequencies are only a function of the unit cell 
   volume, and are not directly altered by temperature (via 
   the anharmonic nature of atomic vibrations) and, furthermore, 
   that a static cell optimisation gives the cell parameters and
   internal structure appropreate at some higher temperature and
   lower pressure.

   This script provides tools to fit Castep thermodynamics data
   to a set of isothermal EOS and thus find V(P,T). """
import re
import numpy as np
import scipy.optimize as spopt

# Some regular expressions that get use a lot,
# so we compile them when the module is loaded
_vol_re = re.compile(r'Current cell volume =\s+(\d+\.\d+)\s+A\*\*3')
_zpe_re = re.compile(r'Zero-point energy =\s+(\d+\.\d+)\s+eV')
_tmo_re = re.compile(
   r'(\d+\.\d+)\s+(\d+\.\d+)\s+([+-]?\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)')
_ufb_re = re.compile(
   r'Total energy corrected for finite basis set =\s+([+-]?\d+\.\d+)\s+eV')

def parse_castep_file(filename, current_data=[]):
    """Read a Castep output file with thermodynamic data and extract it

       This appends thermodynamic data, in the form of a tuple, for 
       each temperature at which lattice dynamics was used to calculate
       thermodynamic data. The tuple includes (in order) the cell
       volume (A**3), static internal energy including the finite basis
       set correctio (eV), the zero point energy (eV), the temperature 
       (K), vibrational energy (eV), vibriational component of the 
       Helmohotz free energy (eV), the entropy (J/mol/K) and heat 
       capacity (J/mol/K). These tuples are returned in a list
       in the order found in the file. Multiple runs can be joined 
       together in one file and this is handled correctly. If multile 
       files need to be parsed the function can be called repetedly 
       with the output list passed in as the optional current_data
       argument and new data will be appended to this list.
    """

    fh = open(filename, 'r')
    current_volume = None
    in_thermo = False
    skip_lines = 0
    for line in fh:
        if skip_lines > 0:
            skip_lines = skip_lines - 1
            continue
        match = _vol_re.search(line)
        if match: 
            # We need to keep track of the *current* volume
            current_volume = float(match.group(1))
            continue
        match = _ufb_re.search(line)
        if match:
            # We need to keep track of the *current* internal energy
            U = float(match.group(1))
            continue
        match = _zpe_re.search(line)
        if match:
            # A line with the zero point energy must start a
            # thermo block. We need to skip three 
            # lines first though.
            zpe = float(match.group(1))
            in_thermo = True
            skip_lines = 3
            continue
        if in_thermo:
            # This is the bulk of the data in the table
            match = _tmo_re.search(line)
            if match:
                T = float(match.group(1))
                E = float(match.group(2))
                F = float(match.group(3))
                S = float(match.group(4))
                Cv = float(match.group(5))
                current_data.append((current_volume, U, zpe, T, 
                                                       E, F, S, Cv))
                continue
            else:
                # Must be at the end of this thermo table
                in_thermo = False
                zpe = None
                U = None
                current_volume = None
                continue

    fh.close()
    return current_data


def get_VF(data_table, T):
    """Given the data file from parse_castep_file return useful data at T

       The data table is searched for data at the target temperature, T
       (K), and numpy arrays of volumes (A**3) and the Helmholtz free 
       energy, F, (eV) is returned. Note that: 

           F(V,T) = U(V) + F_{vib}(V,T)

       where U(V) (static) potential energy of the system at the chosen
       volume and F_{vib}(V,T) is the vibrational Helmholtz free energy
       given by:

           F_{vib}(V,T) = ZPE(V) + E_{vib}(V,T) - T.S_{vib}(V,T)

       i.e. the sum of the zero point energy, the phonon internal 
       energy and the phonon entropy. This second summation is 
       performed by Castep and F_{vib} is reported in the table of 
       thermodynamic quantaties. 

       If T==0 this function returns U(V)+ZPE(V), which can be used to 
       fit a true zero K EOS. If T=='static' just U(V) is returned, which
       can be used to fit a athermal, or static, EOS. 
    """
    # For static or 0K runs, we can use any T we choose, so use the
    # first one in the table.
    if T=='static':
        mode = 'static'
        T = data_table[0][3]
    elif T==0:
        mode = 'zpe'
        T = data_table[0][3]
    else:
        mode = 'f'
    
    F = []
    V = []
    for line in data_table:
        if line[3] == T:
            if mode == 'static':
                F.append(line[1]) # F(V,0) = U(V)
                V.append(line[0])
            elif mode == 'zpe':
                F.append(line[2] + line[1]) # F(V,0) = U(V)+ZPE(V)
                V.append(line[0])
            else:
                # Move to total helmholtz free energy
                # this is U_0(V) + F_vib(V,T)
                F.append(line[5]+line[1])
                V.append(line[0])
    F = np.array(F)
    V = np.array(V)
    return V, F


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

if __name__=='__main__':
    import sys
    data = []
    for file in sys.argv[1:]:
        data = parse_castep_file(file, data)

    Ts = [0, 300, 500, 1000, 1500, 2000, 2500, 4000]
    Vs = []
    Fs = []
    K0s = []
    Kp0s = []
    E0s = []
    V0s = []
    for T in Ts:
        V, F = get_VF(data, T)
        V0, E0, K0, Kp0 =  fit_BM3_EOS(V, F, verbose=True)
        Vs.append(V)
        Fs.append(F)
        K0s.append(K0)
        Kp0s.append(Kp0)
        E0s.append(E0)
        V0s.append(V0)

    print "Athermal EOS"
    Vstat, Fstat = get_VF(data, 'static')
    fit_BM3_EOS(Vstat, Fstat, verbose=True)
    print "0K EOS"
    Vzpe, Fzpe = get_VF(data, 0)
    fit_BM3_EOS(Vzpe, Fzpe, verbose=True)

    BM3_EOS_energy_plot(Vs, Fs, V0s, E0s, K0s, Kp0s, Ts=Ts)
    BM3_EOS_pressure_plot(60, 80, V0s, K0s, Kp0s, Ts=Ts)

    fV0, fE0, fK0, fKp0 = fit_parameters_quad(Ts, V0s, E0s, K0s, Kp0s,
        plot=True)

    print 0, 300, get_V(0, 300, fV0, fK0, fKp0)
    print 10, 300, get_V(10, 300, fV0, fK0, fKp0)
    print 10, 500, get_V(10, 500, fV0, fK0, fKp0)
    print 20, 500, get_V(20, 500, fV0, fK0, fKp0)
    print 30, 500, get_V(30, 500, fV0, fK0, fKp0)
    print 30, 1500, get_V(30, 1500, fV0, fK0, fKp0)
    print 30, 2000, get_V(30, 2000, fV0, fK0, fKp0)
    
