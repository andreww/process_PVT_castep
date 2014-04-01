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
import bm3_eos as eos

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


if __name__=='__main__':
    import sys
    data = []
    for file in sys.argv[1:]:
        data = parse_castep_file(file, data)

    Ts = [0, 500, 1000, 1500, 2000, 2500, 3000, 3500]
    Vs = []
    Fs = []
    K0s = []
    Kp0s = []
    E0s = []
    V0s = []
    min_V = 1.0E12
    max_V = 0.0
    for T in Ts:
        V, F = get_VF(data, T)
        V0, E0, K0, Kp0 =  eos.fit_BM3_EOS(V, F, verbose=True)
        if np.max(V) > max_V: max_V = np.max(V)
        if np.min(V) < min_V: min_V = np.min(V)
        Vs.append(V)
        Fs.append(F)
        K0s.append(K0)
        Kp0s.append(Kp0)
        E0s.append(E0)
        V0s.append(V0)

    print "Athermal EOS"
    Vstat, Fstat = get_VF(data, 'static')
    V0stat, Estat, Kstat, Kpstat = eos.fit_BM3_EOS(Vstat, Fstat, 
        verbose=True)
    print "0K EOS"
    Vzpe, Fzpe = get_VF(data, 0)
    eos.fit_BM3_EOS(Vzpe, Fzpe, verbose=True)

    #eos.BM3_EOS_energy_plot(Vs, Fs, V0s, E0s, K0s, Kp0s, Ts=Ts)
    #eos.BM3_EOS_pressure_plot(np.floor(min_V), np.ceil(max_V), V0s, 
    #    K0s, Kp0s, Ts=Ts)

    eos.BM3_EOS_twoplots(np.floor(min_V), np.ceil(max_V), 
       Vs, Fs, V0s, E0s, K0s, Kp0s, Ts, filename='EOSfits.eps')

    fV0, fE0, fK0, fKp0 = eos.fit_parameters_quad(Ts, V0s, E0s, K0s, Kp0s,
        plot=True, filename='EOSparams.eps', table='EOSparams.tex')

    print "P (GPa) T (K) V (ang**3)"
    print 0, 0, eos.get_V(0, 0, fV0, fK0, fKp0)
    print 0, 300, eos.get_V(0, 300, fV0, fK0, fKp0)
    print 25, 0, eos.get_V(25, 0, fV0, fK0, fKp0)
    print 25, 2600, eos.get_V(25, 2500, fV0, fK0, fKp0)
    print 25, 3200, eos.get_V(25, 3500, fV0, fK0, fKp0)
    print 60, 0, eos.get_V(60, 0, fV0, fK0, fKp0)
    print 60, 3000, eos.get_V(60, 3000, fV0, fK0, fKp0)
    print 60, 4000, eos.get_V(60, 4000, fV0, fK0, fKp0)
    print "Extrapolating for forsterite"
    print 60, 3500, eos.get_V(60, 3500, fV0, fK0, fKp0) 
    print 60, 4000, eos.get_V(60, 3250, fV0, fK0, fKp0) + ((eos.get_V(60, 3500, fV0, fK0, fKp0)-eos.get_V(60, 3000, fV0, fK0, fKp0))/500.0)*750
    
