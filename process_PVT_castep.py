#!/usr/bin/env python
"""Fit F(V,T) data to find EOS parameters from Castep"""
import re

import numpy as np

# extract to do the fitting
# and some graphing

vol_re = re.compile(r'Current cell volume =\s+(\d+\.\d+)\s+A\*\*3')
zpe_re = re.compile(r'Zero-point energy =\s+(\d+\.\d+)\s+eV')
tmo_re = re.compile(
   r'(\d+\.\d+)\s+(\d+\.\d+)\s+([+-]?\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)')
ufb_re = re.compile(
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
        match = vol_re.search(line)
        if match: 
            # We need to keep track of the *current* volume
            current_volume = float(match.group(1))
            continue
        match = ufb_re.search(line)
        if match:
            # We need to keep track of the *current* internal energy
            U = float(match.group(1))
            continue
        match = zpe_re.search(line)
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
            match = tmo_re.search(line)
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
       (K), and numpy arrays of volumes (A**3) and the vibrational 
       part of the helmoltz free energy (eV) is returned.
    """
    F = []
    V = []
    for line in data_table:
        if line[3] == T:
            F.append(line[5])
            V.append(line[0])
    F = np.array(F)
    V = np.array(V)
    return V, F

if __name__=='__main__':
    import sys
    data = []
    for file in sys.argv[1:]:
        data = parse_castep_file(file, data)

    F, V = get_VF(data, 3000)
    print F
    print V
    
    
