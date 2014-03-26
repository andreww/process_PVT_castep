#!/usr/bin/env python
"""Fit F(V,T) data to find EOS parameters from Castep"""
import re

import numpy as np

# For each file - get the final cell volume
# then extract the T, F data
# put everything in a table
# extract to do the fitting
# and some graphing

vol_re = re.compile(r'Current cell volume =\s+(\d+\.\d+)\s+A\*\*3')
zpe_re = re.compile(r'Zero-point energy =\s+(\d+\.\d+)\s+eV')
tmo_re = re.compile(
   r'(\d+\.\d+)\s+(\d+\.\d+)\s+([+-]?\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)')

def parse_castep_file(filename, current_data=[]):

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
                current_data.append((current_volume, zpe, T, E, F, S, Cv))
                continue
            else:
                # Must be at the end of this thermo table
                in_thermo = False
                zpe = None
                continue

    fh.close()
    return current_data

        


if __name__=='__main__':
    import sys
    data = []
    for file in sys.argv[1:]:
        data = parse_castep_file(file, data)

    print data
    
    
