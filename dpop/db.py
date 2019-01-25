# public items
__all__ = ['DataBase',
           'LAMDA']

# standard library
import re
from pathlib import Path
from pkgutil import get_data
from urllib.request import urlopen
from collections import OrderedDict

# dependent packages
import yaml
import numpy as np
from astropy import constants
from astropy import units as u

# module constants
CONFIG = yaml.load(get_data('dpop', 'data/config.yaml'))
c = constants.c
h = constants.h
k = constants.k_B


class DataBase(object):
    def __init__(self, molname):
        self.molname = molname

    @staticmethod
    def _skip_lines(f, pattern):
        pattern = re.compile(pattern)

        line = ''
        while not pattern.search(line):
            line = f.readline()

        return f

    def __repr__(self):
        classname = self.__class__.__name__
        return f'{classname}({self.molname})'


class LAMDA(DataBase):
    available = list(CONFIG['lamda_mol'].keys())

    def __init__(self, molname, *, encoding='utf-8'):
        super().__init__(molname)

        mol = CONFIG['lamda_mol'][molname]
        path = Path(mol)

        if not path.exists():
            url = CONFIG['lamda_url'] + '/' + mol
            with urlopen(url) as data, path.open('w') as f:
                f.write(data.read().decode(encoding))

        with path.open('r') as f:
            self.energy_levels = self._get_energy_levels(f)
            self.transitions = self._get_transitions(f)

    def Z(self, T_ex):
        if not isinstance(T_ex, u.Quantity):
            T_ex = T_ex * u.K

        Z = 0.0
        for val in self.energy_levels.values():
            g = val['weight']
            E = h * c * val['energy']
            Z += g * np.exp(-E/(k*T_ex))

        return Z.to(u.one)

    def dZdT(self, T_ex):
        if not isinstance(T_ex, u.Quantity):
            T_ex = T_ex * u.K

        dZdT = 0.0
        for val in self.energy_levels.values():
            g = val['weight']
            E = h * c * val['energy']
            dZdT += g * E/(k*T_ex**2) * np.exp(-E/(k*T_ex))

        return dZdT.to(u.K**-1)

    def _get_energy_levels(self, f):
        energy_levels = OrderedDict()
        self._skip_lines(f, 'ENERGY LEVELS$')

        n_levels = int(f.readline())
        self._skip_lines(f, '^!LEVEL')

        for i in range(n_levels):
            elems  = f.readline().split()
            energy = float(elems[1]) * u.cm**-1
            weight = float(elems[2]) * u.one

            key = elems[3]
            val = {'energy': energy, 'weight': weight}
            energy_levels[key] = val

        return energy_levels

    def _get_transitions(self, f):
        transitions = OrderedDict()
        levels = list(self.energy_levels.keys())
        self._skip_lines(f, 'RADIATIVE TRANSITIONS$')

        n_transitions = int(f.readline())
        self._skip_lines(f, '^!TRANS')

        for i in range(n_transitions):
            elems  = f.readline().split()
            upper  = levels[int(elems[1])-1]
            lower  = levels[int(elems[2])-1]
            A_ul   = float(elems[3]) * u.s**-1
            f_rest = float(elems[4]) * u.GHz
            E_u    = float(elems[5]) * u.K

            key = (upper, lower)
            val = {'A_ul': A_ul, 'f_rest': f_rest, 'E_u': E_u}
            transitions[key] = val

        return transitions
