# public items
__all__ = ['LAMDA']

# standard library
import re
from pathlib import Path
from pkgutil import get_data
from urllib.request import urlopen
from collections import OrderedDict

# dependent packages
import yaml
from astropy import units as u

# module constants
config = yaml.load(get_data('dpop', 'data/config.yaml'))


# classes
class LAMDA(object):
    avalilable = list(config['lamda_mol'].keys())

    def __init__(self, molname, *, encoding='utf-8'):
        mol = config['lamda_mol'][molname]
        url = config['lamda_url'] + f'/{mol}'
        path = Path(mol)

        if not path.exists():
            with urlopen(url) as data, path.open('w') as f:
                f.write(data.read().decode(encoding))

        with path.open('r') as f:
            self.molname = molname
            self.energy_levels = self._get_energy_levels(f)
            self.transitions = self._get_transitions(f)

        self.available = list(self.transitions.keys())

    def _get_energy_levels(self, f):
        kwd = ''
        pat = re.compile('energy levels', re.IGNORECASE)
        while not pat.search(kwd):
            kwd = f.readline()

        n_levels = int(f.readline())
        f.readline()

        energy_levels = OrderedDict()

        for i in range(n_levels):
            elems  = f.readline().split()
            energy = float(elems[1]) / u.cm
            weight = float(elems[2]) / u.dimensionless_unscaled
            level  = elems[3]

            energy_levels[level] = {'energy': energy, 'weight': weight}

        return energy_levels

    def _get_transitions(self, f):
        kwd = ''
        pat = re.compile('radiative transitions', re.IGNORECASE)
        while not pat.search(kwd):
            kwd = f.readline()

        n_transitions = int(f.readline())
        f.readline()

        transitions = OrderedDict()
        levels = list(self.energy_levels.keys())

        for i in range(n_transitions):
            elems  = f.readline().split()
            upper  = levels[int(elems[1])-1]
            lower  = levels[int(elems[2])-1]
            A_ul   = float(elems[3]) / u.s
            f_rest = float(elems[4]) * u.GHz
            E_u    = float(elems[5]) * u.K

            name = f'{upper}-{lower}'
            transitions[name] = {'A_ul': A_ul, 'f_rest': f_rest, 'E_u': E_u}

        return transitions
