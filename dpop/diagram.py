# public items
__all__ = ['PopDiagram']

# standard library
from collections import OrderedDict

# dependent packages
import dpop
import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from astropy import constants

# module constants
c = constants.c
h = constants.h
k = constants.k_B


class PopDiagram(OrderedDict):
    def __init__(self, molname_or_db):
        if isinstance(molname_or_db, str):
            # add try-except later for multiple DBs
            self.db = dpop.LAMDA(molname_or_db)
        elif isinstance(molname_or_db, dpop.DataBase):
            self.db = molname_or_db
        else:
            raise ValueError(molname_or_db)

    def calc(self):
        pass

    def plot(self):
        pass

    def __setitem__(self, key, val):
        """For key, value checking"""
        super().__setitem__(key, val)

    def __repr__(self):
        return f'PopDiagram({self.db.molname})'


class _PopDiagram(object):
    def __init__(self, db):
        '''Create a population diagram calculator.

        Parameters
        ----------
        db: LAMDA or CDMS instance
            A database of molecule used for population diagram

        Examples
        ----------
        >>> co = LAMDA('co.dat')
        >>> diag = PopDiagram(co)
        '''
        self.db = db
        self.clear()

    def input(self, upper_level, lower_level, I_b, I_b_error):
        '''Input integrated brightness temperature and its error.

        Parameters
        ----------
        upper_level: str
            Upper level of transition.
        lower_level: str
            Lower level of transition.
        I_b: float
            Integrated brightness temperature in units of K*km/s.
        I_b_error: float
            Error of I_b in units of K*km/s.

        Examples
        ----------
        Input CO(3-2) with I_b = 10 +/- 0.1 K*km/s:
        >>> co = LAMDA('co.dat')
        >>> diag = PopDiagram(co)
        >>> diag.input('3', '2', 10.0, 0.1)
        '''
        if type(I_b) != u.quantity.Quantity:
            I_b *= u.K * u.km / u.s
        if type(I_b_error) != u.quantity.Quantity:
            I_b_error *= u.K * u.km / u.s

        d = self.db(upper_level, lower_level)
        N_u = (8*pi*k/(h*c**3) * d['f_rest']**2/d['A_ul'] * I_b).to(u.cm**(-2))
        N_u_error = N_u * I_b_error/I_b

        # calc. x and y of y=a+bx
        x = d['E_u'].value
        y = np.log10((N_u/d['g_u']).value)
        y_error = (N_u_error/N_u/np.log(10)).value
        x_label = '{}({}-{})'.format(self.db.molname, upper_level, lower_level)
        self.x = np.append(self.x, x)
        self.y = np.append(self.y, y)
        self.y_error = np.append(self.y_error, y_error)
        self.x_label = np.append(self.x_label, x_label)

    def clear(self):
        '''Clear all values in the calculator.'''
        self.y = np.array([])
        self.x = np.array([])
        self.x_label = np.array([])
        self.y_error = np.array([])
        self.a, self.a_error = None, None
        self.b, self.b_error = None, None
        self.T_ex, self.T_ex_error = None, None
        self.N_tot, self.N_tot_error = None, None

    def calc(self, log=False):
        '''Calculate T_ex, N_tot and their errors.'''
        w = self.y_error**(-2)
        sum_w = np.sum(w)
        sum_wx = np.sum(w * self.x)
        sum_wy = np.sum(w * self.y)
        sum_wxx = np.sum(w * self.x**2)
        sum_wxy = np.sum(w * self.x * self.y)
        D = sum_w*sum_wxx-sum_wx**2

        # calc. a, b, and their errors via y=a+bx
        a = (sum_wy*sum_wxx - sum_wxy*sum_wx)/D
        b = (sum_w*sum_wxy - sum_wx*sum_wy)/D
        a_error = np.sqrt(sum_wxx/D)
        b_error = np.sqrt(sum_w/D)
        self.a, self.a_error = a, a_error
        self.b, self.b_error = b, b_error

        # calc. T_ex and its error
        T_ex = -b**(-1)/np.log(10)
        T_ex_error = b_error*b**(-2)/np.log(10)
        self.T_ex, self.T_ex_error = T_ex*u.K, T_ex_error*u.K

        # calc. N_tot and its error
        elem_1 = self.db.Z_diff(self.T_ex) * self.T_ex_error
        elem_2 = self.db.Z(self.T_ex) * np.log(10) * self.a_error
        N_tot = self.db.Z(self.T_ex) * 10**a
        N_tot_error = np.sqrt(elem_1**2 + elem_2**2) * 10**a
        self.N_tot, self.N_tot_error = N_tot, N_tot_error

        # print results
        self.T_ex_result = 'T_ex  = {:.1f} +/- {:.1f} (K)'.format(T_ex, T_ex_error)
        self.N_tot_result = 'N_tot = {:.1e} +/- {:.1e} (cm^-2)'.format(N_tot, N_tot_error)
        print('Results of fitting')
        print('--------------------')
        print(self.T_ex_result)
        print(self.N_tot_result)

    def plot(self, savefig=False, extension='pdf'):
        '''Plot and show a population diagram using matplotlib.

        Parameters
        ----------
        savefig: bool
            If True, the resulting plot is saved in local.
        extension: str
            An extension (pdf, png, etc) of the saved figure.
        '''
        text_result = '{}\n{}'.format(self.T_ex_result, self.N_tot_result)
        x = np.arange(0.0, np.max(self.x)+10.0, 0.1)

        # plot data points and resulting line
        plt.plot(x, self.a+self.b*x, label=text_result)
        plt.errorbar(self.x, self.y, yerr=self.y_error, fmt='o')

        # cosmetics
        plt.title('Population Diagram of {}'.format(self.db.molname))
        for i in range(len(self.x)):
            plt.annotate(self.x_label[i], (self.x[i], self.y[i]),
                xytext=(3, 3), textcoords='offset points')
        plt.xlim([0.0, np.max(x)])
        plt.xlabel('Eu (K)')
        plt.ylabel('Log10 (Nu/gu)')
        plt.legend(loc='lower left')
        plt.grid(True)

        # show or save figure
        if savefig:
            print('saving the resulting plot...')
            plt.savefig('popdiagram-{}.{}'.format(self.db.molname, extension))
        else:
            plt.show()

    def __repr__(self):
        return 'PopDiagram({})'.format(self.db.molname)

    def __str__(self):
        return 'PopDiagram({})'.format(self.db.molname)
