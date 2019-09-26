import numpy as np
from numpy import pi, sin, cos
from scipy.integrate import trapz
import scipy.constants as sc
import matplotlib.pyplot as plt
#from decimal import *
import time

import antenna_pattern as ap
import eccUtils as eu
import waveform

#np.set_printoptions(precision=30)
#getcontext().prec = 30

start_time = time.time()

TSUN = sc.G / sc.c**3 * 1.98855e30
KPC2S = sc.parsec / sc.c * 1e3
MPC2S = sc.parsec / sc.c * 1e6


def add_ecc_cgw(toa, psrra, psrdec, psrdist, gwra, gwdec, gwdist,
				mc, q, F0, e0, l0, gamma0, inc, psi, l_P = None, 
				gamma_P = None, tref = 0, psrterm = True, evol = True,
				waveform_cal = 'Num'):
	toas = np.double(toa * 86400)
	n0 = 2 * pi * F0
	m = (((1+q)**2)/q)**(3/5) * mc
	eta = q/(1+q)**2
	
	cosmu, Fp, Fx = ap.antenna_pattern(gwra, gwdec, psrra, psrdec)
	
	Sp, Sx = waveform.calculate_sp_sx(toas, gwdist, mc, q, n0, e0, l0, gamma0, inc, psi, tref, evol, waveform_cal)
	
	if psrterm:
		Dp = psrdist * KPC2S
		dt_P = 700 * 365.25 * 86400 #Dp * (1 - cosmu)
		
		toas_P = toas - dt_P
		tref_P = tref - dt_P
		
		n0_P, e0_P, l0_P, gamma0_P = eu.evolve_orbit(tref_P, mc, q, n0, e0, l0, gamma0, tref)
		
		if l_P is not None:
			l0_P = l_P
		if gamma_P is not None:
			gamma0_P = gamma_P

		Spp, Sxp = waveform.calculate_sp_sx(toas_P, gwdist, mc, q, n0_P, e0_P, l0_P, gamma0_P, inc, psi, tref_P, evol, waveform_cal)
		
		Sp = Spp - Sp
		Sx = Sxp - Sx
	
	#return (Sp, Sx, Spp, Sxp)
	
		
	res = (Fp * Sp) + (Fx * Sx)		#[s]
	toas_new = toa + (res)/86400.		#[day]
	return (res/86400., toas_new)
	


toas = np.linspace(0, 10*365.25, 100, dtype = np.longdouble)
#toa_sample = np.linspace(np.min(toas), np.max(toas), int((np.max(toas) - np.min(toas))/86400))

residual, toas_new = add_ecc_cgw(toas, 1.21, -0.825, .4, 2.33, 0.351, 1.e3, 4352752816.48062, 1, 1/(5*365.25*86400), 0.1, 0, 0, pi/6, 0)

end_time = time.time()
print("Runtime =",end_time - start_time,"sec")

res = toas_new - np.longdouble(toas)
print('%.18f' % toas[-12], '%.18f' %toas_new[-12], '%.18f' %residual[-12], '%.18f' %res[-12])

plt.plot(toas/365.25, res)
plt.plot(toas/365.25, residual)
plt.show()
