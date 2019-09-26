import numpy as np
from numpy import pi, sin, cos, sqrt
from scipy.integrate import trapz
import scipy.constants as sc

import eccUtils as eu


TSUN = sc.G / sc.c**3 * 1.98855e30
KPC2S = sc.parsec / sc.c * 1e3
MPC2S = sc.parsec / sc.c * 1e6


def get_hA_hB(i, e, u, phi):
	ci = cos(i)
	si = sin(i)
	chi = e * cos(u)
	xi = e* sin(u)
	OTS = sqrt(1 - e*e)
	
	hA = ((-2 * (ci*ci + 1) * OTS * xi * sin(2*phi) + (ci*ci + 1) * (2*e*e
		  - chi*chi + chi -2) * cos(2*phi) + si*si * (1 - chi) * chi)/(1 - chi)**2)
	
	hB = 2*ci * (2 * OTS * xi * cos(2*phi) + (2*e*e - chi*chi + chi -2) * sin(2*phi))/(1 - chi)**2
	
	return(hA, hB)


def calculate_sp_sx(toas, gwdist, mc, q, n0, e0, l0, gamma0, inc, psi, 
				tref, evol, waveform_cal):
	
	toa_sample = np.linspace(np.min(toas), np.max(toas), int((np.max(toas) - np.min(toas))/86400))
	
	m = (((1+q)**2)/q)**(3/5) * mc
	eta = q/(1+q)**2
	e02 = e0*e0
	
	if evol:
		ns, es, ls, gammas = eu.evolve_orbit(toa_sample, mc, q, n0, e0, l0, gamma0, tref)
	else:
		ns = n0
		es = e0
		
		k0 = eu.get_k(n0, e0, m, eta)
		ls = l0 + n0 * (toa_sample - tref)
		gammas = gamma0 + k0 *n0 * (toa_sample - tref)
		
	if waveform_cal == 'Num':
		xs, us, phis = eu.get_x_u_phi(ns, es, ls, gammas, m, eta, evol)
		
		H0 = TSUN * m * eta * xs / (gwdist * MPC2S)
		hA, hB = get_hA_hB(inc, es, us, phis)
		
		hps = H0 * (hA * cos(2*psi) - hB * sin(2*psi))
		hxs = H0 * (hB * cos(2*psi) + hA * sin(2*psi))
		
		Sps = []
		Sxs = []
		
		for t in toas:
			ts = toa_sample[toa_sample < t]
			hp = hps[toa_sample < t]
			hx = hxs[toa_sample < t]
			
			Sp = trapz(hp, ts)
			Sx = trapz(hx,ts)
			
			Sps.append(Sp)
			Sxs.append(Sx)
	
	return(np.array(Sps), np.array(Sxs))
