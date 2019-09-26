import numpy as np
from numpy import pi, sin, cos, sqrt
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.integrate import odeint
from scipy.special import hyp2f1

import mikkola_array

TSUN = 4.925675695249395e-06	#[s]


def get_dtaude(tau,e):
	dtaude = (e**(29./19) * (121*e*e + 304)**(1181./2299))/(1 - e*e)**(3./2)
	return dtaude



try:
	data_tau_e = np.loadtxt("tau_e.txt")
	data_read = True
except:
	print("Warning!!! No input file. Calculating tau(e)!!!!")
	data_read = False

if data_read:
	tau = data_tau_e[:,0]
	e = data_tau_e[:,1]
	
	fun_e_tau = interp1d(tau,e)
	fun_tau_e = interp1d(e, tau)
	
else:
	es = np.linspace(0,0.99, 100)
	tau0 = 0
	taus = odeint(get_dtaude, tau0, es)
	taus_arr = np.reshape(taus,100)
	tau_e = np.vstack((taus_arr, es)).T
	np.savetxt('tau_e.txt', tau_e, fmt='%1.6e')
	
	fun_e_tau = interp1d(taus_arr,es)
	fun_tau_e = interp1d(es, taus_arr)
	
	#plt.plot(taus,es)
	#plt.show()


def e_from_tau(tau):
	return(fun_e_tau(tau))


def tau_from_e(e):
	return(fun_tau_e(e))

#print(e_from_tau([10,10,30,100]))


def n_from_e(e, n0, e0):
	e2 = e*e
	e02 = e0*e0
	n = ( n0 * (e0/e)**(18./19) * ((1 - e2)/(1 - e02))**1.5 
	     * ((304 + 121*e02)/(304 + 121*e2))**(1305./2299) )
	return(n)


def compute_alpha_coeff(A, n0, e0):
	e02 = e0*e0
	alpha = ( (3./A) * (1 - e02)**(5./2)/(n0**(5./3) * e0**(30./19) 
	         * (304 + 121*e02)**(2175./2299)) )
	return(alpha)

def compute_beta_coeff(A, m, n0, e0):
	e02 = e0*e0
	beta = ( (9./A) * (TSUN * m)**(2./3) * (1 - e02)**(3./2)/(n0 * e0**(18./19)
			* (304 + 121*e02)**(1305./2299)) )
	return(beta)

def lbar_from_e(e):
	coeff_l = (19**(2175./2299))/(30 * 2**(496./2299))
	lbar = coeff_l * e**(30./19) * hyp2f1(124./2299, 15./19, 34./19, -121.*e*e/304)
	return(lbar)


def gbar_from_e(e):
	coeff_g = (19**(1305./2299))/(36 * 2**(1677./2299))
	gbar = coeff_g * e**(18./19) * hyp2f1(994./2299,  9./19, 28./19, -121.*e*e/304)
	return(gbar)


def evolve_orbit(ts, mc, q, n0, e0, l0, gamma0, tref = 0):
	m = (((1+q)**2)/q)**(3/5) * mc
	eta = q/(1+q)**2
	e02 = e0*e0
	
	A = (TSUN * mc)**(5./3)/5
	P = (A/3.)* n0**(8./3)*e0**(48./19)*(304 + 121*e02)**(3480./2299)/(1 - e02)**4
	
	tau0 = tau_from_e(e0)
	
	taus = tau0 - P*(ts - tref)
	es = e_from_tau(taus)
	ns = n_from_e(es, n0, e0)
	
	alpha = compute_alpha_coeff(A, n0, e0)
	lbar0 = lbar_from_e(e0)
	lbars = lbar_from_e(es)
	ls = l0 + (lbar0 - lbars)*alpha
	
	beta = compute_beta_coeff(A, m, n0, e0)
	gbar0 = gbar_from_e(e0)
	gbars = gbar_from_e(es)
	gammas = gamma0 + (gbar0 - gbars)*beta
	
	return(ns, es, ls, gammas)







def get_k(n, e, m, eta):
	xi = (TSUN * m * n)**(2./3)
	#OTS = sqrt(1 - e*e)
	OTS2 = 1 - e*e
	k = 3*xi/OTS2 + ((78 + e*e*(51 - 26*eta) - 28*eta)* xi*xi)/(4.*OTS2*OTS2)
	return(k)


def get_PN_x(n, m, k):
	x = (TSUN * m * n * (1 + k))**(2./3)
	return(x)


def get_x_ephi(n, e, m, eta, k):
	x = (TSUN * m * n * (1 + k))**(2./3)
	OTS = sqrt(1 - e*e)
	
	ep = e*(1 + x*(4 - eta) + + x*x *(4*(-12*(26 + 15*OTS) 
		+ eta*(17 + 72*OTS + eta)) + e*e *(1152 + eta*(-659 + 41*eta)))/(96*(-1 + e*e)))
	return(x, ep)


def get_x_u_phi(n, e, l, gamma, m, eta, evol):
	u = mikkola_array.get_u(l,e)
	L = l + gamma
	
	su = sin(u)
	cu = cos(u)
	
	k = get_k(n, e, m, eta)
	x, ephi = get_x_ephi(n, e, m, eta, k)
	
	if evol:
		e_p = ephi[-1]
	else:
		e_p = ephi
	
	if e_p>1.e-15:
		betaphi = (1 - sqrt(1 - ephi**2))/ephi
	else:
		betaphi = e/2. + e**3/8. + e**5/16
	
	v_u = 2 * np.arctan2(betaphi*su, 1 - betaphi*cu)
	v_l = v_u + e*su
	
	W = (1 + k) * v_l
	phi = L + W
	
	return(x, u, phi)
