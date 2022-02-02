"""
============================================================
conv_fit: Calculating deltaE
------------------------------------------------------------

    Determination of the beam width based on 
    the gaussian convolution of the Wannier 
    profile that describes the reference signal 
    (typically Ar+)

    Usage:
    >  python3 conv_fit.py Ar.txt [or reference data]

        - Lucas Cornetta FEV/2022

============================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from scipy.optimize import curve_fit
import sys

"""
=======================> Functions <========================
"""

# Wannier function
def wannier(E,IP,b,c,p):
    return b + np.real(np.heaviside(E-IP,0)*c*(abs(E-IP))**p)

# gaussian function
def gaussian(E,E0,dE):
    return np.exp(-(E-E0)*(E-E0)/(2*dE))

# exponential convolution of the Wannier function
def conv(E,IP,dE,b,c,p):
    if dE == 0:
        return wannier(E,IP,b,c,p)

    if len(E) == 1:
        Egrid = np.linspace(0,35,50000)
        return integrate.simps(gaussian(Egrid,E,dE)*wannier(Egrid,IP,b,c,p),Egrid)

    elif len(E) > 1:
        Egrid = np.linspace(0,35,50000)
        convo = []
        for i in range(len(E)):
            convo.append(integrate.simps(gaussian(Egrid,E[i],dE)*wannier(Egrid,IP,b,c,p),Egrid))
        return convo

# least squares error
def lsqerror(Y1,Y2):
    assert len(Y1) == len(Y2)
    error = 0
    for i in range(len(Y1)):
        error += (Y1[i] - Y2[i])**2
    return error


"""
=========================> Main <==========================
"""

if __name__=="__main__":

    print("Convolution fit for the electron beam width determination\n")
    print("* Reading referece data from %s *"%sys.argv[1])
    # Read reference file
    Eexp, EffYield = [], []
    f = open(sys.argv[1],"r")
    for line in f.readlines():
        Eexp.append(float(line.split()[0]))
        EffYield.append(float(line.split()[1]))
    f.close()

    # Energy grid
    E = np.linspace(14,16.8,1000)

    print("* Fitting data using a regular Wannier function *\n")
    # First fit
    popt, pcov = curve_fit(lambda E,IP,b,c,p: wannier(E,IP,b,c,p),xdata=Eexp, ydata=EffYield, maxfev=1000, p0=[15.5,0,7,2.5])
    IP, b, c, p = popt
    IPerr, berr, cerr, perr = np.sqrt(np.diag(pcov))
    print(""" 
        - Adjusted parameters:

            IP = %f +- %f
            b  = %f +- %f
            c  = %f +- %f
            p  = %f +- %f

    """%(IP,IPerr,b,berr,c,cerr,p,perr))

    print("* Fitting data using the (gaussian) convolution of the Wannier profile *\n")
    # Second fit
    popt2, pcov2 = curve_fit(lambda E,IP,dE,b,c,p: conv(E,IP,dE,b,c,p),xdata=Eexp, ydata=EffYield, maxfev=1000, p0=[14.72529088,3.96933463e-02,0.38840849,5.6209942,2.64259897], bounds=((14,0,0,2,1.30738049),(16.2,0.5,1.0,300,7)))
    IP2,dE,b2,c2,p2 = popt2
    IP2err,dEerr,b2err,c2err,p2err = np.sqrt(np.diag(pcov2))
    print(""" 
        - Adjusted parameters:

            IP     = %f +- %f
            b      = %f +- %f
            c      = %f +- %f
            p      = %f +- %f

            deltaE = %f +- %f

    """%(IP2,IP2err,b2,b2err,c2,c2err,p2,p2err,dE,dEerr))

    print("Beam width: %f +- %f meV\n"%(1000*np.sqrt(dE),500*dEerr/np.sqrt(dE)))

    # Plotting fit
    plt.rcParams['font.family'] = 'serif'

    plt.scatter(Eexp,EffYield,color='tab:red')
    plt.plot(E,wannier(E,IP,b,c,p),color='black')
    plt.plot(E,conv(E,IP2,dE,b2,c2,p2),color='tab:blue')
    plt.ylabel("Ion yield (arb. units)",fontsize=14)
    plt.xlabel("energy (eV)",fontsize=14)
    plt.show()
