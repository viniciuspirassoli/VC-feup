import numpy as np

def charbonnier(err, beta = 1):
    return 1/(np.sqrt(1+(err*err)/(beta*beta)))

def gaussian(x, rad):
    return np.exp(-(x**2) / (2*(rad**2)))

def geman_mcclure(err):
    return 1/((1+err**2)**2)