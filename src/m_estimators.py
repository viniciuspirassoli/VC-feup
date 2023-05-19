import numpy as np

def charbonnier(err, beta = 1):
    return 1/(np.sqrt(1+(err*err)/(beta*beta)))

def gaussian(x, sigma):
    return np.exp(-(x**2) / (2*(sigma**2)))

def geman_mcclure(err):
    return 1/((1+err**2)**2)

def gaussianKernel(n: int, sigma: float):
    kernel1D = np.linspace(-(n//2), n//2, n)
    kernel1D = np.exp(-(kernel1D**2) / (2 * sigma**2))

    kernel2D = np.outer(kernel1D, kernel1D)
    kernel2D /= np.sum(kernel2D)
    
    kernel3D = np.zeros((n, n, 3))
    kernel3D[..., 0] = kernel2D 
    kernel3D[..., 1] = kernel2D 
    kernel3D[..., 2] = kernel2D 

    return kernel3D
