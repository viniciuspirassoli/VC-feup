import numpy as np

CHARBONNIER = 0
GEMAN_MCCLURE = 1

def gaussian_kernel(d, sigma:int = 1):
    """
    Generates a d x d Gaussian kernel with standard deviation sigma.
    """
    k = (d-1)//2
    x, y = np.meshgrid(np.arange(-k, k+1), np.arange(-k, k+1))
    kernel = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    return kernel / np.sum(kernel)

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
