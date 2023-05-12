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

def gaussian(x, rad):
    return np.exp(-(x**2) / (2*(rad**2)))

def geman_mcclure(err):
    return 1/((1+err**2)**2)

def create_bilateral_kernel(mat, m_estimator: int = CHARBONNIER):
    out = np.zeros_like(mat)
    n = mat.shape[0]
    center = mat[int((n-1)/2)][int((n-1)/2)]
    out = mat - center
    if m_estimator == CHARBONNIER:
        out = charbonnier(out)
    elif m_estimator == GEMAN_MCCLURE:
        out = geman_mcclure(out)
    gauss = np.stack([gaussian_kernel(n)]*3, axis=2)
    out = gauss * out
    out = out / np.sum(out)
    return out

    #print(out)
