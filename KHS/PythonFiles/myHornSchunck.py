from scipy.ndimage.filters import convolve as filter2
import numpy as np
from typing import Tuple
from pyoptflow.plots import figure, draw, pause

HSkernel = np.array([[1/12, 1/6, 1/12],
                     [1/6, 0, 1/6],
                     [1/12, 1/6, 1/12]], float)

Xkernel = np.array([[-1, 1],
                    [-1, 1]])*.25

Ykernel = np.array([[-1, -1],
                    [1, 1]])*.25

Tkernel = np.ones((2, 2))*.25

"this function is based on the a function founed on the internet"
def myHornSchunck(image1: np.ndarray, image2: np.ndarray, alpha: float = 0.001, NumOfIter: int = 8,
                verbose: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    im1: image at t=0
    im2: image at t=1
    alpha: regularization constant
    Niter: number of iteration
    """
    im1 = image1.astype(np.float32)
    im2 = image2.astype(np.float32)

    # set up initial velocities
    uInitial = np.zeros([im1.shape[0], im1.shape[1]])
    vInitial = np.zeros([im1.shape[0], im1.shape[1]])

    # Set initial value for the flow vectors
    U = uInitial
    V = vInitial

    # Estimate derivatives
    [fx, fy, ft] = computeDerivatives(im1, im2)

    if verbose:
        from pyoptflow.plots import plotderiv
        plotderiv(fx, fy, ft)

#    print(fx[100,100],fy[100,100],ft[100,100])

        # Iteration to reduce error
    for _ in range(NumOfIter):
        # %% Compute local averages of the flow vectors
        uAvg = filter2(U, HSkernel)
        vAvg = filter2(V, HSkernel)
# %% common part of update step
        der = (fx*uAvg + fy*vAvg + ft) / (alpha**2 + fx**2 + fy**2)
# %% iterative step
        U = uAvg - fx * der
        V = vAvg - fy * der

    return U, V


def computeDerivatives(im1: np.ndarray, im2: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    fx = filter2(im1, Xkernel) + filter2(im2, Xkernel)
    fy = filter2(im1, Ykernel) + filter2(im2, Ykernel)

    # ft = im2 - im1
    ft = filter2(im1, Tkernel) + filter2(im2, -Tkernel)

    return fx, fy, ft

def myCompareGraphs(u, v, Inew, scale: int = 3, quivstep: int = 5, fn = None, UorL = 'upper'):
    """
    makes quiver
    """

    ax = figure().gca()
    ax.imshow(Inew, cmap='gray', origin=UorL)
    # plt.scatter(POI[:,0,1],POI[:,0,0])
    for i in range(0, u.shape[0], quivstep):
        for j in range(0, v.shape[1], quivstep):
            ax.arrow(j, i, v[i, j]*scale, u[i, j]*scale, color='red',
                     head_width=0.5, head_length=1)

        # plt.arrow(POI[:,0,0],POI[:,0,1],0,-5)
    if fn:
        ax.set_title(fn)

    draw()
    pause(0.01)