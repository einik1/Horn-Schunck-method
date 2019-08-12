import imageio
from matplotlib.pyplot import show
from argparse import ArgumentParser
from pyoptflow import getimgfiles
from myHornSchunck import myHornSchunck, myCompareGraphs


FILTER = 7

def main():
    p = ArgumentParser(description='Pure Python Horn Schunck Optical Flow')
    p.add_argument('stem', help='path/stem of files to analyze')
    p.add_argument('pat', help='glob pattern of files', default='*.bmp')
    p.add_argument('alpha', type=float, help='regularization constant', default='1.')
    p.add_argument('Niter', type=int, help='number of iter in calc for a pixel', default='8')
    p = p.parse_args()

    horn_schunck(p.stem, p.pat, p.alpha, p.Niter)

    show()

def horn_schunck(stem, pat: str, alpha: float, Niter: int, UorL = 'lower'):
    flist = getimgfiles(stem, pat)

    for i in range(len(flist)-1):
        fn1 = flist[i]
        image1 = imageio.imread(fn1, as_gray=True)

        fn2 = flist[i+1]
        image2 = imageio.imread(fn2, as_gray=True)

        A, B = myHornSchunck(image1[::-1], image2[::-1], alpha, Niter)
        myCompareGraphs(A, B, image2[::-1], 3, 5, fn2.name, UorL)



if __name__ == '__main__':
    main()
