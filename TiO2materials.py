import numpy as np
import scipy.interpolate

def actuallyReasonableInterp(xi, yi, xs, ys, zs, method = 'cubic'):
    # make it so we can spline interp without having to deal with the point formatting
    xg, yg = np.meshgrid(xs, ys)
    xxg = np.reshape(xg, -1)
    yyg = np.reshape(yg, -1)
    zzg = np.reshape(zs, -1)
    return scipy.interpolate.griddata((xxg, yyg), zzg, (xi, yi), method = 'cubic')

def TiO2_n():
    # get index of TiO2 at temperature=25C and wavelength=640u
    #sourced from the paper Muhammad Rizwan Saleem et al 2014 IOP Conf. Ser.: Mater. Sci. Eng. 60 012008
    return 2.3705

def oxide2D(temp, wavelength):
    # same for oxide, neglect TOC for now 

    data = np.loadtxt('Devore-o.csv', delimiter=',');
    return scipy.interpolate.interp1d(data[:,0], data[:,1])(wavelength)
def oxide2D_ellipsometry(temp, wavelength):
    # same for oxide, neglect TOC for now

    data = np.loadtxt('TiO2_ellipsometry.txt');
    return scipy.interpolate.interp1d(data[:,0], data[:,1])(wavelength*1000)
