from collections import OrderedDict
from tkinter import filedialog
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import shapely
import shapely.affinity
from scipy.constants import epsilon_0, speed_of_light
import scipy.interpolate
from shapely.ops import clip_by_rect
from skfem import Basis, ElementTriP0
from skfem.io.meshio import from_meshio

from femwell.maxwell.waveguide import compute_modes
from femwell.mesh import mesh_from_OrderedDict
from femwell.visualization import plot_domains
from typing import Callable

import TiO2materials
import materials


def runTempSweep(thickness, width, temps, lambdas, simBuffer,
                       coreMat, cladMat, coreRes = 0.03, cladRes = 1):
    print("--- TEMP SWEEP ---")
    # run simulation to get neff versus width data (versus wavelength) for 1 wg
    # units can be whatever as long as they are consistent
    
    # the "Mat" variables can be a single index value or a function which takes the wavelength and returns index
    
    # precompute index values
    coreN = np.array([[coreMat(t,l) if isinstance(coreMat, Callable) else coreMat for l in lambdas] for t in temps])
    cladN = np.array([[cladMat(t,l) if isinstance(cladMat, Callable) else cladMat for l in lambdas] for t in temps])
    
    # result array
    sweepResult = {"TE": np.zeros((len(temps), len(lambdas))), 
                   "TM": np.zeros((len(temps), len(lambdas)))}
    
    # only need to mesh once
    core = shapely.geometry.box(-width / 2, 0, width / 2, thickness)
    env = shapely.affinity.scale(core.buffer(simBuffer, resolution=8), xfact=1)
    polygons = OrderedDict(
        core=core,
        box=clip_by_rect(env, -np.inf, -np.inf, np.inf, 0), # left over from what I copied this from, simplify later
        side=clip_by_rect(env, -np.inf, 0, np.inf, thickness),
        top=clip_by_rect(env, -np.inf, thickness, np.inf, np.inf),
    )
    resolutions = dict(core={"resolution": coreRes, "distance": 0.5})
    mesh = from_meshio(mesh_from_OrderedDict(polygons, resolutions, default_resolution_max=10))
    #mesh.draw().show()
    
    for thisTemp in enumerate(temps):
        print("Temp= "+str(thisTemp[1]))
        for thisLambda in enumerate(lambdas):
            print("Lambda= "+str(thisLambda[1])+f", n_SiO2 = {cladN[thisTemp[0],thisLambda[0]]}, n_TiO2 = {coreN[thisTemp[0],thisLambda[0]]}")
            # only differences each time are eps values, and lambda in sim
            basis0 = Basis(mesh, ElementTriP0())
            epsilon = basis0.zeros()
            indexDict = {"core": (coreN[thisTemp[0], thisLambda[0]])-0.3, 
                         "box": cladN[thisTemp[0],thisLambda[0]], 
                         "side": cladN[thisTemp[0],thisLambda[0]],
                         "top": cladN[thisTemp[0],thisLambda[0]]}
            for subdomain, n in indexDict.items():
                epsilon[basis0.get_dofs(elements=subdomain)] = n**2
            #if(thisTemp[0] == 0 and thisLambda[0] == 0):
            #    basis0.plot(epsilon, colorbar=True).show()
            modes = compute_modes(basis0, epsilon, wavelength=thisLambda[1], num_modes=6, order=2)
            """
            for mode in modes:
                print(f"(t = {thisTemp[1]}, l = {thisLambda[1]}) -> neff = {np.real(mode.n_eff):.6f}, TE = {mode.te_fraction}, TM = {mode.tm_fraction}")
                mode.show(mode.E.real, colorbar=True, direction="x") 
            """
            """
            fig, ax = plt.subplots()
            modes[0].plot_intensity(ax=ax)
            plt.title("Normalized Intensity")
            plt.tight_layout()
            plt.show()
            """
            # modes are sorted by decreasing neff by default
            # pick the highest index TE mode and highest index TM mode
            # polarization criterion is > 90% TE or TM
            
            fig, axis = plt.subplots(2,len(modes))
            polCutoff = 0
            index=0
            # first loop through until we find TE mode we want
            for mode in modes:
                if (mode.n_eff>cladN[thisTemp[0],thisLambda[0]] and mode.n_eff<((coreN[thisTemp[0],thisLambda[0]])-0.3)):
                    valid=1;
                else:
                    valid=0;
               #TE section
                if mode.te_fraction > mode.tm_fraction: #polCutoff:
                    #print(str(ax[index]))
                    #print(str(ax1))
                    sweepResult["TE"][thisTemp[0], thisLambda[0]] = np.real(mode.n_eff)
                    mode.plot_intensity(axis[0][index])
                    axis[0][index].set_title(f"TE{index}")
                    print(f"TE{index}, (t = {thisTemp[1]}, l = {thisLambda[1]}) -> neff = {np.real(mode.n_eff):.6f}, valid = {valid}, TE = {mode.te_fraction}, TM = {mode.tm_fraction}")
                #TM section
                else: #mode.tm_fraction > polCutoff:
                    sweepResult["TM"][thisTemp[0], thisLambda[0]] = np.real(mode.n_eff)
                    mode.plot_intensity(axis[1][index])
                    axis[1][index].set_title(f"TM{index}")
                    print(f"TM{index}: (t = {thisTemp[1]}, l = {thisLambda[1]}) -> neff = {np.real(mode.n_eff):.6f}, valid = {valid}, TM = {mode.tm_fraction}, TE = {mode.te_fraction}")

                index+=1
                    
            #plt.show()
            plt.close('all')

            
           #print("made it past for loops")
    return sweepResult

if __name__ == "__main__":
    #widths = np.array([0.3])
    widths = np.array([0.21, 0.225, 0.262, 0.300]) # width
    thicknesses=np.array([0.15, 0.125, 0.1, 0.075])
    #thicknesses= np.linspace(0.1,0.2, 5) # waveguide thickness
    lambdas = np.array([0.6])
    #lambdas = np.linspace(0.6,0.7,2)
    # get index of TiO2 at temperature=25C and wavelength=640nm
    #sourced from the paper Muhammad Rizwan Saleem et al 2014 IOP Conf. Ser.: Mater. Sci. Eng. 60 012008
    temps=np.array([300])
    for index in range(4):
        w0=widths[index]
        t0=thicknesses[index]
        simMargin = 10*w0
        coreres = w0/20
        cladres = w0/4
        print()
        print("----------------------------------")
        print(f"width = {w0*1000}nm,   thickness = {t0*1000}nm")
        print()
        sweepResult = runTempSweep(t0, w0, temps, lambdas, simMargin,
                               coreMat = TiO2materials.oxide2D, cladMat = materials.oxide2D,
                               coreRes=coreres, cladRes=cladres)
#%%
'''
prefix = '200nmx50nm_TiO2_just_300K_640nm'
filename = datetime.now().strftime('simData/' + prefix + "_%Y-%m-%d-%H-%M-%S")
print(filename)
np.savez(filename + '.npz',  temps = temps, thisLambda = 1e-6*lambdas, sweepResult = sweepResult)
scipy.io.savemat(filename + '.mat', {'indexTemps': temps, 'indexLambda': 1e-6*lambdas, 'indexSimResult': sweepResult})
'''
