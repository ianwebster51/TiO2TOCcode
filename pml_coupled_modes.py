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
from skfem import Basis, ElementTriP1
from skfem.io.meshio import from_meshio

from femwell.maxwell.waveguide import compute_modes
from femwell.maxwell.waveguide import plot_mode
from femwell.mesh import mesh_from_OrderedDict
from femwell.visualization import plot_domains
from typing import Callable
from shapely.geometry import Polygon, MultiPolygon
from shapely.geometry import Point
from shapely.affinity import scale
from shapely import contains_xy
from scipy.ndimage import distance_transform_edt
from scipy.interpolate import RegularGridInterpolator

import TiO2materials
import materials


def runTempSweep(thickness, width, temps, lambdas, simBuffer,
                       coreMat, cladMat, coreRes = 0.03, cladRes = 1, off=0, sep=0.05, pml_thickness=0):
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
    left_core = shapely.geometry.box(-width-sep/2, -thickness/2, -sep/2, thickness/2)
    right_core = shapely.geometry.box(sep/2, -thickness/2, width+sep/2, thickness/2)
    cores=MultiPolygon([left_core, right_core])
    env = Point(0, 0).buffer(simBuffer, resolution=8)
    #env = scale(env, xfact=5, yfact=1)

    # Define a surrounding box for PML
    pml_outer = shapely.affinity.scale(env.buffer(pml_thickness, resolution=8), xfact=1)
    pml_region = pml_outer.difference(env)

    polygons = OrderedDict(
    core=cores,
    clad=env.difference(cores),
    pml=pml_region
    )
    """
        box=clip_by_rect(env, -np.inf, -np.inf, np.inf, -thickness/2), # left over from what I copied this from, simplify later
        side=clip_by_rect(env, -np.inf, -thickness/2, np.inf, thickness/2),
        top=clip_by_rect(env, -np.inf, thickness/2, np.inf, np.inf),

    """

    resolutions = dict(core={"resolution": coreRes, "distance": width})
    mesh = from_meshio(mesh_from_OrderedDict(polygons, resolutions))#, default_resolution_max=10))
    #mesh.draw().show()
    
    for thisTemp in enumerate(temps):
        print("Temp= "+str(thisTemp[1]))
        for thisLambda in enumerate(lambdas):
            print("Lambda= "+str(thisLambda[1])+f", n_SiO2 = {cladN[thisTemp[0],thisLambda[0]]}, n_TiO2 = {coreN[thisTemp[0],thisLambda[0]]+off}")
            basis0 = Basis(mesh, ElementTriP0())
                        # ---- Parameters (tune these) ----
            nx, ny = 400, 400            # rasterization resolution (higher = more accurate distances)
            xmin, xmax = -width - simBuffer - 0.1, width + simBuffer + 0.1
            ymin, ymax = -0.1, thickness + 0.1   # set bounds to cover your domain + PML
            sigma_max = 8              # maximum imaginary part of n at outermost PML edge
            m = 2                        # grading exponent
            buffer_thickness = simBuffer # thickness of the env buffer (as you created it)

            # ---- 1) make raster grid covering domain ----
            xg = np.linspace(xmin, xmax, nx)
            yg = np.linspace(ymin, ymax, ny)
            Xg, Yg = np.meshgrid(xg, yg)  # shape (ny, nx)

            # ---- 2) rasterize "inside env" mask using shapely.vectorized.contains ----
            # env is your shapely geometry from earlier
            mask_inside = contains_xy(env, Xg, Yg)   # boolean array: True inside env

            # ---- 3) compute distance field: distance from each grid point to the env boundary/outside region
            # We want distance = 0 inside env, and >0 outside env (distance outward)
            # distance_transform_edt computes distance to the nearest nonzero; we want distance of outside points to inside region.
            outside_mask = ~mask_inside
            # distances in grid cells
            dist_px = distance_transform_edt(mask_inside)  # distance of each outside cell to *nearest zero*? careful:
            # We can produce distance from outside to inside by computing edt on the inverse mask:
            dist_out_px = distance_transform_edt(mask_inside == False)  # distance in pixels from each point to nearest inside point
            # Convert pixel distances to physical units (assume uniform spacing)
            dx = (xg[-1] - xg[0]) / (nx - 1)
            dy = (yg[-1] - yg[0]) / (ny - 1)
            # Use average spacing for isotropic distance (approx)
            dist_phys = dist_out_px * np.sqrt(dx*dx + dy*dy)

            # Set distances inside env to zero (we don't want PML inside)
            dist_phys[mask_inside] = 0.0

            # Optionally clip distances to the buffer_thickness (so beyond buffer we saturate)
            dist_clipped = np.clip(dist_phys, 0.0, buffer_thickness)

            # ---- 4) create smooth graded sigma profile on grid ----
            # normalized distance 0..1 across buffer thickness
            norm = dist_clipped / (buffer_thickness + 1e-16)
            sigma_grid = sigma_max * (norm**m)    # shape (ny, nx)

            # ---- 5) build interpolator to get distance (or sigma) at DOF coordinates ----
            # Assume basis0.dofcoords gives an array shape (num_dofs, 2) with (x,y) coordinates
            dof_coords_all = basis0.doflocs.T    # numpy array (N_dofs, 2)

            # Prepare interpolator (Grid is xg, yg but note RegularGridInterpolator expects axes in order (y, x) for 2D arrays)
            interp = RegularGridInterpolator((yg, xg), sigma_grid, bounds_error=False, fill_value=0.0)

            # ---- 6) get DOF indices for pml subdomain and coordinates for those DOFs ----
            pml_dofs = basis0.get_dofs(elements="pml")   # array-like indices of DOFs in pml region
            coords_pml = dof_coords_all[pml_dofs, :]     # shape (num_pml_dofs, 2)

            # RegularGridInterpolator expects shape (..., 2) with order (y, x) -> so keep (x,y) but pass as (y,x) pairs:
            sample_points = np.column_stack((coords_pml[:,1], coords_pml[:,0]))  # (y, x)
            sigma_at_dofs = interp(sample_points)   # shape (num_pml_dofs,)

            # ---- 7) build complex refractive index per-DOF and assign epsilon at those DOFs ----
            n_real = cladN[thisTemp[0], thisLambda[0]]
            n_array = n_real + 1j * sigma_at_dofs

            # If epsilon stores permittivity (epsilon = n^2), assign per DOF:
            epsilon_pml_values = n_array**2

            # Ensure epsilon is complex dtype
            epsilon = basis0.zeros(dtype=complex)

            # Assign for pml DOFs
            epsilon[pml_dofs] = epsilon_pml_values

            # For other subdomains, you can assign scalar values as before:
            indexDict = {
                "core": (coreN[thisTemp[0], thisLambda[0]] + off),
                "clad": cladN[thisTemp[0], thisLambda[0]],
            }
            for subdomain, n in indexDict.items():
                dofs = basis0.get_dofs(elements=subdomain)
                epsilon[dofs] = (n**2)

            """
            # only differences each time are eps values, and lambda in sim
            basis0 = Basis(mesh, ElementTriP0())
            epsilon = basis0.zeros(dtype=complex)
            indexDict = {"core": (coreN[thisTemp[0], thisLambda[0]]+off), 
                         "box": cladN[thisTemp[0],thisLambda[0]], 
                         "side": cladN[thisTemp[0],thisLambda[0]],
                         "top": cladN[thisTemp[0],thisLambda[0]],
                         "pml": cladN[thisTemp[0], thisLambda[0]]+1j
                         }
            for subdomain, n in indexDict.items():
                epsilon[basis0.get_dofs(elements=subdomain)] = n**2
            #if(thisTemp[0] == 0 and thisLambda[0] == 0):
            #    basis0.plot(epsilon, colorbar=True).show()
            """





            modes = compute_modes(basis0, epsilon, wavelength=thisLambda[1], num_modes=4, order=2)
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
                if (mode.n_eff>cladN[thisTemp[0],thisLambda[0]] and mode.n_eff<((coreN[thisTemp[0],thisLambda[0]])+off)):
                    valid=1;
                else:
                    valid=0;
               #TE section
                if mode.te_fraction > mode.tm_fraction: #polCutoff:
                    #print(str(ax[index]))
                    #print(str(ax1))
                    #sweepResult["TE"][thisTemp[0], thisLambda[0]] = np.real(mode.n_eff)
                    #mode.plot_intensity(axis[0][index])
                    #axis[0][index].set_title(f"TE{index}")
                    #print(mode.__dict__)

                    plot_mode(mode.basis, mode.E.real, title=f"TE{index}")
                    print(f"TE{index}, (t = {thisTemp[1]}, l = {thisLambda[1]}) -> neff = {np.real(mode.n_eff):.6f}, valid = {valid}, TE = {mode.te_fraction}, TM = {mode.tm_fraction}")
                #TM section
                else: #mode.tm_fraction > polCutoff:
                    #sweepResult["TM"][thisTemp[0], thisLambda[0]] = np.real(mode.n_eff)
                    #mode.plot_intensity(axis[1][index])
                    #axis[1][index].set_title(f"TM{index}")
                    plot_mode(mode.basis, mode.E.real, title=f"TM{index}")
                    print(f"TM{index}: (t = {thisTemp[1]}, l = {thisLambda[1]}) -> neff = {np.real(mode.n_eff):.6f}, valid = {valid}, TM = {mode.tm_fraction}, TE = {mode.te_fraction}")

                index+=1
                    
            #plt.show()
            #plt.close('all')

            
           #print("made it past for loops")
    return sweepResult

if __name__ == "__main__":
    #widths = np.array([0.3])
    widths = np.array([0.5]) # width
    thicknesses=np.array([0.25])
    #thicknesses= np.linspace(0.1,0.2, 5) # waveguide thickness
    lambdas = np.array([1.55])
    #lambdas = np.linspace(0.6,0.7,2)
    temps=np.array([300])
    #offset for the refractive index just to make sure modes work within a certain range of the RI values sourced from other experiments
    Off=0
    Sep=0.2
    for index in range(1):
        w0=widths[0]
        t0=thicknesses[0]
        simMargin = 3*w0
        coreres = w0/20
        cladres = w0/4
        pml_t=lambdas[0]*3
        print()
        print("----------------------------------")
        print(f"width = {w0*1000}nm,   thickness = {t0*1000}nm")
        print()
        sweepResult = runTempSweep(t0, w0, temps, lambdas, simMargin,
                               coreMat = TiO2materials.oxide2D_ellipsometry, cladMat = materials.oxide2D,
                               coreRes=coreres, cladRes=cladres, off=Off, sep=Sep, pml_thickness=pml_t)
#%%
'''
prefix = '200nmx50nm_TiO2_just_300K_640nm'
filename = datetime.now().strftime('simData/' + prefix + "_%Y-%m-%d-%H-%M-%S")
print(filename)
np.savez(filename + '.npz',  temps = temps, thisLambda = 1e-6*lambdas, sweepResult = sweepResult)
scipy.io.savemat(filename + '.mat', {'indexTemps': temps, 'indexLambda': 1e-6*lambdas, 'indexSimResult': sweepResult})
'''
