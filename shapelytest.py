from collections import OrderedDict
import matplotlib.pyplot as plt
import numpy as np
import shapely
import shapely.affinity
from scipy.constants import epsilon_0, speed_of_light
from shapely.ops import clip_by_rect
from skfem import Basis, ElementTriP0
from skfem.io.meshio import from_meshio
from femwell.maxwell.waveguide import compute_modes
from femwell.mesh import mesh_from_OrderedDict
from femwell.visualization import plot_domains
from shapely.geometry import Polygon, MultiPolygon
from shapely.geometry import Point
"""
wg_width = 2.5
wg_thickness = 0.3
core = shapely.geometry.box(-wg_width / 2, 0, +wg_width / 2, wg_thickness)
env = shapely.affinity.scale(core.buffer(5, resolution=4))
polygons = OrderedDict(
    core=core,
    box=clip_by_rect(env, -np.inf, -np.inf, np.inf, 0),
    clad=clip_by_rect(env, -np.inf, 0, np.inf, np.inf),
)

resolutions = dict(core={"resolution": 0.03, "distance": 2})

mesh = from_meshio(mesh_from_OrderedDict(polygons, resolutions, default_resolution_max=1))
"""
#code with two modes
width=0.35
thickness=0.2
sep=0.3
simBuffer=2*width
coreRes = width/20
cladRes = width/20
radius = 5*width
left_core = shapely.geometry.box(-width-sep/2, -thickness/2, -sep/2, thickness/2)
right_core = shapely.geometry.box(sep/2, -thickness/2, width+sep/2, thickness/2)
cores=MultiPolygon([left_core, right_core])
env = Point(0, 0).buffer(radius, resolution=16)

# Define a surrounding box for PML
pml_thickness = 2*width  # microns, this assumes only one element in lamdas
pml_outer = shapely.affinity.scale(env.buffer(pml_thickness, resolution=16), xfact=1)
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
resolutions = dict(core={"resolution": coreRes, "distance": simBuffer})
mesh = from_meshio(mesh_from_OrderedDict(polygons, resolutions, default_resolution_max=10))


#mesh.draw().show()

#plot_domains(mesh)
#plt.show()
