from manta import *
import numpy as np
import sys
import scipy.ndimage
import scipy.misc

from utils import get_parameter

# Variable Parameter
y_position = float(sys.argv[1])
y_index = int(sys.argv[2])

# Parameters
obstacle_radius_factor = get_parameter("obstacle_radius_factor")
smoke_source_radius_factor = get_parameter("smoke_source_radius_factor")

secOrderBc = True
dim        = 2
res        = get_parameter("resolution")
gs         = vec3(2*res,res,res) # unten 2x mal so lang wie rechts
if (dim==2): gs.z = 1
x_position = gs.x * get_parameter("relative_x_position")

s          = FluidSolver(name='main', gridSize = gs, dim=dim)
s.timestep = 1.

flags     = s.create(FlagGrid)
density   = s.create(RealGrid)
vel       = s.create(MACGrid)
pressure  = s.create(RealGrid)
fractions = s.create(MACGrid)
phiWalls  = s.create(LevelsetGrid)

flags.initDomain(inflow="xX", phiWalls=phiWalls, boundaryWidth=0)

#obstacle  = Sphere(   parent=s, center=gs*vec3(0.25,0.5,0.5), radius=res*0.2)
obstacle  = Cylinder(parent=s, center=vec3(x_position, y_position, 1), 
        radius=res*obstacle_radius_factor, z=gs*vec3(0, 0, 1.0))
phiObs    = obstacle.computeLevelset()

# slightly larger copy for density source
densInflow  = Cylinder(parent=s, center=vec3(x_position, y_position, 1), 
        radius=res*smoke_source_radius_factor, z=gs*vec3(0, 0, 1.0))

phiObs.join(phiWalls)
updateFractions( flags=flags, phiObs=phiObs, fractions=fractions)
setObstacleFlags(flags=flags, phiObs=phiObs, fractions=fractions)
flags.fillGrid()

velInflow = vec3(*get_parameter("velocity_in"))
vel.setConst(velInflow)

# optionally randomize y component
if 1:
	noise = s.create(NoiseField, loadFromFile=True)
	noise.posScale = vec3(75)
	noise.clamp    = True
	noise.clampNeg = -1.
	noise.clampPos =  1.
	testall = s.create(RealGrid); testall.setConst(-1.);
	addNoise(flags=flags, density=density, noise=noise, sdf=testall, scale=0.1 )

setComponent(target=vel, source=density, component=1)
density.setConst(0.)

# cg solver params
cgAcc    = 1e-04
cgIter = 5

timings = Timings()

# GUI
if get_parameter("show_gui"):
	gui = Gui()
	gui.show()
	#gui.pause()

#main loop
for t in range(get_parameter("nr_frames")):
	#mantaMsg('\nFrame %i, simulation time %f' % (s.frame, s.timeTotal))

	densInflow.applyToGrid( grid=density, value=2. )

	advectSemiLagrange(flags=flags, vel=vel, grid=density, order=2, orderSpace=1)  
	advectSemiLagrange(flags=flags, vel=vel, grid=vel    , order=2, strength=1.0)

	if(secOrderBc):
		extrapolateMACSimple( flags=flags, vel=vel, distance=2 , intoObs=True);
		setWallBcs(flags=flags, vel=vel, fractions=fractions, phiObs=phiObs)

		setInflowBcs(vel=vel,dir='xX',value=velInflow)
		solvePressure( flags=flags, vel=vel, pressure=pressure, fractions=fractions, cgAccuracy=cgAcc, cgMaxIterFac=cgIter)

		extrapolateMACSimple( flags=flags, vel=vel, distance=5 , intoObs=True);
		setWallBcs(flags=flags, vel=vel, fractions=fractions, phiObs=phiObs)
	else:
		setWallBcs(flags=flags, vel=vel)
		setInflowBcs(vel=vel,dir='xX',value=velInflow)
		solvePressure( flags=flags, vel=vel, pressure=pressure, cgAccuracy=cgAcc, cgMaxIterFac=cgiter ) 
		setWallBcs(flags=flags, vel=vel)

	setInflowBcs(vel=vel,dir='xX',value=velInflow)

	#timings.display()
	s.step()

	inter = 10
	if 0 and (t % inter == 0):
		gui.screenshot( 'karman_%04d.png' % int(t/inter) );

# write velocity array to file		
velocity_target = np.empty(shape=(res, 2*res, 3))
copyGridToArrayVec3(vel, velocity_target)

# create colorful image
pil_image = scipy.misc.toimage(velocity_target)
pil_image.save("fluidSamples6432Images/{:04d}.png".format(y_index))

# throw away z axis
velocity_target = velocity_target[:,:,:2]
print(velocity_target.shape)

# save high res image
np.save("fluidSamples6432/{:04d}".format(y_index), velocity_target)

# scale down image
velocity_target = scipy.ndimage.zoom(velocity_target, get_parameter("downscaling_factors"), order=1)
# save low res image
np.save("fluidSamples1608/{:04d}".format(y_index), velocity_target)

print("Finished iteration " + str(y_index))
