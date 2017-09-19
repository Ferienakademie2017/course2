from manta import *
import numpy as np
import sys

# Variable Parameter
y_position = float(sys.argv[1])
y_index = int(sys.argv[2])

# Parameters
obstacle_radius_factor = 0.0625
smoke_source_radius_factor = 0.0725

secOrderBc = True
dim        = 2
res        = 32
#res        = 124
gs         = vec3(2*res,res,res) # unten 2x mal so lang wie rechts
if (dim==2): gs.z = 1
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
obstacle  = Cylinder( parent=s, center=gs*vec3(0.25,y_position,0.5), radius=res*obstacle_radius_factor, z=gs*vec3(0, 0, 1.0))
phiObs    = obstacle.computeLevelset()

# slightly larger copy for density source
densInflow  = Cylinder( parent=s, center=gs*vec3(0.25,y_position,0.5), radius=res*smoke_source_radius_factor, z=gs*vec3(0, 0, 1.0))

phiObs.join(phiWalls)
updateFractions( flags=flags, phiObs=phiObs, fractions=fractions)
setObstacleFlags(flags=flags, phiObs=phiObs, fractions=fractions)
flags.fillGrid()

velInflow = vec3(0.9, 0, 0)
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
if (False):
	gui = Gui()
	gui.show()
	#gui.pause()

#main loop
for t in range(300):
	mantaMsg('\nFrame %i, simulation time %f' % (s.frame, s.timeTotal))

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

	timings.display()
	s.step()

	inter = 10
	if 0 and (t % inter == 0):
		gui.screenshot( 'karman_%04d.png' % int(t/inter) );

# write velocity array to file		
target = np.empty(shape=(2*res, res, 3))
copyGridToArrayVec3(vel, target)
np.save("fluidSamples/{:03d}".format(y_index), target)

print("Finished iteration " + str(y_index))