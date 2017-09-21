from manta import *
import numpy
import utils
import Sim1Result
import random
import TrainingConfiguration
import ObstacleContainer



def generateTrainingExamples(trainingConfiguration,initialConditions,obstacleCreator = ObstacleContainer.simpleCylinder):

    zComp = numpy.zeros( (trainingConfiguration.resY, trainingConfiguration.resX,1), dtype='f')
    initialConditions = numpy.concatenate((initialConditions, zComp), axis=2)
    secOrderBc = True
    dim        = 2
    res        = 32
    #res        = 124
    gs         = vec3(trainingConfiguration.resX,trainingConfiguration.resY,trainingConfiguration.resY)
    if (dim==2): gs.z = 1
    s          = FluidSolver(name='main', gridSize = gs, dim=dim)
    s.timestep = 1.

    printSimulation = False
    GUI = False

    #Zylinder_Position
    pos = [0.25,0.5,0.5]

    #Parameters for saving data
    simPath = trainingConfiguration.simPath
    savedata = trainingConfiguration.savedata
    saveppm = trainingConfiguration.saveppm
    # savepng = trainingConfiguration.savepng  # todo
    interval = trainingConfiguration.saveInterval
    offset = trainingConfiguration.timeOffset
    npVel = numpy.zeros( (trainingConfiguration.resY, trainingConfiguration.resX, 3), dtype='f')
    npObs = numpy.zeros( (trainingConfiguration.resY, trainingConfiguration.resX), dtype='f')

    #Number of generated Images
    NumObsPosX = trainingConfiguration.NumObsPosX
    NumObsPosY = trainingConfiguration.NumObsPosY

    flags     = s.create(FlagGrid)
    density   = s.create(RealGrid)
    vel       = s.create(MACGrid)
    density   = s.create(RealGrid)
    pressure  = s.create(RealGrid)
    fractions = s.create(MACGrid)
    phiWalls  = s.create(LevelsetGrid)

    flags.initDomain(inflow="xX", phiWalls=phiWalls, boundaryWidth=0)

    inflow = initialConditions[:,0,:]

    for simNo in range(0,NumObsPosX*NumObsPosY):
        #obstacle  = Sphere(   parent=s, center=gs*vec3(0.25,0.5,0.5), radius=res*0.2)
        for obstacle in obstacleCreator(s,trainingConfiguration,simNo):
            phiObs = obstacle.computeLevelset()
            phiWalls.join(phiWalls)


        # slightly larger copy for density source
        #densInflow  = Cylinder( parent=s, center=gs*vec3(0.25,0.5,0.5), radius=res*0.21, z=gs*vec3(0, 0, 1.0))

        updateFractions( flags=flags, phiObs=phiWalls, fractions=fractions)
        setObstacleFlags(flags=flags, phiObs=phiWalls, fractions=fractions)
        flags.fillGrid()

        velInflow = vec3(0.9, 0, 0)
        copyArrayToGridVec3(target = vel,source = initialConditions)

        # optionally randomize y component
        if 0:
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

        if (GUI):
            gui = Gui()
            gui.show()
            #gui.pause()

        #main loop
        for t in range(trainingConfiguration.NumSteps + 1):

            #densInflow.applyToGrid( grid=density, value=2. )

            advectSemiLagrange(flags=flags, vel=vel, grid=density, order=2, orderSpace=1)
            advectSemiLagrange(flags=flags, vel=vel, grid=vel    , order=2, strength=1.0)

            if(secOrderBc):
                extrapolateMACSimple( flags=flags, vel=vel, distance=2 , intoObs=True);
                setWallBcs(flags=flags, vel=vel, fractions=fractions, phiObs=phiWalls)

                setInflowBcs(vel=vel,dir='xX',value=velInflow)
                solvePressure( flags=flags, vel=vel, pressure=pressure, fractions=fractions, cgAccuracy=cgAcc, cgMaxIterFac=cgIter)

                extrapolateMACSimple( flags=flags, vel=vel, distance=5 , intoObs=True);
                setWallBcs(flags=flags, vel=vel, fractions=fractions, phiObs=phiWalls)
            else:
                setWallBcs(flags=flags, vel=vel)
                setInflowBcs(vel=vel,dir='xX',value=velInflow)
                solvePressure( flags=flags, vel=vel, pressure=pressure, cgAccuracy=cgAcc, cgMaxIterFac=cgiter )
                setWallBcs(flags=flags, vel=vel)

            #setInflowBcs(vel=vel,dir='xX',value=velInflow)
            copyGridToArrayVec3(source = vel, target = npVel)
            applyBoundaryValues(initialConditions,npVel,vel)

            if printSimulation:
                mantaMsg('\nFrame %i, simulation time %f' % (s.frame, s.timeTotal))
                timings.display()

            s.step()

            # save data
            if savedata and t>=offset and (t-offset)%interval==0:
                tf = (t-offset)//interval
                #framePath = simPath + 'frame_%04d/' % tf
                #os.makedirs(framePath)
                copyGridToArrayVec3(source = vel, target = npVel)
                copyGridToArrayLevelset(source = phiObs, target = npObs)
                npVel = np.transpose(npVel, (1, 0, 2))
                npObs = np.transpose(npObs)
                result = Sim1Result.Sim1Result(npVel, pos, npObs)
                utils.sim1resToImage(result)
                utils.serialize(simPath+trainingConfiguration.getFileNameFor(simNo,t), result)
                if(saveppm):
                    projectPpmFull( density, simPath + 'density_{}_{}.ppm'.format(simNo, tf), 0, 1.0 )

            inter = 10
            if 0 and (t % inter == 0):
                gui.screenshot( 'karman_{}.png'.format(int(t/inter)) );

def applyBoundaryValues(initialConditions,npVel,vel):
    #print(npVel[:,1:,:])
    #print(initialConditions[:,0,:])
    npVel[:,0,:] = initialConditions[:,0,:]
    #npVel = numpy.concatenate((initialConditions[:,0,:],npVel[:,1:,:]), axis=0)
    vel = copyArrayToGridVec3(source = npVel,target = vel)


initialConditions = numpy.concatenate((numpy.ones((32,64,1), dtype='f'),numpy.zeros((32,64,1), dtype='f')),axis = 2)
trainingConfiguration = TrainingConfiguration.TrainingConfiguration()
generateTrainingExamples(trainingConfiguration,initialConditions,obstacleCreator=ObstacleContainer.generateObstacleContainer)
list = trainingConfiguration.loadGeneratedData()
