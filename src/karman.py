from manta import *
import numpy as np
import utils
import Sim1Result
import random
import TrainingConfiguration
import ObstacleContainer
import copy



def generateTrainingExamples(trainingConfiguration,initialConditions,obstacleCreator = ObstacleContainer.simpleCylinder):

    zComp = np.zeros( (trainingConfiguration.resY, trainingConfiguration.resX,1), dtype='f')
    initialConditions = np.concatenate((initialConditions, zComp), axis=2)
    secOrderBc = True
    dim        = 2
    res        = 32
    #res        = 124
    gs         = vec3(trainingConfiguration.resX,trainingConfiguration.resY,trainingConfiguration.resY)
    if (dim==2): gs.z = 1
    s          = FluidSolver(name='main', gridSize = gs, dim=dim)
    s.timestep = 1.

    printSimulation = False
    GUI = trainingConfiguration.GUI

    #Zylinder_Position
    pos = [0.25,0.5,0.5]

    #Parameters for saving data
    simPath = trainingConfiguration.simPath
    savedata = trainingConfiguration.savedata
    saveppm = trainingConfiguration.saveppm
    # savepng = trainingConfiguration.savepng  # todo
    interval = trainingConfiguration.saveInterval
    offset = trainingConfiguration.timeOffset
    npVel = np.zeros( (trainingConfiguration.resY, trainingConfiguration.resX, 3), dtype='f')
    npObs = np.zeros( (trainingConfiguration.resY, trainingConfiguration.resX), dtype='f')

    #Number of generated Images
    NumObsPosX = trainingConfiguration.NumObsPosX
    NumObsPosY = trainingConfiguration.NumObsPosY

    flags     = s.create(FlagGrid, name="flags")
    density   = s.create(RealGrid, name="density")
    vel       = s.create(MACGrid, name="vel")
    density   = s.create(RealGrid, name="density")
    pressure  = s.create(RealGrid, name="pressure")
    fractions = s.create(MACGrid, name="fractions")
    phiWallsOrig  = s.create(LevelsetGrid)
    phiWalls = s.create(LevelsetGrid)

    flags.initDomain(inflow="xX", phiWalls=phiWallsOrig, boundaryWidth=0)

    inflow = initialConditions[:,0,:]

    for simNo in range(0,NumObsPosX*NumObsPosY):
        result_List = []
        #pos = [random.random(),random.random(),0.5]
        #pos = [0.4, 0.8, 0.5]
        #1. Komponente ist x-Komponente, 2. Komponente ist y-Komponente
        pos = [(simNo % NumObsPosX)*1.0/NumObsPosX, (simNo//NumObsPosX)*1.0/NumObsPosY,0.5]

        #obstacle  = Sphere(   parent=s, center=gs*vec3(0.25,0.5,0.5), radius=res*0.2)
        phiWalls.setConst(1000)
        phiWalls.join(phiWallsOrig)

        obstacleList = obstacleCreator(s,trainingConfiguration,simNo)

        for obstacle in obstacleList:
            phiObs = obstacle.computeLevelset()
            phiWalls.join(phiObs)

        posVec3 = obstacleList[0].getCenter()
        pos = [posVec3.x,posVec3.y,posVec3.z]


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
        cgIter = 500

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
                copyGridToArrayVec3(source=vel, target=npVel)
                copyGridToArrayLevelset(source=phiWalls, target=npObs)
                applyBoundaryValues(initialConditions, npObs, npVel, vel)
                solvePressure( flags=flags, vel=vel, pressure=pressure, fractions=fractions, cgAccuracy=cgAcc, cgMaxIterFac=cgIter)

                extrapolateMACSimple( flags=flags, vel=vel, distance=5 , intoObs=True);
                setWallBcs(flags=flags, vel=vel, fractions=fractions, phiObs=phiWalls)
            else:
                setWallBcs(flags=flags, vel=vel)
                setInflowBcs(vel=vel,dir='xX',value=velInflow)
                copyGridToArrayVec3(source=vel, target=npVel)
                copyGridToArrayLevelset(source=phiWalls, target=npObs)
                applyBoundaryValues(initialConditions, npObs, npVel, vel)
                solvePressure( flags=flags, vel=vel, pressure=pressure, cgAccuracy=cgAcc, cgMaxIterFac=cgiter )
                setWallBcs(flags=flags, vel=vel)

            #setInflowBcs(vel=vel,dir='xX',value=velInflow)
            copyGridToArrayVec3(source=vel, target=npVel)
            copyGridToArrayLevelset(source=phiWalls, target=npObs)
            applyBoundaryValues(initialConditions, npObs, npVel, vel)

            if printSimulation:
                mantaMsg('\nFrame %i, simulation time %f' % (s.frame, s.timeTotal))
                timings.display()

            s.step()

            # save data
            if savedata and t>=offset and (t-offset)%interval==0:
                tf = (t-offset)//interval
                #framePath = simPath + 'frame_%04d/' % tf
                #os.makedirs(framePath)

                npVelsave = np.transpose(npVel, (1, 0, 2))
                npObssave = np.transpose(npObs)
                result = Sim1Result.Sim1Result(npVelsave,pos, npObssave,t)
                result_List.append(result)
                # utils.sim1resToImage(result)
                if(saveppm):
                    projectPpmFull( density, simPath + 'density_{}_{}.ppm'.format(simNo, tf), 0, 1.0 )

            inter = 10
            if 0 and (t % inter == 0):
                gui.screenshot( 'karman_{}.png'.format(int(t/inter)) )
        utils.serialize(simPath + trainingConfiguration.getFileNameFor(trainingConfiguration.counter), result_List)
        trainingConfiguration.counter += 1

def applyBoundaryValues(initialConditions,npObs,npVel,vel):
    #print(npVel[:,1:,:])
    #print(initialConditions[:,0,:])
    #npVel = np.concatenate((initialConditions[:,0,:],npVel[:,1:,:]), axis=0)
    it = np.nditer(npObs, flags=['multi_index'])
    while not it.finished:
        if it[0] < 0:
            ind = it.multi_index
            for i in range(3):
                tuple = (ind[0], ind[1], i)
                npVel[ind[0], ind[1], i] = 0.0
            # vel[ind] = np.zeros(3,dtype='f')
        it.iternext()
    npVel[:,0,:] = initialConditions[:,0,:]
    copyArrayToGridVec3(source=npVel, target=vel)


#initialConditions = np.concatenate((np.ones((32,64,1), dtype='f'),np.zeros((32,64,1), dtype='f')),axis = 2)
#trainingConfiguration = TrainingConfiguration.TrainingConfiguration()
#generateTrainingExamples(trainingConfiguration,initialConditions,obstacleCreator=ObstacleContainer.generateObstacleContainer)
#list = trainingConfiguration.loadGeneratedData()
