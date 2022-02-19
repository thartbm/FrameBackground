import time, os, random, sys, json, copy
import numpy as np
from psychopy import visual, core, event, monitors

# 600 dots
# 0.16667 dot life time
# 0.015 dot size


def run_exp(expno=1, setup='laptop'):

    cfg = {}
    cfg['expno'] = expno
    cfg['expstart'] = time.time()

    # get participant ID, set up data folder for them:
    cfg = getParticipant(cfg)

    # set up monitor and visual objects:
    cfg = getStimuli(cfg, setup=setup)

    # set up blocks and trials/tasks within them:
    cfg = getTasks(cfg)

    # try-catch statement in which we try to run all the tasks:
    # each trial saves its own data?
    # at the end a combined data file is produced?

    cfg = runTasks(cfg)

    # save cfg, except for hardware related stuff (window object and stimuli pointing to it)
    saveCfg(cfg)

    # shut down the window object
    cleanExit(cfg)

def doDotTrial(cfg):

    trialtype = cfg['blocks'][cfg['currentblock']]['trialtypes'][cfg['currenttrial']]
    trialdict = cfg['conditions'][trialtype]

    # DO THE TRIAL HERE
    trial_start_time = time.time()

    # straight up copies from the PsychoJS version:
    frequency = 1/trialdict['period']
    distance = trialdict['amplitude']

    # change frequency and distance for static periods at the extremes:
    frequency = frequency + (4/30)
    distance = distance + (distance/((frequency/2)*15))
    p = frequency
    d = distance
    # d = distance / 2 ?

    previous_frame_time = 0
    # # # # # # # # # #
    # WHILE NO RESPONSE

    while (time.time() - trial_start_time) < 3:

        # on every frame:
        this_frame_time = time.time() - trial_start_time
        frame_time_elapsed = this_frame_time - previous_frame_time
        print(frame_time_elapsed)

        # shorter variable for equations:
        t = this_frame_time
        #print(t)

        # move the frame:
        offsetX = (abs((t % p) - (p/2)) / (p/2)) - 0.5
        offsetX = offsetX * d * 10

        flash_red  = False
        flash_blue = False

        if ( ((t + (1/30)) % p) < (2/30)):
            flash_red = True
            if (abs(offsetX) > (d/2)):
                offsetX = np.sign(offsetX) * (d/2)

        if ( ((t + (1/30) + (p/2)) % p) < (2/30) ):
            flash_blue = True
            if (abs(offsetX) > (d/2)):
                offsetX = np.sign(offsetX) * (d/2)

        # cfg['hw']['dotfield'] is a dict with 4 entries:
        # - 'dotsarray': am ElementArrayStim with N Rect objects
        # - 'maxdotlife': a float
        # - 'dotlifetimes': a list of N floats
        # - 'dotpos': a Nx2 array with x,y coordinates

        cfg['hw']['dotfield']['dotlifetimes'] += frame_time_elapsed
        idx = np.nonzero(cfg['hw']['dotfield']['dotlifetimes'] > cfg['hw']['dotfield']['maxdotlife'])[0]
        #print(idx)
        cfg['hw']['dotfield']['dotlifetimes'][idx] -= cfg['hw']['dotfield']['maxdotlife']
        cfg['hw']['dotfield']['xys'][idx,0] = np.random.random(size=len(idx))

        xys = copy.deepcopy(cfg['hw']['dotfield']['xys'])
        xys = xys * 10
        xys[:,0] += offsetX
        cfg['hw']['dotfield']['dotsarray'].setXYs(xys)
        cfg['hw']['dotfield']['dotsarray'].draw()



        cfg['hw']['win'].flip()

        previous_frame_time = this_frame_time

    #for (var idx=0; idx < ndots; idx++) {
    #    dots_lifetime[idx] = (dots_lifetime[idx] + frame_time_elapsed)
    #    if (dots_lifetime[idx] > max_lifetime) {
    #        dots_lifetime[idx] = dots_lifetime[idx] - max_lifetime
    #        dots_pos[idx][0] = randomX(wrap_margins)
    #    }
    #    // dots[idx].setPos([dots_coords[idx][0]+offsetX,Ypos[idx]], false);
    #    // wrap around margins:
    #    if ( (dots_coords[idx][0] + offsetX) < margins[0]) {
    #        // left of left margin, increase by area width, right?
    #        dots[idx].setPos([dots_coords[idx][0]+offsetX+margin_diff,Ypos[idx]], false);
    #    } else if ( (dots_coords[idx][0] + offsetX) > margins[1] ) {
    #        // right of right margin, decrease by area width, right?
    #        dots[idx].setPos([dots_coords[idx][0]+offsetX-margin_diff,Ypos[idx]], false);
    #    } else {
    #        dots[idx].setPos([dots_coords[idx][0]+offsetX,Ypos[idx]], false);
    #    }
    #}

    return(cfg)



def runTasks(cfg):

    if not('currentblock' in cfg):
        cfg['currentblock'] = 0
    if not('currenttrial' in cfg):
        cfg['currenttrial'] = 0

    while cfg['currentblock'] < len(cfg['blocks']):

        # do the trials:

        while cfg['currenttrial'] < len(cfg['blocks'][cfg['currentblock']]['trialtypes']):

            trialtype = cfg['blocks'][cfg['currentblock']]['trialtypes'][cfg['currenttrial']]
            trialdict = cfg['conditions'][trialtype]

            if trialdict['motion'] in ['dotframe','dotbackground']:

                cfg = doDotTrial(cfg)

            cfg['currenttrial'] += 1

        cfg['currentblock'] += 1
        cfg['currenttrial'] = 0

    return(cfg)

def getStimuli(cfg, setup='tablet'):

    gammaGrid = np.array([[0., 1., 1., np.nan, np.nan, np.nan],
                          [0., 1., 1., np.nan, np.nan, np.nan],
                          [0., 1., 1., np.nan, np.nan, np.nan],
                          [0., 1., 1., np.nan, np.nan, np.nan]], dtype=float)
    # for vertical tablet setup:
    if setup == 'tablet':
        gammaGrid = np.array([[0., 136.42685, 1.7472667, np.nan, np.nan, np.nan],
                              [0.,  26.57937, 1.7472667, np.nan, np.nan, np.nan],
                              [0., 100.41914, 1.7472667, np.nan, np.nan, np.nan],
                              [0.,  9.118731, 1.7472667, np.nan, np.nan, np.nan]], dtype=float)
        waitBlanking = True
        resolution = [1680, 1050]
        size = [47.4, 29.6]
        distance = 47

    if setup == 'laptop':
    # for my laptop:
        waitBlanking = True
        resolution   = [1920, 1080]
        size = [34.5, 19.5]
        distance = 40


    mymonitor = monitors.Monitor(name='temp',
                                 distance=distance,
                                 width=size[0])
    mymonitor.setGammaGrid(gammaGrid)
    mymonitor.setSizePix(resolution)

    cfg['gammaGrid']    = list(gammaGrid.reshape([np.size(gammaGrid)]))
    cfg['waitBlanking'] = waitBlanking
    cfg['resolution']   = resolution

    cfg['hw'] = {}

    # first set up the window and monitor:
    cfg['hw']['win'] = visual.Window( fullscr=True,
                                      size=resolution,
                                      units='deg',
                                      waitBlanking=waitBlanking,
                                      color=[0,0,0],
                                      monitor=mymonitor)

    res = cfg['hw']['win'].size
    cfg['relResolution'] = [x / res[1] for x in res]

    cfg['hw']['bluedot'] = visual.Circle(win=cfg['hw']['win'],
                                         units='deg',
                                         size=[.5,.5],
                                         edges=180,
                                         lineWidth=0,
                                         fillColor=[-1,-1,1],
                                         pos=[0,0.20])
    cfg['hw']['reddot'] = visual.Circle(win=cfg['hw']['win'],
                                         units='deg',
                                         size=[.5,.5],
                                         edges=180,
                                         lineWidth=0,
                                         fillColor=[1,-1,-1],
                                         pos=[0,-0.20])


    ndots = 1000
    maxdotlife = 1/6
    xys = [[random.random()-0.5,y] for y in np.linspace(-0.5,0.5,ndots)]
    colors = [[-.5,-.5,-.5],[.5,.5,.5]] * 500
    dotlifetimes = [random.random() * maxdotlife for x in range(ndots)]
    dotMask = np.ones([2,2])

    dotsarray = visual.ElementArrayStim(win = cfg['hw']['win'],
                                        units='deg',
                                        fieldPos=(0,0),
                                        nElements=ndots,
                                        sizes=0.25,
                                        colors=colors,
                                        xys=xys,
                                        elementMask=dotMask
                                        )

    dotfield = {}
    dotfield['maxdotlife']   = maxdotlife
    dotfield['dotlifetimes'] = np.array(dotlifetimes)
    dotfield['dotsarray']    = dotsarray
    dotfield['xys']          = np.array(xys)

    cfg['hw']['dotfield'] = dotfield


    cfg['hw']['bluedot_ref'] = visual.Circle(win=cfg['hw']['win'],
                                         units='deg',
                                         size=[.5,.5],
                                         edges=180,
                                         lineWidth=0,
                                         fillColor=[-1,-1,1],
                                         pos=[0,0.20])
    cfg['hw']['reddot_ref'] = visual.Circle(win=cfg['hw']['win'],
                                         units='deg',
                                         size=[.5,.5],
                                         edges=180,
                                         lineWidth=0,
                                         fillColor=[1,-1,-1],
                                         pos=[0,-0.20])


    return(cfg)


def saveCfg(cfg):

    scfg = copy.copy(cfg)
    del scfg['hw']

    with open('%scfg.json'%(cfg['datadir']), 'w') as fp:
        json.dump(scfg, fp,  indent=4)

def getTasks(cfg):

    if cfg['expno']==1:

        condictionary = [{'period':0.5, 'amplitude':0.55, 'motion':'dotframe'},
                         {'period':0.6, 'amplitude':0.55, 'motion':'dotframe'},
                         {'period':0.7, 'amplitude':0.55, 'motion':'dotframe'},
                         {'period':0.8, 'amplitude':0.55, 'motion':'dotframe'},
                         {'period':0.9, 'amplitude':0.55, 'motion':'dotframe'},
                         {'period':0.7, 'amplitude':0.35, 'motion':'dotframe'},
                         {'period':0.7, 'amplitude':0.45, 'motion':'dotframe'},
                         {'period':0.7, 'amplitude':0.55, 'motion':'dotframe'},
                         {'period':0.7, 'amplitude':0.65, 'motion':'dotframe'},
                         {'period':0.7, 'amplitude':0.75, 'motion':'dotframe'},
                         {'period':0.5, 'amplitude':0.55, 'motion':'dotbackground'},
                         {'period':0.6, 'amplitude':0.55, 'motion':'dotbackground'},
                         {'period':0.7, 'amplitude':0.55, 'motion':'dotbackground'},
                         {'period':0.8, 'amplitude':0.55, 'motion':'dotbackground'},
                         {'period':0.9, 'amplitude':0.55, 'motion':'dotbackground'},
                         {'period':0.7, 'amplitude':0.35, 'motion':'dotbackground'},
                         {'period':0.7, 'amplitude':0.45, 'motion':'dotbackground'},
                         {'period':0.7, 'amplitude':0.55, 'motion':'dotbackground'},
                         {'period':0.7, 'amplitude':0.65, 'motion':'dotbackground'},
                         {'period':0.7, 'amplitude':0.75, 'motion':'dotbackground'}]
        # shorter version:
        condictionary = [{'period':0.5, 'amplitude':0.55, 'motion':'dotframe'},
                         {'period':0.7, 'amplitude':0.55, 'motion':'dotframe'},
                         {'period':0.9, 'amplitude':0.55, 'motion':'dotframe'},
                         {'period':0.7, 'amplitude':0.35, 'motion':'dotframe'},
                         {'period':0.7, 'amplitude':0.75, 'motion':'dotframe'},
                         {'period':0.5, 'amplitude':0.55, 'motion':'dotbackground'},
                         {'period':0.7, 'amplitude':0.55, 'motion':'dotbackground'},
                         {'period':0.9, 'amplitude':0.55, 'motion':'dotbackground'},
                         {'period':0.7, 'amplitude':0.35, 'motion':'dotbackground'},
                         {'period':0.7, 'amplitude':0.75, 'motion':'dotbackground'}]
        cfg['conditions'] = condictionary

        cfg = getMaxAmplitude(cfg)

        blocks = []
        for block in range(1):

            blockconditions = []

            for repeat in range(1):
                trialtypes = list(range(len(condictionary)))
                random.shuffle(trialtypes)
                blockconditions += trialtypes

            blocks += [{'trialtypes':blockconditions,
                        'instruction':'do the trials?'}]

        cfg['blocks'] = blocks

    return(cfg)

def getMaxAmplitude(cfg):

    maxamplitude = 0
    for cond in cfg['conditions']:
        maxamplitude = max(maxamplitude, cond['amplitude'])

    cfg['maxamplitude'] = maxamplitude

    return(cfg)

def foldout(a):
  # http://code.activestate.com/recipes/496807-list-of-all-combination-from-multiple-lists/

  r=[[]]
  for x in a:
    r = [ i + [y] for y in x for i in r ]

  return(r)


def getParticipant(cfg):

    # we need to get an integer number as participant ID:
    IDnotANumber = True

    # and we will only be happy when this is the case:
    while (IDnotANumber):
        # we ask for input:
        ID = input('Enter participant number: ')
        # and try to see if we can convert it to an integer
        try:
            IDno = int(ID)
            if isinstance(ID, int):
                pass # everything is already good
            # and if that integer really reflects the input
            if isinstance(ID, str):
                if not(ID == '%d'%(IDno)):
                    continue
            # only then are we satisfied:
            IDnotANumber = False
            # and store this in the cfg
            cfg['ID'] = IDno
        except Exception as err:
            print(err)
            # if it all doesn't work, we ask for input again...
            pass

    # set up folder's for groups and participants to store the data
    for thisPath in ['data', 'data/exp_%d'%(cfg['expno']), 'data/exp_%d/p%03d'%(cfg['expno'],cfg['ID'])]:
        if os.path.exists(thisPath):
            if not(os.path.isdir(thisPath)):
                os.makedirs
                sys.exit('"%s" should be a folder'%(thisPath))
            else:
                # if participant folder exists, don't overwrite existing data?
                if (thisPath == 'data/exp_%d/p%03d'%(cfg['expno'],cfg['ID'])):
                    sys.exit('participant already exists (crash recovery not implemented)')
        else:
            os.mkdir(thisPath)

    cfg['datadir'] = 'data/exp_%d/p%03d/'%(cfg['expno'],cfg['ID'])

    # we need to seed the random number generator:
    random.seed(99999 * IDno)

    return cfg


def cleanExit(cfg):

    cfg['expfinish'] = time.time()

    saveCfg(cfg)

    # still need to store data...
    print('no data stored on call to exit function...')

    cfg['hw']['win'].close()

    return(cfg)
