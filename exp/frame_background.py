import time, os, random, sys, json, copy
import numpy as np
from psychopy import visual, core, event, monitors

# 600 dots
# 0.16667 dot life time
# 0.015 dot size


def run_exp(expno=1):

    cfg = {}
    cfg['expno'] = expno
    cfg['expstart'] = time.time()

    # get participant ID, set up data folder for them:
    cfg = getParticipant(cfg)

    # set up monitor and visual objects:
    cfg = getStimuli(cfg)

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

    dotfield = cfg['hw']['dotfield']

    # DO THE TRIAL HERE
    trialstarttime = time.time()

    

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

def getStimuli(cfg):

    # for vertical tablet setup:
    gammaGrid = np.array([[0., 136.42685, 1.7472667, np.nan, np.nan, np.nan],
                          [0.,  26.57937, 1.7472667, np.nan, np.nan, np.nan],
                          [0., 100.41914, 1.7472667, np.nan, np.nan, np.nan],
                          [0.,  9.118731, 1.7472667, np.nan, np.nan, np.nan]], dtype=float)
    # for my laptop:
    waitBlanking = True
    resolution   = [1920, 1080]



    mymonitor = monitors.Monitor(name='temp')
    mymonitor.setGammaGrid(gammaGrid)

    cfg['gammaGrid']    = list(gammaGrid.reshape([np.size(gammaGrid)]))
    cfg['waitBlanking'] = waitBlanking
    cfg['resolution']   = resolution

    cfg['hw'] = {}

    # first set up the window and monitor:
    cfg['hw']['win'] = visual.Window( fullscr=True,
                                      size=resolution,
                                      units='height',
                                      waitBlanking=waitBlanking,
                                      color=[0,0,0],
                                      monitor=mymonitor)

    res = cfg['hw']['win'].size
    cfg['relResolution'] = [x / res[1] for x in res]

    cfg['hw']['bluedot'] = visual.Circle(win=cfg['hw']['win'],
                                         size=[.05,.05],
                                         edges=180,
                                         lineWidth=0,
                                         fillColor=[-1,-1,1],
                                         pos=[0,0.20])
    cfg['hw']['reddot'] = visual.Circle(win=cfg['hw']['win'],
                                         size=[.05,.05],
                                         edges=180,
                                         lineWidth=0,
                                         fillColor=[1,-1,-1],
                                         pos=[0,-0.20])

    dotfield = []
    dotpositions = np.linspace(0,1,1001)
    fillcolors = [[-.5,-.5,-.5],[.5,.5,.5]]
    color_idx = 0
    for dotpos in dotpositions:
        dot = {}
        color_idx = [1,0][color_idx]
        dot['life'] = random.random()
        dot['pos_y'] = dotpos
        dot['dot']  = visual.Rect(win=cfg['hw']['win'],
                                  width=0.015, height=0.015,
                                  lineWidth=0,
                                  fillColor=fillcolors[color_idx],
                                  pos=[dotpos,random.random()])
        dotfield += [dot]

    cfg['hw']['dotfield'] = dotfield


    cfg['hw']['bluedot_ref'] = visual.Circle(win=cfg['hw']['win'],
                                         size=[.05,.05],
                                         edges=180,
                                         lineWidth=0,
                                         fillColor=[-1,-1,1],
                                         pos=[0,0.20])
    cfg['hw']['reddot_ref'] = visual.Circle(win=cfg['hw']['win'],
                                         size=[.05,.05],
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
