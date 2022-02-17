import time, os, random, sys, json
import numpy as np
from psychopy import visual, core, event, monitors

def run_exp(expno=1):

    cfg = {}
    cfg['expno'] = expno
    cfg['expstart'] = time.time()

    # get participant ID, set up data folder for them:
    cfg = getParticipant(cfg)

    # set up monitor and visual objects:
    cfg = setStimuli(cfg)

    # set up blocks and trials/tasks within them:
    #cfg = getTasks(cfg)

    saveCfg(cfg)

    cleanExit(cfg)


def setStimuli(cfg):

    monitors.Monitor()

    # first set up the window and monitor:
    cfg['win'] = visual.Window(fullscr=True, units='pix', waitBlanking=False, color=[-1,-1,-1])

    return(cfg)


def saveCfg(cfg):

    with open('%scfg.json'%(cfg['datadir']), 'w') as fp:
        json.dump(cfg, fp,  indent=4)


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

    saveCfg(cfg)

    # still need to store data...
    print('no data stored on call to exit function...')

    cfg['win'].close()

    return(cfg)
