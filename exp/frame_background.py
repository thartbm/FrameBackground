import time, os, random, sys, json, copy
import numpy as np
import pandas as pd
from psychopy import visual, core, event, monitors, tools

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

    opacities = np.array([1]*len(cfg['hw']['dotfield']['dotlifetimes']))

    # straight up copies from the PsychoJS version:
    period = trialdict['period']
    frequency = 1/copy.deepcopy(trialdict['period'])
    distance = trialdict['amplitude']

    # change frequency and distance for static periods at the extremes:
    p = period + (4/30)
    d = distance + (distance/((p/2)*15))
    #p = 1/f

    # DO THE TRIAL HERE
    trial_start_time = time.time()


    previous_frame_time = 0
    # # # # # # # # # #
    # WHILE NO RESPONSE

    frame_times = []
    frame_pos   = []
    blue_on     = []
    red_on      = []

    # we show a blank screen for 1/3 - 2.3 of a second (uniform dist):
    blank = 1/3 + (random.random() * 1/3)

    # the frame motion gets multiplied by -1 or 1:
    xfactor = [-1,1][random.randint(0,1)]

    # the mouse response has a random offset between -3 and 3 degrees
    mouse_offset = (random.random() - 0.5) * 6

    waiting_for_response = True

    while waiting_for_response:

        while (time.time() - trial_start_time) < blank:
            event.clearEvents(eventType='mouse')
            event.clearEvents(eventType='keyboard')
            cfg['hw']['win'].flip()



        # on every frame:
        this_frame_time = time.time() - trial_start_time
        frame_time_elapsed = this_frame_time - previous_frame_time
        #print(round(1/frame_time_elapsed))

        # shorter variable for equations:
        t = this_frame_time
        #print(t)

        # move the frame:

        offsetX = abs( (t % p) - (p/2) ) * (2/p) - 0.5
        offsetX = offsetX * d


        flash_red  = False
        flash_blue = False

        if ( ((t + (1/30)) % p) < (2/30)):
            flash_red = True
            if (abs(offsetX) > (distance/2)):
                offsetX = np.sign(offsetX) * (distance/2)

        if ( ((t + (1/30) + (p/2)) % p) < (2/30) ):
            flash_blue = True
            if (abs(offsetX) > (distance/2)):
                offsetX = np.sign(offsetX) * (distance/2)

        offsetX = offsetX * xfactor

#        minOffset = min(minOffset, offsetX)
#        maxOffset = max(maxOffset, offsetX)
        # cfg['hw']['dotfield'] is a dict with 4 entries:
        # - 'dotsarray': am ElementArrayStim with N Rect objects
        # - 'maxdotlife': a float
        # - 'dotlifetimes': a list of N floats
        # - 'dotpos': a Nx2 array with x,y coordinates
        if trialdict['stimtype'] in ['dotframe','dotbackground']:

            cfg['hw']['dotfield']['dotlifetimes'] += frame_time_elapsed
            idx = np.nonzero(cfg['hw']['dotfield']['dotlifetimes'] > cfg['hw']['dotfield']['maxdotlife'])[0]
            #print(idx)
            cfg['hw']['dotfield']['dotlifetimes'][idx] -= cfg['hw']['dotfield']['maxdotlife']
            cfg['hw']['dotfield']['xys'][idx,0] = np.random.random(size=len(idx)) - 0.5

            xys = copy.deepcopy(cfg['hw']['dotfield']['xys'])
            xys[:,0] = xys[:,0] * (24 + cfg['maxamplitude'] - cfg['hw']['dotfield']['dotsize'])
            xys[:,1] = xys[:,1] * (15 - cfg['hw']['dotfield']['dotsize'])

            opacities[:] = 1
            if (trialdict['stimtype'] == 'dotframe'):
                opacities[np.nonzero(abs(xys[:,0]) > (12 - (cfg['hw']['dotfield']['dotsize']/2)))[0]] = 0
            xys[:,0] += offsetX
            if (trialdict['stimtype'] == 'dotbackground'):
                opacities[np.nonzero(abs(xys[:,0]) > (12 - (cfg['hw']['dotfield']['dotsize']/2)))[0]] = 0
            xys[:,0] = xys[:,0] - cfg['stim_offsets'][0]
            xys[:,1] = xys[:,1] - cfg['stim_offsets'][1]
            cfg['hw']['dotfield']['dotsarray'].setXYs(xys)
            cfg['hw']['dotfield']['dotsarray'].opacities = opacities
            cfg['hw']['dotfield']['dotsarray'].draw()

        if trialdict['stimtype'] in ['classicframe']:
            frame_pos = [offsetX-cfg['stim_offsets'][0], -cfg['stim_offsets'][1]]
            cfg['hw']['white_frame'].pos = frame_pos
            cfg['hw']['white_frame'].draw()
            cfg['hw']['gray_frame'].pos = frame_pos
            cfg['hw']['gray_frame'].draw()


        if flash_red:
            cfg['hw']['reddot'].draw()
        if flash_blue:
            cfg['hw']['bluedot'].draw()


        cfg['hw']['win'].flip()

        previous_frame_time = this_frame_time

        frame_times += [this_frame_time]
        frame_pos   += [offsetX]
        blue_on     += [flash_blue]
        red_on      += [flash_red]

        # in DEGREES:
        mousepos = cfg['hw']['mouse'].getPos()
        percept = (mousepos[0] + mouse_offset) / 4

        # blue is on top:
        cfg['hw']['bluedot_ref'].pos = [percept+(2.5*cfg['stim_offsets'][0]),cfg['stim_offsets'][1]+10]
        cfg['hw']['reddot_ref'].pos = [-percept+(2.5*cfg['stim_offsets'][0]),cfg['stim_offsets'][1]+6]
        cfg['hw']['bluedot_ref'].draw()
        cfg['hw']['reddot_ref'].draw()

        keys = event.getKeys(keyList=['space'])
        if len(keys):
            waiting_for_response = False

    # save a data frame as csv?
    #stimulus_data = pd.DataFrame({'time':frame_times,
    #                              'frameX':frame_pos,
    #                              'red_flashed':red_on,
    #                              'blue_flashed':blue_on})
    #stimulus_data.to_csv('data/exp_1/%0.1fs_%0.1fdeg.csv'%(period,distance), index=False)

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

    #print([minOffset, maxOffset])

    return(cfg)



def runTasks(cfg):

    cfg = getMaxAmplitude(cfg)

    if not('currentblock' in cfg):
        cfg['currentblock'] = 0
    if not('currenttrial' in cfg):
        cfg['currenttrial'] = 0

    while cfg['currentblock'] < len(cfg['blocks']):

        # do the trials:

        while cfg['currenttrial'] < len(cfg['blocks'][cfg['currentblock']]['trialtypes']):

            trialtype = cfg['blocks'][cfg['currentblock']]['trialtypes'][cfg['currenttrial']]
            trialdict = cfg['conditions'][trialtype]

            if trialdict['stimtype'] in ['dotframe','dotbackground']:

                cfg = doDotTrial(cfg)

            if trialdict['stimtype'] in ['classicframe']:

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

    cfg['stim_offsets'] = [6,2]

    cfg['hw']['bluedot'] = visual.Circle(win=cfg['hw']['win'],
                                         units='deg',
                                         size=[1.5,1.5],
                                         edges=180,
                                         lineWidth=0,
                                         fillColor=[-1,-1,1],
                                         pos=[0-cfg['stim_offsets'][0],6-cfg['stim_offsets'][1]])
    cfg['hw']['reddot'] = visual.Circle(win=cfg['hw']['win'],
                                         units='deg',
                                         size=[1.5,1.5],
                                         edges=180,
                                         lineWidth=0,
                                         fillColor=[1,-1,-1],
                                         pos=[0-cfg['stim_offsets'][0],-6-cfg['stim_offsets'][1]])


    ndots = 600
    maxdotlife = 1/5
    ypos = np.linspace(-0.5,0.5,ndots)
    random.shuffle(ypos)
    xys = [[random.random()-0.5,y] for y in ypos]
    #colors = [[-.25,-.25,-.25],[.25,.25,.25]] * 400
    colors = [[-.3,-.3,-.3],[.3,.3,.3]] * 300
    dotlifetimes = [random.random() * maxdotlife for x in range(ndots)]
    dotMask = np.ones([32,32])
    dotsize = 1

    dotsarray = visual.ElementArrayStim(win = cfg['hw']['win'],
                                        units='deg',
                                        fieldPos=(0,0),
                                        nElements=ndots,
                                        sizes=dotsize,
                                        colors=colors,
                                        xys=xys,
                                        elementMask=dotMask,
                                        elementTex=dotMask
                                        )

    dotfield = {}
    dotfield['maxdotlife']   = maxdotlife
    dotfield['dotlifetimes'] = np.array(dotlifetimes)
    dotfield['dotsarray']    = dotsarray
    dotfield['xys']          = np.array(xys)
    dotfield['dotsize']      = dotsize

    cfg['hw']['dotfield'] = dotfield

    print(tools.monitorunittools.deg2pix(0.75, monitor=mymonitor))
    #cfg['hw']['frame'] = visual.Rect(win=cfg['hw']['win'],
    #                                 width=24,
    #                                 height=15,
    #                                 units='deg',
    #                                 lineWidth=60,
    #                                 lineColor=[1,1,1],
    #                                 fillColor=None,
    #                                 )
    cfg['hw']['white_frame'] = visual.Rect(win=cfg['hw']['win'],
                                           width=24,
                                           height=15,
                                           units='deg',
                                           lineColor=None,
                                           lineWidth=0,
                                           fillColor=[1,1,1])
    cfg['hw']['gray_frame'] =  visual.Rect(win=cfg['hw']['win'],
                                           width=23,
                                           height=14,
                                           units='deg',
                                           lineColor=None,
                                           lineWidth=0,
                                           fillColor=[0,0,0])


    cfg['hw']['bluedot_ref'] = visual.Circle(win=cfg['hw']['win'],
                                         units='deg',
                                         size=[0.5,0.5],
                                         edges=180,
                                         lineWidth=0,
                                         fillColor=[-1,-1,1],
                                         pos=[0,0.20])
    cfg['hw']['reddot_ref'] = visual.Circle(win=cfg['hw']['win'],
                                         units='deg',
                                         size=[0.5,0.5],
                                         edges=180,
                                         lineWidth=0,
                                         fillColor=[1,-1,-1],
                                         pos=[0,-0.20])

    # we also want to set up a mouse object:
    cfg['hw']['mouse'] = event.Mouse(visible=False, newPos=None, win=cfg['hw']['win'])
    # keyboard is not an object, already accessible through psychopy.event

    return(cfg)


def saveCfg(cfg):

    scfg = copy.copy(cfg)
    del scfg['hw']

    with open('%scfg.json'%(cfg['datadir']), 'w') as fp:
        json.dump(scfg, fp,  indent=4)

def getTasks(cfg):

    if cfg['expno']==1:

        # 1.0 - 0.3333 seconds, 12 deg motion:
        # durations: 1.000, 0.6666, 0.5000, 0.4000 and 0.3333

        condictionary = [{'period':1.0, 'amplitude':12, 'stimtype':'dotframe'},
                         {'period':2/3, 'amplitude':12, 'stimtype':'dotframe'},
                         {'period':1/2, 'amplitude':12, 'stimtype':'dotframe'},
                         {'period':2/5, 'amplitude':12, 'stimtype':'dotframe'},
                         {'period':1/3, 'amplitude':12, 'stimtype':'dotframe'},
                         {'period':1/3, 'amplitude':4, 'stimtype':'dotframe'},
                         {'period':1.3, 'amplitude':6, 'stimtype':'dotframe'},
                         {'period':1/3, 'amplitude':8, 'stimtype':'dotframe'},
                         {'period':1/3, 'amplitude':10, 'stimtype':'dotframe'},
                         {'period':1/3, 'amplitude':12, 'stimtype':'dotframe'},
                         {'period':1.0, 'amplitude':12, 'stimtype':'dotbackground'},
                         {'period':2/3, 'amplitude':12, 'stimtype':'dotbackground'},
                         {'period':1/2, 'amplitude':12, 'stimtype':'dotbackground'},
                         {'period':2/5, 'amplitude':12, 'stimtype':'dotbackground'},
                         {'period':1/3, 'amplitude':12, 'stimtype':'dotbackground'},
                         {'period':1/3, 'amplitude':4, 'stimtype':'dotbackground'},
                         {'period':1/3, 'amplitude':6, 'stimtype':'dotbackground'},
                         {'period':1/3, 'amplitude':8, 'stimtype':'dotbackground'},
                         {'period':1/3, 'amplitude':10, 'stimtype':'dotbackground'},
                         {'period':1/3, 'amplitude':12, 'stimtype':'dotbackground'},
                         ]
        # shorter version:
        condictionary = [{'period':1/3, 'amplitude':12, 'stimtype':'dotframe'},
                         {'period':1/3, 'amplitude':8, 'stimtype':'dotframe'},
                         {'period':1/3, 'amplitude':4., 'stimtype':'dotframe'},
                         {'period':1.0, 'amplitude':12, 'stimtype':'dotframe'},
                         {'period':1/2, 'amplitude':12, 'stimtype':'dotframe'},
                         {'period':1/3, 'amplitude':12, 'stimtype':'dotbackground'},
                         {'period':1/3, 'amplitude':8, 'stimtype':'dotbackground'},
                         {'period':1/3, 'amplitude':4., 'stimtype':'dotbackground'},
                         {'period':1.0, 'amplitude':12, 'stimtype':'dotbackground'},
                         {'period':1/2, 'amplitude':12, 'stimtype':'dotbackground'},
                         {'period':1/3, 'amplitude':12, 'stimtype':'classicframe'},
                         {'period':1/3, 'amplitude':4., 'stimtype':'classicframe'},
                         {'period':1.0, 'amplitude':12, 'stimtype':'classicframe'},
                         {'period':1/2, 'amplitude':12, 'stimtype':'classicframe'},
                         ]

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
