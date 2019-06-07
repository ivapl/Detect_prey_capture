# THIS CODE WILL DETERMINE THE SELECTED PREY AND ITS LOCATION
# BASED ON THE ONSET OF THE CONTRA-LATERAL EYE CONVERGENCE
# The onset is defined as the point between the peak of eye velocity
# and the start of the tail bout.
# Last update: 23 AUG 2018, Ivan

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy.signal import savgol_filter
from operator import add

from csvfiles_reader import *
from extract_taileyebouts import *
from eye_movement_filters import *
from get_preylocation import *
import operator
from itertools import izip_longest
import pickle
import seaborn as sns

plt.style.use(['dark_background'])

# ============================ MACRO FOR PREY SELECTION ANALYSIS =======================================================
# PARAMETERS FOR EXTRACTING THE BOUTS, AND PEAKS
Fs = 300
bout_thresh = 0.40  # 0.00 - 1.00 threshold value for extracting bouts, higher more bouts
tfactor = 0.3  # convert frames to ms
peakthres = 4  # 0.00 - 20.00 lower vale more peaks, for calculating tail beat frequency
speed = 20.0 # prey speed
filter_avg_binocular = 17.0 # threshold for eye angle binocular convergence
filter_length_bout = 10 # length of bout based on frame number
filter_eye_vel = -0.1 # diverging eye velocity in degrees/ms
filter_eye_diverge = -1.0 # eye divergence by degrees
thresh_saccade_speed = 0.2 # the onset of eye movement should be greater than saccade threshold
sensory_delay = 30 # delay of sensory transformation in number of frames
preypoints = [10, 70] # direction of prey in visual angle
delay = 50 # window delay for finding the peak
prey_size = [2,3] # prey sizes to compare in ascending order
padding_nonstimulus = [1800,1650] # to add the frame for waiting time in prey location
# =================== LOW PASS FILTER ========================
order = 3
fs = 300.0  # sample rate, Hz
cutoff = 10  # desired cutoff frequency of the filter, Hz

# ========================= GENERATE ALL THE FILES =================================================================
# Directories for 80 deg visual angle
filename = "Summary of 1st PC vanish dot"
maindir = 'D:\\Semmelhack lab\\002 ANALYSIS\\Vanish dot\\'
dir = maindir + 'vanish\\'
dir_output = dir

eye_files = []
tail_files = []

fishIDs = list()
fishIDs_dir = list()

fishIDs += [fish for fish in os.listdir(dir)]  # store the fish filenames
print fishIDs
fishIDs_dir += [{str(fish): [0, 0], str(fish) +'_bigorsmall': [], str(fish) +'_resptime': [] } for fish in fishIDs] # for storing the right or left information for each fish
neyes = [] # to check if there are missing files
ntail = [] # to check if there are missing files
for fish in fishIDs:

    eyefile = dir + fish + '\\results\\'
    tailfile = dir + fish + '\\tail\\'

    for file in os.listdir(eyefile):
        if file.endswith(".csv"):
            eye_files.append([os.path.join(eyefile, file), dir_output, preypoints, fish, speed, file])
    for file in os.listdir(tailfile):
        if file.endswith(".shv"):
            tail_files.append([os.path.join(tailfile, file), dir_output, preypoints, fish, speed, file])

print len(eye_files), len(tail_files)
print len(fishIDs)

# print fishIDs

prey_positions1 = []
eye_diffs = []
tails = []
rtime = []
one_bout = 0
multi_bout = 0
nbouts = 0
eye_onsets = []
fish_movements = []
right_prey = 0
left_prey = 0
PC = 0
fish_PC = []
prey_side = []
response_time = []
bout_duration = []
prey_size = []
sample_fish = []
mean_binocular = []
binocular = []


for j in range(0, len(eye_files)):
    # ============ PREY LOCATION ==================
    dir_output = str(eye_files[j][1])
    # prey = prey_location_osci(40.0,70.0,-50.0,-10.0,'ccw',5) # with oscillation (speed,p1,p2,p3,dir,times)
    prey_loc = prey_location(float(eye_files[j][4]), float(eye_files[j][2][0]), float(eye_files[j][2][1]), 0)  # without oscillation, (speed,p1,p2,times)
    #first_padding = [preypoints[0]] * padding_nonstimulus[0]
    #end_padding = [preypoints[0]] * padding_nonstimulus[1]
    prey_loc = prey_loc + list(reversed(prey_loc)) + prey_loc
    print 'PREY GOING FROM ', eye_files[j][2][0], ' TO', eye_files[j][2][1]

    print 'ONE DOT'
    smaller = None
    if 'L_' in eye_files[j][5]:
        preyside = 'l'
        prey_loc = list([-int(pr) for pr in prey_loc])
    elif 'R_' in eye_files[j][5]:
        preyside = 'r'
        prey_loc = prey_loc

    print 'PROCESSING: '
    print eye_files[j][0]
    print tail_files[j][0]

    # read the left and right eye tracks
    eyes = eye_reader(str(eye_files[j][0]))
    eyes[0]['LeftEye'] = butter_freq_filter(eyes[0]['LeftEye'], cutoff, fs, order)
    eyes[0]['RightEye'] = butter_freq_filter(eyes[0]['RightEye'], cutoff, fs, order)

    # compute the velocity
    LeftVel = savgol_filter(eyes[0]['LeftEye'], 3, 2, 1, mode = 'nearest')
    RightVel = savgol_filter(eyes[0]['RightEye'], 3, 2, 1, mode = 'nearest')
    eyes_vel = [{'LeftVel': LeftVel, 'RightVel': RightVel}]

    # DETECT THE TAIL BOUTS AND CORRESPONDING EYE MOVEMENTS
    TailEye = extract_tail_eye_bout(eyes, eyes_vel, str(tail_files[j][0]), Fs, bout_thresh, peakthres, delay)

    if not TailEye:
        print 'WARNING!! TailEye empty: ', eye_files[j]
        continue

    if len(TailEye) == 1:
        one_bout += 1
    else:
        multi_bout += 1

    time = range(0, len(TailEye[0]['left']))
    time = [t / tfactor for t in time]
    first_pc_bout = 1 # an ID for first prey capture bout
    pc_per_trial = []
    resp_time_per_trial = []

    for i in range(0, len(TailEye)):
        t1 = TailEye[i]['frames'][0]
        t2 = TailEye[i]['frames'][1]
        if t1 < sensory_delay:
            print "Too early bout"
            continue
        if first_pc_bout == 0:
            print "First prey capture bout already found"
            continue

        print '============= BOUT #', i, '=================='
        if i == 0:
            rtime.append((TailEye[0]['frames'][0] / tfactor))
        print eye_files[j][0]
        print tail_files[j][0]
        print 'Tail bout', TailEye[i]['frames']
        print 'Mean binocular angle', np.mean(TailEye[i]['sum_eyeangles'])
        print 'TAIL BOUT FREQ', TailEye[i]['tailfreq']
        tail = np.mean(TailEye[i]['bout_angles'])

        if np.abs(tail) < 5.0:
            print 'Not strong enough tail movement'
            continue
        # FILTER BASED ON BINOCULAR CONVERGENCE
        if np.mean(TailEye[i]['sum_eyeangles']) < filter_avg_binocular:  # [(mid_list  - tenth):(mid_list + tenth)]) < 30:
            print 'Not converge enough, bout #: ', i, 'Sample: ', TailEye[i]['filename']
            continue

        # FILTER BASED ON LENGTH OF BOUT
        if int(TailEye[i]['frames'][1] - TailEye[i]['frames'][0]) < filter_length_bout:
            print 'TOO SHORT BOUT'
            continue

        right_eyebout = np.mean(TailEye[i]['right_eyeangles'])
        left_eyebout = np.mean(TailEye[i]['left_eyeangles'])

        right_v = (TailEye[i]['right_eyeangles'][-1] - TailEye[i]['right_eyeangles'][0]) / (
                len(TailEye[i]['right_eyeangles']) / tfactor)
        left_v = (TailEye[i]['left_eyeangles'][-1] - TailEye[i]['left_eyeangles'][0]) / (
                len(TailEye[i]['left_eyeangles']) / tfactor)

        # FILTER BASED ON THE VELOCITY
        if right_v <= filter_eye_vel or left_v <= filter_eye_vel:
            print 'Diverging eyes'
            continue

        right_v = (TailEye[i]['right_eyeangles'][-1] - TailEye[i]['right_eyeangles'][0])# / (
                #len(TailEye[i]['right_eyeangles']) / tfactor)
        left_v = (TailEye[i]['left_eyeangles'][-1] - TailEye[i]['left_eyeangles'][0])# / (
                #len(TailEye[i]['left_eyeangles']) / tfactor)

        # FILTER BASED ON THE MAGNITUDE OF EYE DIVERGENCE
        if right_v < filter_eye_diverge or left_v < filter_eye_diverge:
            print 'Big divergence of eye'
            continue

        # get the index, and value of the saccade onset based on velocity
        # Get the velocity maximum peak
        print TailEye[i]['frames'][0], TailEye[i]['frames'][1]
        r_maxpeak, r_max = max(enumerate(TailEye[i]['right_vel_delay']), key=operator.itemgetter(1))
        l_maxpeak, l_max = max(enumerate(TailEye[i]['left_vel_delay']), key=operator.itemgetter(1))

        # Get the minimum peak between the start of the bout to the peak velocity
        # You only want the minimum peak BEFORE the maximum peak
        # There are cases when the maximum peak is found within the first two points
        # one reason is the starting point of the detected tail bout is greatly delayed compared to
        # its corresponding eye movement

        if r_maxpeak == 0:
            r_min = TailEye[i]['right_vel_delay'][0]
            r_minpeak = 0
        else:
            r_minpeak, r_min = min(enumerate(TailEye[i]['right_vel_delay'][0:r_maxpeak]), key=operator.itemgetter(1))

        if l_maxpeak == 0:
            l_min = TailEye[i]['left_vel_delay'][0]
            l_minpeak = 0
        else:
            l_minpeak, l_min = min(enumerate(TailEye[i]['left_vel_delay'][0:l_maxpeak]), key=operator.itemgetter(1))
        '''
        r_sac_on = int(math.ceil((r_maxpeak + r_minpeak) / 2))
        l_sac_on = int(math.ceil((l_maxpeak + l_minpeak) / 2))

        r_sac = TailEye[i]['right_vel_delay'][r_sac_on]
        l_sac = TailEye[i]['left_vel_delay'][l_sac_on]

        right_con = TailEye[i]['right_eyeangles_delay'][r_sac_on]
        left_con = TailEye[i]['left_eyeangles_delay'][l_sac_on]

        right_dir = TailEye[i]['right_eyeangles_delay'][-1] - TailEye[i]['right_eyeangles_delay'][r_sac_on]
        left_dir = TailEye[i]['left_eyeangles_delay'][-1] - TailEye[i]['left_eyeangles_delay'][l_sac_on]


        first_onset = 'none'
        contra_eye = 'none'
        eye_convergence = 0

        if r_max < thresh_saccade_speed and l_max < thresh_saccade_speed:
            print "======WARNING===== THE VELOCITY OF THE ONSET IS LESS THAN 150 deg/s"
            eye_convergence = 1

        # If there's a saccade, use it as the criteria else use eye convergence
        if eye_convergence == 0:
            if r_max < thresh_saccade_speed <= l_max:
                first_onset = 'left'
            elif l_max < thresh_saccade_speed <= r_max:
                first_onset = 'right'
            elif r_sac_on < l_sac_on:
                first_onset = 'right'
            elif l_sac_on < r_sac_on:
                first_onset = 'left'

            eye_onsets.append(first_onset)

        elif eye_convergence == 1:

            if right_eyebout > left_eyebout:
                contra_eye = 'right'
            elif left_eyebout > right_eyebout:
                contra_eye = 'left'
        if tail < -5.0:
            print "TAIL WINS", tail, first_onset, contra_eye, right_eyebout - left_eyebout
            contra_eye = 'right'

        elif tail > 5.0:
            print "TAIL WINS", tail, first_onset, contra_eye, right_eyebout - left_eyebout
            contra_eye = 'left'
        elif first_onset == 'right' or contra_eye == 'right':
            contra_eye = 'right'
            print "EYE WINS", tail, first_onset, contra_eye, right_eyebout - left_eyebout
        elif first_onset == 'left' or contra_eye == 'left':
            print "EYE WINS", tail, first_onset, contra_eye, right_eyebout - left_eyebout
            contra_eye = 'left'
        else:
           prey_pos = 'NONE'

        if contra_eye == 'left':
            convel = LeftVel[t1-100:t1 + 300] # velocity of contra eye
        elif contra_eye == 'right':
            convel = RightVel[t1-100:t1 + 300]
        '''

        if len(eyes[0]['LeftEye']) - t1 < 150:
            continue
        if first_pc_bout == 1:  # first bout
            PC+=1 # is there a prey capture in this trial?
            fish_PC.append(eye_files[j][0][-26:-3])
            #prey_side.append(preyside)
            response_time.append((TailEye[i]['frames'][0])/ 0.3)
            bout_duration.append((TailEye[i]['frames'][1]-TailEye[i]['frames'][0])/0.3)
            first_pc_bout = 0
            prey_size.append(2)
            sample_fish.append(eye_files[j][3])
            mean_binocular.append(np.mean(TailEye[i]['sum_eyeangles']))
            sum_eyeangles = map(add, LeftVel[t1-50:t1+150], RightVel[t1-50:t1+150]) # get the binocular eyeangle
            #sum_eyeangles = map(add, eyes[0]['LeftEye'][t1:t1+600], eyes[0]['RightEye'][t1:t1+600]) # get the binocular eyeangle
            #sum_eyeangles = [float(i - sum_eyeangles[0]) for i in sum_eyeangles]
            sum_eyeangles = [float(i)*1000 for i in sum_eyeangles]
            binocular.append(sum_eyeangles)
            #binocular.append(convel)


    print TailEye[0]['filename']
    print 'done'

print set(sample_fish)
print len(set(sample_fish))
print PC
print np.mean(bout_duration), np.std(bout_duration)
results = izip_longest(fish_PC, prey_side, prey_size, response_time, bout_duration, binocular, mean_binocular)
header = izip_longest(['Fish'], ['Prey Side'], ['Prey Size'], ['Response time'], ['Duration'], ['Binocular'], ['Mean Bino'])

with open(maindir + filename + '.csv', 'wb') as myFile:
    # with open(dir_output + 'Velocity_Acceleration_' + filename + '.csv', 'wb') as myFile:
    wr = csv.writer(myFile, delimiter=',')
    for head in header:
        wr.writerow(head)
    for rows in results:
        wr.writerow(rows)
summary = {"binocular": binocular}
#summary = {"binocular": binocular, "mean_bino": mean_binocular}

pickle_out = open(maindir + filename + '.pickle', 'w')
pickle.dump(summary,pickle_out)
pickle_out.close()
'''
binwidth = 250
n1, bins1, patches1 = plt.hist(response_time, color=[0, 0, 1],edgecolor ='black',
                           bins=np.arange(min(response_time), max(response_time) + binwidth, binwidth))
response_time = [r/1000.0 for r in response_time]

sns.distplot(response_time,rug=True,hist=False)
plt.show()
'''

