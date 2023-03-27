from abaqus import *
from abaqusConstants import *
from caeModules import *
from driverUtils import executeOnCaeStartup
executeOnCaeStartup()
import numpy as np
import os
import glob

# Specify the path of the Abaqus input file
abaqus_inp_path='R:\\Abaqus_MAC\\Abaqus_teeter_inp\\teeter.inp'

# Read section properties from file
flap_stiffness = []
lead_stiffness = []
mass = []

with open('R:\\Abaqus_MAC\\Abaqus_teeter_inp\\section_property.txt') as f:
    for i, line in enumerate(f):
        if i >= 40:
            break 
        data = line.split() 
        flap_stiffness.append(float(data[0]))
        lead_stiffness.append(float(data[1]))
        mass.append(float(data[2]))

## Open the part module
mdb.ModelFromInputFile(name='teeter', inputFileName=abaqus_inp_path)
a = mdb.models['teeter'].rootAssembly

# Change section density
for i in range(1, 41):
    section_name = 'Section-' + str(i) + '-SET-' + str(i)
    section_density = mass[i-1] * 1e-13
    mdb.models['teeter'].sections[section_name].setValues(poissonRatio=0.0, 
        density=section_density, thermalExpansion=ON, temperatureDependency=OFF, 
        dependencies=0, table=((1.0, 1.0, 0.0), ), alphaDamping=0.0, 
        betaDamping=0.0, compositeDamping=0.0, centroid=(0.0, 0.0), shearCenter=(
        0.0, 0.0))

# Change section stiffness
for i in range(1, 41):
    section_name = 'Profile-' + str(i)
    mdb.models['teeter'].profiles[section_name].setValues(
        i11=flap_stiffness[i-1]*1e6, i22=lead_stiffness[i-1]*1e6)

## Define loads for 700 RPM
mdb.models['teeter'].loads['CENTRIF-1'].setValues(magnitude=73.3)

## Add keywords in Abaqus/CAE to output point displacements
mdb.models['teeter'].keywordBlock.synchVersions(storeNodesAndElements=False)
mdb.models['teeter'].keywordBlock.replace(236, """
*Output, field, variable=PRESELECT
*NODE PRINT, Nset=JIANCE, FREQUENCY=1
U
*NODE FILE""")

## Create a job for 700 RPM
myJob = mdb.Job(name='teeter_700', model='teeter', description='', type=ANALYSIS, 
    atTime=None, waitMinutes=0, waitHours=0, queue=None, memory=90, 
    memoryUnits=PERCENTAGE, getMemoryFromAnalysis=True, 
    explicitPrecision=SINGLE, nodalOutputPrecision=SINGLE, echoPrint=OFF, 
    modelPrint=OFF, contactPrint=OFF, historyPrint=OFF, userSubroutine='', 
    scratch='', resultsFormat=ODB, multiprocessingMode=DEFAULT, numCpus=4, 
    numDomains=4, numGPUs=0)

## Submit the job for 700 RPM
myJob.submit()
myJob.waitForCompletion()

## Define loads for 825 RPM
mdb.models['teeter'].loads['CENTRIF-1'].setValues(magnitude=86.4)

## Create a job for 950 RPM
myJob = mdb.Job(name='teeter_825', model='teeter', description='', type=ANALYSIS, 
    atTime=None, waitMinutes=0, waitHours=0, queue=None, memory=90, 
    memoryUnits=PERCENTAGE, getMemoryFromAnalysis=True, 
    explicitPrecision=SINGLE, nodalOutputPrecision=SINGLE, echoPrint=OFF, 
    modelPrint=OFF, contactPrint=OFF, historyPrint=OFF, userSubroutine='', 
    scratch='', resultsFormat=ODB, multiprocessingMode=DEFAULT, numCpus=4, 
    numDomains=4, numGPUs=0)

## Submit the job for 825 RPM
myJob.submit()
myJob.waitForCompletion()

## Define loads for 950 RPM
mdb.models['teeter'].loads['CENTRIF-1'].setValues(magnitude=99.5)

## Create a job for 950 RPM
myJob = mdb.Job(name='teeter_950', model='teeter', description='', type=ANALYSIS, 
    atTime=None, waitMinutes=0, waitHours=0, queue=None, memory=90, 
    memoryUnits=PERCENTAGE, getMemoryFromAnalysis=True, 
    explicitPrecision=SINGLE, nodalOutputPrecision=SINGLE, echoPrint=OFF, 
    modelPrint=OFF, contactPrint=OFF, historyPrint=OFF, userSubroutine='', 
    scratch='', resultsFormat=ODB, multiprocessingMode=DEFAULT, numCpus=4, 
    numDomains=4, numGPUs=0)

## Submit the job for 950 RPM
myJob.submit()
myJob.waitForCompletion()

# extract frequencys from dat files
dat_path_three_jobs = ['R:/temp/teeter_700.dat','R:/temp/teeter_825.dat','R:/temp/teeter_950.dat']
RPMs = [700, 825, 950]
flap_couple_mode_all_conditions = []
flap_periodic_mode_all_conditions = []
lead_mode_all_conditions = []

for path in range(3):

    data_path = dat_path_three_jobs[path]

    # open dat file
    with open(data_path, 'r') as f:
        lines = f.readlines()

    k = []  # initialize list to store polynomial coefficients
    flap_couple_modes = []  # srore flap couple modes
    flap_periodic_modes = [] # store flap periodic modes
    lead_modes = [] # store lead modes
    freq_num = 0 # initialize frequency number

    # loop through each instance of 'N O D E   O U T P U T'
    for i, line in enumerate(lines):
        if 'N O D E   O U T P U T' in line:
            start_index = i + 1
            freq_num = freq_num + 1

            # extract data lines
            data_lines = lines[start_index+9:start_index+89]
            # split lines
            disp = []
            for line in data_lines:
                split_line = line.split()
                if len(split_line) == 7:
                    disp.append(list(map(float, split_line)))

            # find max
            max_val = -float('inf')
            max_col = -1
            for j in range(1, 4):
                col_max = max([abs(row[j]) for row in disp])
                if col_max > max_val:
                    max_val = col_max
                    max_col = j

            #compare with maximum and minium value
            max_value = max(disp[i][j] for i in range(80) for j in range(1, 4))
            min_value = min(disp[i][j] for i in range(80) for j in range(1, 4))
            if abs(max_value) < abs(min_value):
                max_val = -max_val

            # Extracting the maximum displacement from the dat file.
            dis_single_right = [row[max_col] for row in disp[:40]]
            dis_single_left =  [row[max_col] for row in disp[40:80]]
            dis_single_left. reverse()
            zero = [0]
            dis_two = dis_single_left + zero + dis_single_right
            for d in range(len(dis_two)):
                dis_two[d] /= max_val

            # determine which mode belongs
            if max_col == 2 :
                if abs(dis_two[0] - dis_two[-1]) < 0.3 :
                    flap_couple_modes.append(freq_num)
                else :
                    flap_periodic_modes.append(freq_num)
            else :
                lead_modes.append(freq_num)

    # Reading the dat file and extracting the frequencies from the dat file.
    with open(data_path, 'r') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            if 'E I G E N V A L U E    O U T P U T  ' in line:
                lines = lines[i+6:i+26]
                freqs = [float(line.split()[3]) for line in lines]

    # Extracting the frequencies from the dat file and then adding 0's to the list if the length of the
    # list is less than 5 or 6.
    values_flap_couple_modes = [freqs[i-1] for i in flap_couple_modes]
    values_flap_periodic_modes = [freqs[i-1] for i in flap_periodic_modes]
    values_lead_modes = [freqs[i-1] for i in lead_modes]

    values_flap_couple_modes += [0] * (5 - len(values_flap_couple_modes))
    values_flap_periodic_modes += [0] * (5 - len(values_flap_periodic_modes))
    values_lead_modes += [0] * (6 - len(values_lead_modes))

    # Replacing the 0 values with the maximum value in the list.
    max_value = max(values_flap_couple_modes)
    for i in range(len(values_flap_couple_modes)):
       if values_flap_couple_modes[i] == 0:
           values_flap_couple_modes[i] = max_value
    max_value = max(values_flap_periodic_modes)
    for i in range(len(values_flap_periodic_modes)):
       if values_flap_periodic_modes[i] == 0:
           values_flap_periodic_modes[i] = max_value
    max_value = max(values_lead_modes)
    for i in range(len(values_lead_modes)):
       if values_lead_modes[i] == 0:
           values_lead_modes[i] = max_value

    flap_couple_mode_all_conditions.append(values_flap_couple_modes)
    flap_periodic_mode_all_conditions.append(values_flap_periodic_modes)
    lead_mode_all_conditions.append(values_lead_modes)
    
    # Reading the total mass of the model from the dat file.
    with open(data_path, 'r') as f:
        for line in f:
            if 'TOTAL MASS OF MODEL' in line:
                next_line = next(f) 
                next_line = next(f) 
                totalmass = float(next_line.strip()) 
                break

    # Writing the data to a file.
    data = {
        'flap_couple_modes': values_flap_couple_modes[:5],
        'flap_periodic_modes': values_flap_periodic_modes[:5],
        'lead_modes': values_lead_modes[:6]
    }

    name = 'freq_sorts_' + str(RPMs[path]) +'.txt'

    with open(name, 'w') as f:
        f.write(str(totalmass)+'\n')
        for key, value in data.items():
            f.write(key + '\n')
            for v in value:
                f.write(str(v) + '\n')

# Avoid frequencys, only the flap_couple_modes and flap_periodic_modes
flap_couple_mode_avoid = [[1.7, 2.3], [3.7, 4.3], [5.7, 6.3], [7.7, 8.3], [9.7, 10.3], [11.7, 12.3], [13.7, 14.3]]
flap_periodic_mode_avoid = [[2.7, 3.3], [4.7, 5.3], [6.7, 7.3], [8.7, 9.3], [10.7, 11.3], [12.7, 13.3], [14.7, 15.3]]
# lead_mode_avoid = [[1.7, 2.3], [2.7, 3.3], [3.7, 4.3], [4.7, 5.3], [5.7, 6.3], [6.7, 7.3], [7.7, 8.3], [8.7, 9.3], [9.7, 10.3], [10.7, 11.3], [12.7, 13.3], [13.7, 14.3]]

# Finding the intersection of the curve and the line.
# Caution, minimum solotion to solve intersect_x. but not asure if its right
def distance_intersection(mode_avoid, mode_all_conditions):
    intersect_x = []
    for j in range(len(mode_all_conditions[0])):  # nuubers for mode_all_conditions 5
        for k in range(len(mode_avoid)): #numbers for avoid frequencys 7
            flap_modes_different_RPM = np.array(mode_all_conditions)[:,j]
            a, b, c = np.polyfit(RPMs, flap_modes_different_RPM, 2)
            b_avoid_lower = mode_avoid[k][0] / 60
            b_avoid_upper = mode_avoid[k][1] / 60
            det_lower = (b - b_avoid_lower)**2 - 4 * a * c
            det_upper = (b - b_avoid_upper)**2 - 4 * a * c
            x = [0, 0]
            if det_lower >= 0 :
                x[0] = (-(b - b_avoid_lower) - np.sqrt(det_lower)) / 2 / a
            if det_upper >= 0 :
                x[1] = (-(b - b_avoid_upper) - np.sqrt(det_upper)) / 2 / a
            intersect_x.append(x)
    
    # Calculating the distance between the target interval and the other intervals.
    target_interval = [700, 950]
    distance_list = []
    normalized_distance_list = []

    for interval in intersect_x:
        end, start = interval
        if end < target_interval[0]: # interval is left of target
            distance = target_interval[0] - end
            distance_list.append(distance)
        elif start > target_interval[1]:  # interval is right of target
            distance = start - target_interval[1]
            distance_list.append(distance) 
        else: # interval is overlap with target
            overlap_start = max(start, target_interval[0])
            overlap_end = min(end, target_interval[1])
            overlap_size = overlap_end - overlap_start
            distance = -overlap_size
            distance_list.append(distance)

    # Normalizing the distances
    max_val = max(distance_list)
    min_val = min(distance_list)
    distance_normailzi_list = [abs(val/max_val) if val >= 0 else val/abs(min_val) for val in distance_list]
    
    return distance_list, distance_normailzi_list

# Calculating the distance between the target interval and the other intervals using the distance_intersection function.
distance_normailze_list_flap_couple, distance_list_flap_couple = distance_intersection(flap_couple_mode_avoid, flap_couple_mode_all_conditions)
distance_normailze_list_flap_periodic, distance_list_flap_periodic= distance_intersection(flap_periodic_mode_avoid, flap_periodic_mode_all_conditions)

# Reshaping the list into a 7x5 matrix.
distance_list_flap_couple_reshape = np.array(distance_normailze_list_flap_couple).reshape(5, 7).T.tolist()
distance_list_flap_periodic_reshape = np.array(distance_normailze_list_flap_periodic).reshape(5, 7).T.tolist()
distance_normalize_list_flap_couple_reshape = np.array(distance_list_flap_couple).reshape(5, 7).T.tolist()
distance_normalize_list_flap_periodic_reshape = np.array(distance_list_flap_periodic).reshape(5, 7).T.tolist()

# Write distance_list to a text file.
with open('distance_list_reshape.txt', 'w') as f:
    f.write('total_mass:\n')
    f.write(str(totalmass)+'\n')
    f.write('distance_list_flap_couple_reshape:\n')
    for row in distance_list_flap_couple_reshape:
        f.write('\t'.join('{:10.3f}'.format(e) for e in row) + '\n')

    f.write('\ndistance_list_flap_periodic_reshape:\n')
    for row in distance_list_flap_periodic_reshape:
        f.write('\t'.join('{:10.3f}'.format(e) for e in row) + '\n')

with open('distance_normalize_list.txt', 'w') as f:
    f.write('total_mass:\n')
    f.write(str(totalmass)+'\n')
    f.write('distance_normalize_list_flap_couple:\n')
    for row in distance_normalize_list_flap_couple_reshape:
        f.write('\t'.join('{:.3f}'.format(e) for e in row) + '\n')

    f.write('\ndistance_normalize_list_flap_periodic:\n')
    for row in distance_normalize_list_flap_periodic_reshape:
        f.write('\t'.join('{:.3f}'.format(e) for e in row) + '\n')
        
# End of the code.
print('MISSION COMPLETE')

# Remove .odb file
folder_path = '.'

odb_files = glob.glob(os.path.join(folder_path, '*.odb'))

for file_path in odb_files:
    os.remove(file_path)