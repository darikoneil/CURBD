import numpy as np

from curbd import curbd

from matplotlib import pyplot as plt

sim = curbd.threeRegionSim(number_units=100)

activity = np.concatenate((sim['Ra'], sim['Rb'], sim['Rc']), 0)

Na = sim['params']['Na']
Nb = sim['params']['Nb']
Nc = sim['params']['Nc']

regions = []
regions.append(['Region A', np.arange(0, Na)])
regions.append(['Region B', np.arange(Na, Na + Nb)])
regions.append(['Region C', np.arange(Na + Nb, Na + Nb + Nc)])
regions = np.array(regions, dtype=object)

model = curbd.trainMultiRegionRNN(activity,
                                  dtData=sim['params']['dtData'],
                                  dtFactor=5,
                                  regions=regions,
                                  tauRNN=2*sim['params']['tau']/2,
                                  nRunTrain=500,
                                  verbose=True,
                                  nRunFree=5,
                                  plotStatus=False)

[curbd_arr, curbd_labels] = curbd.computeCURBD(model)

n_regions = 3
n_region_units = 100

fig, ax = plt.subplots(n_regions, n_regions, figsize=[8, 8])

for i in range(n_regions):
    for j in range(n_regions):
        ax[i, j].pcolormesh(model['tRNN'], range(n_region_units),
                            curbd_arr[i, j])
        ax[i, j].set_xlabel('Time (s)')
        ax[i, j].set_ylabel('Neurons in {}'.format(regions[i, 0]))
        ax[i, j].set_title(curbd_labels[i, j])
        ax[i, j].title.set_fontsize(8)
        ax[i, j].xaxis.label.set_fontsize(8)
        ax[i, j].yaxis.label.set_fontsize(8)