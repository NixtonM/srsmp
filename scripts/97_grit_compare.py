from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt

from srsmp.DataHandlers import SpectraData

path_dir = Path("C:/apps/00_MA/data/ingest/FlameNIR")
sd = SpectraData()

sd.load_reference_file(path_dir / 'reference.json')
sd.load_spectra_file(path_dir / 'grit_test.json')


data = sd.data_per_class['grit_test']['1']
data2 = sd.data_per_class['grit_test']['2']

references = {}
intensities = []
for m in data:
    ref_key = m[3]
    for k in ref_key:
        if k not in references:
            references[k] = np.asarray([inten[1] for inten in sd.data_reference[k][0]])
    intensities.append(np.asarray([l[1] for l in m[0]]) / references[m[3][1]])
intensities2 = []
for m in data2:
    ref_key = m[3]
    for k in ref_key:
        if k not in references:
            references[k] = np.asarray([inten[1] for inten in sd.data_reference[k][0]])
    intensities2.append(np.asarray([l[1] for l in m[0]]) / references[m[3][1]])

# intensities = [[i[1] for i in m[0]] for m in data]
wavelengths = [[i[0] for i in m[0]] for m in data]

integration_time = [m[1] for m in data]

# intensities2 = [[i[1] for i in m[0]] for m in data2]
wavelengths2 = [[i[0] for i in m[0]] for m in data2]

integration_time2 = [m[1] for m in data2]

intensities = np.asarray(intensities)
intensities2 = np.asarray(intensities2)
wavelengths = np.asarray(wavelengths)



grit40 = intensities[:10, :].mean(axis=0)
grit80 = intensities[10:20, :].mean(axis=0)
grit120 = intensities[20:30, :].mean(axis=0)
grit240 = intensities[30:40, :].mean(axis=0)
unsanded = intensities2.mean(axis=0)

grit_mean = np.concatenate((intensities,intensities2)).mean(axis=0)

delta_40 = grit_mean - grit40
delta_80 = grit_mean - grit80
delta_120 = grit_mean - grit120
delta_240 = grit_mean - grit240
delta_unsanded = grit_mean - unsanded

delta_mean40 = np.outer(np.ones(10), grit40) - intensities[:10, :]
delta_mean80 = np.outer(np.ones(10), grit80) - intensities[10:20, :]
delta_mean120 = np.outer(np.ones(10), grit120) - intensities[20:30, :]
delta_mean240 = np.outer(np.ones(10), grit240) - intensities[30:40, :]
delta_meanunsanded = np.outer(np.ones(10), unsanded) - intensities2
# delta_mean80 = (grit80.reshape((1, :)) - intensities[10:20].transpose()).transpose()
# delta_mean120 = (grit120.reshape((1, :))- intensities[20:30].transpose()).transpose()
# delta_mean240 = (grit240.reshape((1, :)) - intensities[30:].transpose()).transpose()

plt.plot(wavelengths[0], delta_40, label='40 grit')
plt.plot(wavelengths[0], delta_80, label='80 grit')
plt.plot(wavelengths[0], delta_120, label='120 grit')
plt.plot(wavelengths[0], delta_240, label='240 grit')
plt.plot(wavelengths[0], delta_unsanded, label='Untreated')

plt.legend()
plt.show()

plt.plot(wavelengths[0].transpose(), delta_mean40.transpose())
plt.show()
plt.plot(wavelengths[0].transpose(), delta_mean80.transpose())
plt.show()
plt.plot(wavelengths[0].transpose(), delta_mean120.transpose())
plt.show()
plt.plot(wavelengths[0].transpose(), delta_mean240.transpose())
plt.show()
plt.plot(wavelengths[0].transpose(), delta_meanunsanded.transpose())
plt.show()



print(1)