from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt

from srsmp.DataHandlers import SpectraData

path_dir = Path("C:/apps/00_MA/data/optical_config_compare/ingest/ThorCCS175")
sd = SpectraData()

sd.load_spectra_file(path_dir / 'spec60_full_Thor.json')
# sd.load_spectra_file(path_dir / 'spec_60_allan_swap.json')

print(sd.class_ids)
data_full = sd.data_per_class[sd.class_ids[0]]['0']
# data_spec60 = sd.data_per_class[sd.class_ids[1]]['0']
wavelengths = np.asarray([[i[0] for i in m[0]] for m in data_full])
wave_mask = np.intersect1d(np.where(wavelengths[0] >= 500),np.where(wavelengths[0] <= 1000))

intensities = np.asarray([[i[1] for i in m[0]] for m in data_full])
# spec60 = np.asarray([[i[1] for i in m[0]] for m in data_spec60])

# intensities_corr = np.divide(intensities[:, wave_mask], spec60[:, wave_mask])
intensities_corr = intensities[:, wave_mask]
standard_deviations = list()

for i in range(10):
    std = np.std(intensities_corr, axis=0)
    standard_deviations.append(std)
    # plt.plot(wavelengths[0], std)
    # plt.show()

    split_intensities = np.dstack((intensities_corr[::2, :], intensities_corr[1::2, :]))
    intensities_corr = np.mean(split_intensities, axis=2)

standard_deviations = np.asarray(standard_deviations)
standard_deviations_corrected = standard_deviations / np.absolute(intensities_corr)

plt.plot(wavelengths[0, wave_mask].transpose(), intensities_corr.transpose())
plt.title('\n'.join(['Mean intensities', path_dir.name]))
plt.grid(which='both')
plt.show()
plt.loglog([2**(i+1) for i in range(10)], np.mean(standard_deviations, axis=1), label='Actual')
plt.loglog([2**(i+1) for i in range(10)], np.mean(standard_deviations, axis=1)[0]*(1/np.sqrt(2))**[i for i in range(10)], label='Theoretical')
plt.title('\n'.join(['Mean Standard deviation', path_dir.name]))
plt.legend()
plt.grid(which='both')
plt.show()
plt.plot(wavelengths[0, wave_mask].transpose(), standard_deviations.transpose())
plt.title('\n'.join(['Standard deviations per wavelength', path_dir.name]))
plt.grid(which='both')
plt.show()
plt.loglog([2**(i+1) for i in range(10)], np.mean(standard_deviations_corrected, axis=1), label='Actual')
plt.loglog([2**(i+1) for i in range(10)], np.mean(standard_deviations_corrected, axis=1)[0]*(1/np.sqrt(2))**[i for i in range(10)], label='Theoretical')
plt.title('\n'.join(['Mean Standard deviation (corrected by mean intensities)', path_dir.name]))
plt.legend()
plt.grid(which='both')
plt.show()
plt.plot(wavelengths[0, wave_mask].transpose(), standard_deviations_corrected.transpose())
plt.title('\n'.join(['Standard deviations per wavelength (corrected by mean intensities)', path_dir.name]))
plt.grid(which='both')
plt.show()

print('Done')
