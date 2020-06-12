from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt

from srsmp.DataHandlers import SpectraData

path_dir = Path("C:/apps/00_MA/data/ingest/ThorCCS175")
sd = SpectraData()

sd.load_reference_file(path_dir / 'reference.json')
sd.load_spectra_file(path_dir / 'sample_5_untreated.json')
sd.load_spectra_file(path_dir / 'sample_5_treated.json')
sd.load_spectra_file(path_dir / 'sample_5_noise.json')

data_untreated = sd.data_per_class['sample_5_untreated']['40']
data_treated = sd.data_per_class['sample_5_treated']['40']
data_untreated = sd.data_per_class['sample_5_noise']['0']
wavelengths = [[i[0] for i in m[0]] for m in data_untreated]

# references = {}
# for m in data_untreated:
#     ref_key = m[3]
#     for k in ref_key:
#         if k not in references:
#             references[k] = np.asarray([inten[1] for inten in sd.data_reference[k][0]])
# for m in data_treated:
#     ref_key = m[3]
#     for k in ref_key:
#         if k not in references:
#             references[k] = np.asarray([inten[1] for inten in sd.data_reference[k][0]])

points_untreated = list()
for i in range(1):
    pt = {'intensities': np.asarray([[i[1] for i in m[0]] for m in data_untreated[i * 10:(i + 1) * 10]]),
          'wavelengths': [[i[0] for i in m[0]] for m in data_untreated[i * 10:(i + 1) * 10]]}
    pt['mean'] = np.mean(pt['intensities'], 0)
    pt['std'] = np.std(pt['intensities'], 0)
    # pt['corrected_intensities'] = pt['intensities']/ references[data_untreated[i * 10][3][0]][None, :]
    # pt['corrected_mean'] = pt['mean']/references[data_untreated[i * 10][3][0]]

    points_untreated.append(pt)

    pt['vs_first'] = pt['mean'] - points_untreated[0]['mean']
    plt.plot(pt['wavelengths'][0], pt['mean'])
    plt.show()
    plt.plot(pt['wavelengths'][0], pt['std'])
    plt.show()
    plt.plot(pt['wavelengths'][0], pt['corrected_mean'])
    plt.show()
    plt.plot(pt['wavelengths'][0], pt['vs_first'])
    plt.show()


points_treated = list()
for i in range(5):
    pt = {'intensities': np.asarray([[i[1] for i in m[0]] for m in data_treated[i * 10:(i + 1) * 10]]),
          'wavelengths': [[i[0] for i in m[0]] for m in data_treated[i * 10:(i + 1) * 10]]}
    pt['mean'] = np.mean(pt['intensities'], 0)
    pt['std'] = np.std(pt['intensities'], 0)
    pt['corrected_intensities'] = pt['intensities'] / references[data_treated[i * 10][3][0]][None, :]
    pt['corrected_mean'] = pt['mean'] / references[data_treated[i * 10][3][0]]

    points_treated.append(pt)

    pt['vs_first'] = pt['mean'] - points_untreated[0]['mean']
    plt.plot(pt['wavelengths'][0], pt['mean'], 'r--')
    plt.show()
    plt.plot(pt['wavelengths'][0], pt['std'], 'r--')
    plt.show()
    plt.plot(pt['wavelengths'][0], pt['corrected_mean'], 'r--')
    plt.show()
    plt.plot(pt['wavelengths'][0], pt['vs_first'], 'r--')
    plt.show()



print(data_untreated)
