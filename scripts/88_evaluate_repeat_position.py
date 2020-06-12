from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt

from srsmp.DataHandlers import SpectraData


def repeat_position_evaluation(path_dir, sample_name, boundaries=None):
    colors = ['b', 'r', 'g']
    labels = ['Full', '75%', '25%']

    sd = SpectraData()
    intensities = list()
    standard_deviations = list()
    standard_deviations_corrected = list()
    sd.load_spectra_file(path_dir / (sample_name + '.json'))
    sd.load_spectra_file(path_dir / (sample_name + '_spec60.json'))

    sample_data = sd.data_per_class[sample_name]

    wavelengths = np.asarray([[i[0] for i in m[0]] for m in sample_data['0']])[0]
    sample_intensities = list()
    for measurements in sample_data.values():
        sample_intensities.append(np.average(np.asarray([[i[1] for i in m[0]] for m in measurements]), axis=0))
    sample_intensities = np.asarray(sample_intensities)
    if boundaries is not None:
        wave_mask = np.intersect1d(np.where(wavelengths >= boundaries[0]), np.where(wavelengths <= boundaries[1]))
    else:
        wave_mask = range(wavelengths.shape[0])
    wavelengths = wavelengths[wave_mask]

    sample_intensities = sample_intensities[:, wave_mask]

    ref_data = sd.data_per_class[sample_name]

    ref_intensities = list()
    for measurements in ref_data.values():
        ref_intensities.append(np.average(np.asarray([[i[1] for i in m[0]] for m in measurements]), axis=0))
    ref_intensities = np.asarray(ref_intensities)
    ref_intensities = ref_intensities[:, wave_mask]

    corrected_intensities = list()
    for i, inten in enumerate(sample_intensities):
        corrected_intensities.append(inten / np.average(ref_intensities[i:i+2, :], axis=0))
    corrected_intensities = np.asarray(corrected_intensities)

    delta_intensities = sample_intensities[1:, :] - sample_intensities[0, :]

    plt.plot(wavelengths, sample_intensities.transpose())
    plt.title(sample_name+': '+'Intensities')
    plt.show()

    plt.plot(wavelengths, delta_intensities.transpose())
    plt.title(sample_name+': '+'Difference to First measurement')
    plt.show()

    plt.plot(wavelengths, corrected_intensities.transpose())
    plt.title(sample_name+': '+'Intensities corrected by reference')
    plt.show()


def before_after_evaluation(path_dir, sample_name_before, sample_name_after, boundaries=None):
    colors = ['b', 'r', 'g']
    labels = ['Full', '75%', '25%']

    sd = SpectraData()
    sd.load_spectra_file(path_dir / (sample_name_before + '.json'))
    sd.load_spectra_file(path_dir / (sample_name_after + '.json'))
    sd.load_spectra_file(path_dir / (sample_name_before + '_spec60.json'))
    sd.load_spectra_file(path_dir / (sample_name_after + '_spec60.json'))

    sample_data_before = sd.data_per_class[sample_name_before]

    wavelengths = np.asarray([[i[0] for i in m[0]] for m in sample_data_before['0']])[0]
    sample_intensities = list()
    for measurements in sample_data_before.values():
        sample_intensities.append(np.average(np.asarray([[i[1] for i in m[0]] for m in measurements]), axis=0))
    sample_intensities = np.asarray(sample_intensities)
    if boundaries is not None:
        wave_mask = np.intersect1d(np.where(wavelengths >= boundaries[0]), np.where(wavelengths <= boundaries[1]))
    else:
        wave_mask = range(wavelengths.shape[0])
    wavelengths = wavelengths[wave_mask]

    sample_intensities_before = sample_intensities[:, wave_mask]

    ref_data = sd.data_per_class[sample_name_before]

    ref_intensities = list()
    for measurements in ref_data.values():
        ref_intensities.append(np.average(np.asarray([[i[1] for i in m[0]] for m in measurements]), axis=0))
    ref_intensities = np.asarray(ref_intensities)
    ref_intensities_before = ref_intensities[:, wave_mask]

    corrected_intensities = list()
    for i, inten in enumerate(sample_intensities_before):
        corrected_intensities.append(inten / np.average(ref_intensities_before[i:i+2, :], axis=0))
    corrected_intensities_before = np.asarray(corrected_intensities)

    delta_intensities_before = sample_intensities_before[1:, :] - sample_intensities_before[0, :]

    sample_data_after = sd.data_per_class[sample_name_after]

    sample_intensities = list()
    for measurements in sample_data_after.values():
        sample_intensities.append(np.average(np.asarray([[i[1] for i in m[0]] for m in measurements]), axis=0))
    sample_intensities = np.asarray(sample_intensities)

    sample_intensities_after = sample_intensities[:, wave_mask]

    ref_data = sd.data_per_class[sample_name_after]

    ref_intensities = list()
    for measurements in ref_data.values():
        ref_intensities.append(np.average(np.asarray([[i[1] for i in m[0]] for m in measurements]), axis=0))
    ref_intensities = np.asarray(ref_intensities)
    ref_intensities_after = ref_intensities[:, wave_mask]

    corrected_intensities = list()
    for i, inten in enumerate(sample_intensities_after):
        corrected_intensities.append(inten / np.average(ref_intensities_after[i:i + 2, :], axis=0))
    corrected_intensities_after = np.asarray(corrected_intensities)

    delta_intensities_after = sample_intensities_after[1:, :] - sample_intensities_after[0, :]

    plt.plot(wavelengths, sample_intensities_before.transpose())
    plt.title(sample_name_before+': '+'Intensities')
    plt.show()
    plt.plot(wavelengths, sample_intensities_after.transpose())
    plt.title(sample_name_after+': '+'Intensities')
    plt.show()

    plt.plot(wavelengths, delta_intensities_before.transpose())
    plt.title(sample_name_before+': '+'Difference to First measurement')
    plt.show()
    plt.plot(wavelengths, delta_intensities_after.transpose())
    plt.title(sample_name_after+': '+'Difference to First measurement')
    plt.show()

    plt.plot(wavelengths, corrected_intensities_before.transpose())
    plt.title(sample_name_before + ': ' + 'Intensities corrected by reference')
    plt.show()
    plt.plot(wavelengths, corrected_intensities_after.transpose())
    plt.title(sample_name_after+': '+'Intensities corrected by reference')
    plt.show()

    plt.plot(wavelengths, (corrected_intensities_after - corrected_intensities_before).transpose())
    plt.title(sample_name_after+': '+'Delta Before/After Intensities corrected by reference')
    plt.show()


if __name__ == '__main__':
    path_dir = Path("C:/apps/00_MA/data/ingest/ThorCCS175")
    # sample_name = 'sample47_pos1_before'
    #     # repeat_position_evaluation(path_dir, sample_name, boundaries=(500, 1000))
    before_after_evaluation(path_dir, 'sample47_pos1_before', 'sample47_pos1_after', boundaries=(500, 1000))
    # # sample_name = 'sample47_pos2_before'
    # # repeat_position_evaluation(path_dir, sample_name, boundaries=(500, 1000))
    # # sample_name = 'sample47_pos3_before'
    # # repeat_position_evaluation(path_dir, sample_name, boundaries=(500, 1000))
    # # sample_name = 'sample47_pos4_before'
    # # repeat_position_evaluation(path_dir, sample_name, boundaries=(500, 1000))
    # # sample_name = 'sample47_pos5_before'
    # # repeat_position_evaluation(path_dir, sample_name, boundaries=(500, 1000))
    # # sample_name = 'sample47_pos6_before'
    # # repeat_position_evaluation(path_dir, sample_name, boundaries=(500, 1000))
    before_after_evaluation(path_dir, 'sample47_pos5_before', 'sample47_pos5_after', boundaries=(500, 1000))
    before_after_evaluation(path_dir, 'sample47_pos6_before', 'sample47_pos6_after', boundaries=(500, 1000))
    path_dir = Path("C:/apps/00_MA/data/ingest/FlameNIR")
    # sample_name = 'sample47_pos1_before'
    # repeat_position_evaluation(path_dir, sample_name)
    before_after_evaluation(path_dir, 'sample47_pos1_before', 'sample47_pos1_after')
    # sample_name = 'sample47_pos2_before'
    # repeat_position_evaluation(path_dir, sample_name)
    # sample_name = 'sample47_pos3_before'
    # repeat_position_evaluation(path_dir, sample_name)
    # sample_name = 'sample47_pos4_before'
    # repeat_position_evaluation(path_dir, sample_name)
    before_after_evaluation(path_dir, 'sample47_pos5_before', 'sample47_pos5_after')
    before_after_evaluation(path_dir, 'sample47_pos6_before', 'sample47_pos6_after')
    # sample_name = 'sample47_pos5_before'
    # repeat_position_evaluation(path_dir, sample_name)
    # sample_name = 'sample47_pos6_before'
    # repeat_position_evaluation(path_dir, sample_name)
    print('Done')
