from srsmp.DataHandlers import SpectraData

from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import ScalarFormatter
plt.style.use('thesis_2col')

def allan(path_dir, list_files, boundaries=None):
    colors = ['b', 'r', 'g']
    labels = ['Full', '75%', '25%']

    spectrometer_name = path_dir.name
    plot_dir = Path('./plots/2col/allan/')
    plot_dir.mkdir(parents=True, exist_ok=True)


    sd = SpectraData()
    intensities = list()
    standard_deviations = list()
    standard_deviations_corrected = list()
    for f in list_files:
        f: Path
        sd.load_spectra_file(path_dir / f)
        d = sd.data_per_class[f.stem]['0']
        wavelengths = np.asarray([[i[0] for i in m[0]] for m in d])
        inten = np.asarray([[i[1] for i in m[0]] for m in d])
        if boundaries is not None:
            wave_mask = np.intersect1d(np.where(wavelengths[0] >= boundaries[0]), np.where(wavelengths[0] <= boundaries[1]))
            intensities.append(inten[:, wave_mask])
        else:
            wave_mask = range(wavelengths.shape[1])
            intensities.append(inten)
        std_run = list()
        for i in range(10):
            std = np.std(intensities[-1], axis=0)
            std_run.append(std)
            split_intensities = np.dstack((intensities[-1][::2, :], intensities[-1][1::2, :]))
            intensities[-1] = np.mean(split_intensities, axis=2)
        standard_deviations.append(np.asarray(std_run))
        standard_deviations_corrected.append(standard_deviations[-1] / np.absolute(intensities[-1]))


    for i, inten in enumerate(intensities):
        plt.plot(wavelengths[0, wave_mask].transpose(), inten.transpose(), colors[i]+'-', label=labels[i])
    # plt.plot(wavelengths[0, wave_mask].transpose(), intensities[0].transpose() * 0.75, colors[1]+':', label=labels[1]+' (T)')
    # plt.plot(wavelengths[0, wave_mask].transpose(), intensities[0].transpose() * 0.25, colors[2]+':', label=labels[2]+' (T)')
    plt.xlabel('Wavelength [nm]')
    plt.ylabel('Intensity')
    t = ' - '.join([path_dir.name, 'Average intensities (1024 measurements)'])
    plt.title(t)
    plt.legend(loc='center left', bbox_to_anchor=(1.02, 0.5))
    plt.savefig(plot_dir / (spectrometer_name + '01_mean_intensities'))
    plt.close()

    for i, inten in enumerate(intensities):
        plt.plot(wavelengths[0, wave_mask].transpose(), inten.transpose()/np.mean(inten), colors[i]+'-', label=labels[i])
    plt.xlabel('Wavelength [nm]')
    plt.ylabel('Intensity')
    t = ' - '.join([path_dir.name, 'Average intensities (1024 measurements)\nscaled by mean intensity'])
    plt.title(t)
    plt.legend(loc='center left', bbox_to_anchor=(1.02, 0.5))
    plt.savefig(plot_dir / (spectrometer_name + '01b_mean_intensities_scaled'))
    plt.close()

    for j, std_run in enumerate(standard_deviations):
        plt.loglog([(2 ** (i))*0.5 for i in range(10)], np.mean(std_run, axis=1), colors[j]+'-', label=labels[j]+' (A)')
        plt.loglog([(2 ** (i))*0.5 for i in range(10)],
                   np.mean(std_run, axis=1)[0] * (1 / np.sqrt(2)) ** [i for i in range(10)], colors[j]+'--',
                   label=labels[j]+' (T)')
    plt.gca().xaxis.set_major_formatter(ScalarFormatter())
    # plt.xticks(ticks=[0.5, 1, 5, 10, 50, 100, 500])
    plt.xlabel('Integration time [s]')
    plt.ylabel('Standard deviation over all wavelengths')
    t = ' - '.join([path_dir.name, 'Standard deviation vs. integration time'])
    plt.title(t)
    plt.legend(loc='center left', bbox_to_anchor=(1.02, 0.5))
    plt.savefig(plot_dir / (spectrometer_name + '02_std_integration_time'))
    plt.close()

    for i, std_run_corr in enumerate(standard_deviations_corrected):
        plt.plot(wavelengths[0, wave_mask].transpose(), std_run_corr[0, :].transpose(), colors[i]+'-', label=labels[i])
    plt.xlabel('Wavelength [nm]')
    plt.ylabel('Standard deviation')
    t = ' - '.join([path_dir.name, 'Standard deviations vs integration time'])
    plt.title(t)
    plt.legend(loc='center left', bbox_to_anchor=(1.02, 0.5))
    plt.savefig(plot_dir / (spectrometer_name + '03_std_integration_time_per_wavelength'))
    plt.close()

    for j, std_run_corr in enumerate(standard_deviations_corrected):
        plt.loglog([(2 ** (i))*0.5 for i in range(10)], np.mean(std_run_corr, axis=1), colors[j]+'-', label=labels[j]+' (A)')
        plt.loglog([(2 ** (i))*0.5 for i in range(10)],
                   np.mean(std_run_corr, axis=1)[0] * (1 / np.sqrt(2)) ** [i for i in range(10)], colors[j]+'--',
                   label=labels[j]+' (T)')
    plt.gca().xaxis.set_major_formatter(ScalarFormatter())
    t = ' - '.join([path_dir.name, 'Standard deviation\n(corrected by signal power)'])
    plt.title(t)
    plt.legend(loc='center left', bbox_to_anchor=(1.02, 0.5))
    plt.xlabel("Integration time [s]")
    plt.ylabel('Standard deviation over all wavelengths')
    plt.savefig(plot_dir / (spectrometer_name + '04_mean_std_deviation_corrected'))
    plt.close()

    for i, std_run_corr in enumerate(standard_deviations_corrected):
        plt.plot(wavelengths[0, wave_mask].transpose(), std_run_corr[0, :].transpose(), colors[i]+'-', label=labels[i])
    t = ' - '.join([path_dir.name, 'Standard deviations per wavelength\n(corrected by mean intensities)'])
    plt.title(t)
    plt.xlabel('Wavelength [nm]')
    plt.ylabel('Standard deviation')
    plt.legend(loc='center left', bbox_to_anchor=(1.02, 0.5))
    plt.savefig(plot_dir / (spectrometer_name + '05_std_dev_wavelength_corrected'))
    plt.close()


if __name__ == '__main__':
    path_dir = Path("C:/apps/00_MA/data/optical_config_compare/ingest/ThorCCS175")
    file_list = [Path('spec60_full_Thor.json'), Path('spec60_25_Flame_75_Thor.json'),
                 Path('spec60_75_Flame_25_Thor.json')]
    allan(path_dir, file_list, boundaries=(500, 1000))
    path_dir = Path("C:/apps/00_MA/data/optical_config_compare/ingest/FlameNIR")
    file_list = [Path('spec60_full_Flame.json'), Path('spec60_75_Flame_25_Thor.json'),
                 Path('spec60_25_Flame_75_Thor.json')]
    allan(path_dir, file_list)
    print('Done')
