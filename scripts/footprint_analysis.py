from pathlib import Path
import numpy as np


from matplotlib import pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
plt.style.use('thesis_2col')

def footprint_analysis(csv_file, title):
    plot_dir = Path('./plots/2col/footprint/')
    plot_dir.mkdir(parents=True, exist_ok=True)

    footprint_data = np.genfromtxt(csv_file, delimiter=',')
    distance = footprint_data[:, 0]
    avg_power = np.mean(footprint_data[:, 1:], 1)

    low_power = np.min(avg_power)
    high_power = np.max(avg_power)
    diff_power = high_power - low_power

    middle_power = low_power + 0.5* diff_power

    initial_low_group = np.where(avg_power <= low_power + 0.1*diff_power)
    initial_high_group = np.where(avg_power >= high_power - 0.1*diff_power)

    k = 2

    low_mean = np.mean(avg_power[initial_low_group])
    low_std = np.std(avg_power[initial_low_group])
    low_group = np.where(np.logical_and(avg_power >= low_mean - k * low_std, avg_power <= low_mean + k * low_std))

    high_mean = np.mean(avg_power[initial_high_group])
    high_std = np.std(avg_power[initial_high_group])
    high_group = np.where(np.logical_and(avg_power >= high_mean - k * high_std, avg_power <= high_mean + k * high_std))



    transition_group = np.setdiff1d(np.where(avg_power), np.union1d(low_group, high_group))
    # transition_group_plus = np.union1d(np.union1d(high_group[0][-1], transition_group), low_group[0][0])

    low_group_plus = np.union1d(transition_group[-1], low_group)
    high_group_plus = np.union1d(high_group, transition_group[0])

    median_dist = distance[np.absolute(avg_power - middle_power).argmin()]

    fig, ax = plt.subplots()
    ax.plot((distance - median_dist)[low_group_plus], avg_power[low_group_plus], 'b')
    ax.plot((distance - median_dist)[transition_group], avg_power[transition_group], 'r')
    ax.plot((distance - median_dist)[high_group_plus], avg_power[high_group_plus], 'b')

    ax.axhline(low_mean, ls='--', c='0.1')
    ax.axhline(low_mean + k*low_std, ls=':', c='0.1')
    ax.axhline(low_mean - k*low_std, ls=':', c='0.1')

    ax.axhline(high_mean, ls='--', c='0.1')
    ax.axhline(high_mean + k*high_std, ls=':', c='0.1')
    ax.axhline(high_mean - k*high_std, ls=':', c='0.1')

    ax.axvline((distance-median_dist)[transition_group[0]], ls='-.', c='c')
    ax.axvline((distance-median_dist)[transition_group[-1]], ls='-.', c='c')

    ax.text((distance - median_dist)[transition_group[0]] - 0.05, middle_power, '{:.2f} mm'.format(
        (distance - median_dist)[transition_group[0]]
    ), horizontalalignment='right', color='c', weight='bold')

    ax.text((distance-median_dist)[transition_group[-1]] + 0.05, middle_power, '{:.2f} mm'.format(
        (distance-median_dist)[transition_group[-1]]
    ), color='c', weight='bold')

    ax.set_title(title)
    ax.set_xlabel('Position [mm]')
    ax.set_ylabel('Power level')

    ax.xaxis.set_major_locator(MultipleLocator(1))

    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    # ax.yaxis.set_minor_locator(AutoMinorLocator(2))

    ax.yaxis.set_ticks([low_mean, high_mean])
    ax.yaxis.set_ticklabels(['low', 'high'])

    ax.grid(True, 'minor', linestyle=':')

    fig.savefig(plot_dir / title.replace(' ', '_'))
    plt.show()


if __name__ == '__main__':
    csv_file = Path('C:/apps/00_MA/data/footprint_analysis/big_sep.csv')
    footprint_analysis(csv_file, '10 mm distance')
    csv_file = Path('C:/apps/00_MA/data/footprint_analysis/small_sep_cut.csv')
    footprint_analysis(csv_file, '6 mm distance')
    csv_file = Path('C:/apps/00_MA/data/footprint_analysis/no_sep.csv')
    footprint_analysis(csv_file, '2 mm distance')
