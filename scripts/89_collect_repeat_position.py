from srsmp.SpectraGenerator import SpectraGenerator
from srsmp.Spectrometers import ThorCCS175, FlameNIR


nb_repeats = 8
nb_samples = 16
sample_name = 'sample47_pos1_after'

sg = SpectraGenerator(sample_name, 0, instrument_list=[ThorCCS175(), FlameNIR()])

for i in range(nb_repeats):
    input('Place on reference ({:d}/{:d}) and press [ENTER]'.format(i+1, nb_repeats))
    sg.switch_class(sample_name+'_spec60', i)
    sg.measure_automatic(nb_samples, 500)
    input('Place on sample ({:d}/{:d}) and press [ENTER]'.format(i+1, nb_repeats))
    sg.switch_class(sample_name, i)
    sg.measure_automatic(nb_samples, 500)

input('Place on reference (final) and press [ENTER]')
sg.switch_class(sample_name+'_spec60', nb_repeats)
sg.measure_automatic(nb_samples, 500)

sg.save_spectras()


