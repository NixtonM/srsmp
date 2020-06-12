from srsmp.SpectraGenerator import SpectraGenerator
from srsmp.Spectrometers import ThorCCS175, FlameNIR

sg = SpectraGenerator('spec60_75_Flame_25_Thor', 0, instrument_list=[ThorCCS175(), FlameNIR()])
sg.measure_automatic(1024, 500)
