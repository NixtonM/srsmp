from matplotlib import pyplot as plt

from srsmp.Spectrometers import ThorCCS175, FlameNIR
from srsmp.DataHandlers import SpectraData


sd = SpectraData()
sd.current_class = 'test_1'
sd.current_point_nb = 1

ccs = ThorCCS175()
flame = FlameNIR()

# Take reference
ccs_measurement, integration_time = ccs.get_ideal_scan()
flame.integration_time = integration_time
flame_measurement = flame.get_scan()
sd.add_reference_to_set(ccs_measurement+flame_measurement, integration_time, 'white_board')

for i in range(10):
    ccs_measurement, integration_time = ccs.get_ideal_scan()
    flame.integration_time = integration_time
    flame_measurement = flame.get_scan()
    sd.add_measurement_to_set(ccs_measurement + flame_measurement, integration_time)

sd.finish_measurement_set()
sd.save_class_data()
sd.save_reference_data()

print(ccs_measurement)
print(flame_measurement)
