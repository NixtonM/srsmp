import instrumental
from instrumental import instrument, list_instruments
import datetime
import json
import matplotlib.pyplot as plt

measurements = dict()
material_class = "Thorlabs_Red"
measurement_campaign_nr = 1
current_campaign = {str(measurement_campaign_nr): dict()}

def get_ideal_integration_time(myCCS, target_intensity = 0.8, test_time = 200):
    try:
        data = myCCS.take_data(integration_time='{} ms'.format(test_time))
    except:
        print("reducing time")
        return get_ideal_integration_time(myCCS, target_intensity = target_intensity ,test_time=test_time/2)
    max_intensity = max(data[0])
    if max_intensity < 0.05:
        return get_ideal_integration_time(myCCS, target_intensity = target_intensity ,test_time=test_time*2)
#    print(max_intensity)
    return int(test_time*target_intensity/max_intensity)


paramsets = list_instruments()
#print(paramsets)
myCCS = instrument(paramsets[0])

#myCCS.reset()

ideal_integgration_time = get_ideal_integration_time(myCCS)

print('{} ms'.format(ideal_integgration_time))
data,wavelengths = myCCS.take_data('{} ms'.format(ideal_integgration_time))
epoch = datetime.datetime.now()

current_measurement = (wavelengths.tolist(),data.tolist(),ideal_integgration_time,)
measurement_dict = {
    epoch.strftime("%Y%m%d-%H%M%S"): current_measurement
    }

current_campaign[str(measurement_campaign_nr)] = measurement_dict

measurements[material_class] = current_campaign

plt.plot(wavelengths, data)
plt.show()

with open('{}.txt'.format(material_class), 'w') as json_file:
    json.dump(measurements[material_class], json_file)

#print(json.dump(measurements))