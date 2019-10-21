from instrumental import instrument
import datetime
import json
import matplotlib.pyplot as plt
import os
import numpy as np
import time

measurements = dict()
path_to_datasets = "example_data"
material_class_id = "Thorlabs_Red"
measurement_campaign_nr = 0
force_measurement_campaign_nr = False # in case you want to overwrite a previous campaign
sample_size = 10
ref_sample_1 = 'spec_60'
ref_sample_2 = 'spec_60'

def main():
    global measurements, measurement_campign_nr


    def get_ideal_integration_time(myCCS, target_intensity = 0.8, test_time = 200):
        try:
            data = myCCS.take_data(integration_time='{} ms'.format(test_time), max_attempts=1)
        except:
            print("Reducing time")
            return get_ideal_integration_time(myCCS, target_intensity = target_intensity ,test_time=test_time/2)
        max_intensity = max(data[0])
        if max_intensity < 0.05:
            print("Increasing time")
            return get_ideal_integration_time(myCCS, target_intensity = target_intensity ,test_time=test_time*2)
        return int(test_time*target_intensity/max_intensity)



    def load_previous_measurements():
        global measurements, measurement_campaign_nr, force_measurement_campaign_nr
        if os.path.exists(os.path.join(path_to_datasets,"{}.json".format(material_class_id))):
            with open(os.path.join(path_to_datasets,"{}.json".format(material_class_id)), 'r') as json_file:
                measurements[material_class_id] = json.load(json_file)
            measurement_campaign_nr = measurement_campaign_nr if force_measurement_campaign_nr else (max(list(map(int, list(measurements[material_class_id].keys())))) + 1)
        else:
            measurements[material_class_id] = {}
            measurement_campaign_nr = measurement_campaign_nr if force_measurement_campaign_nr else 0
        return



    def save_measurements():
        if not os.path.exists(path_to_datasets):
            try:
                os.mkdir(path_to_datasets)
            except:
                print("Folder could not be created")
        with open(os.path.join(path_to_datasets,"{}.json".format(material_class_id)), 'w') as json_file:
            json.dump(measurements[material_class_id], json_file, indent=2)



    myCCS = instrument('CCS')
    myCCS.reset()
    load_previous_measurements()
    
    material_class = measurements[material_class_id]
    
    current_campaign = {}
    reference_measurments = {}

    valid_ref_measurement = False
    input("Prepare reference spectra for measurement and press any key to start measurement...")
    while not valid_ref_measurement:
        ideal_integgration_time = get_ideal_integration_time(myCCS)
        try:
            data,wavelengths = myCCS.take_data('{} ms'.format(ideal_integgration_time))
        except:
            print('Retrying measurement')
        else:
            epoch = datetime.datetime.now()
            reference_measurments[epoch.strftime("%Y%m%d-%H%M%S")] = (wavelengths.tolist(),data.tolist(),ideal_integgration_time,ref_sample_1,)
            valid_ref_measurement = True


    i = 0
    input("Press Enter to continue with first sample measurment...")
    while i < sample_size:
        ideal_integgration_time = get_ideal_integration_time(myCCS)

        print('{} ms'.format(ideal_integgration_time))
        try:
            t = time.time()
            data,wavelengths = myCCS.take_data('{} ms'.format(ideal_integgration_time),max_attempts=1)
            print(time.time() -t)
        except:
            print('Retrying measurement')
        else:
            epoch = datetime.datetime.now()
            current_campaign[epoch.strftime("%Y%m%d-%H%M%S")] = (wavelengths.tolist(),data.tolist(),ideal_integgration_time,)
            input("Reposition and press Enter to continue...")
            i += 1

    valid_ref_measurement = False
    input("Prepare second reference spectra for measurement and press any key to start measurement...")
    while not valid_ref_measurement:
        ideal_integgration_time = get_ideal_integration_time(myCCS)
        try:
            data,wavelengths = myCCS.take_data('{} ms'.format(ideal_integgration_time))
        except:
            print('Retrying measurement')
        else:
            epoch = datetime.datetime.now()
            reference_measurments[epoch.strftime("%Y%m%d-%H%M%S")] = (wavelengths.tolist(),data.tolist(),ideal_integgration_time,ref_sample_2,)
            valid_ref_measurement = True


    print(measurement_campaign_nr)
    current_campaign['reference'] = reference_measurments
    material_class[str(measurement_campaign_nr)] = current_campaign
    
    measurements[material_class_id] = material_class


    save_measurements()

 


if __name__ == "__main__":
    main()