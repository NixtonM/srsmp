from instrumental.drivers import instrument
import os, json, datetime, time
from pathlib import Path
import numpy as np


class Thor_CCS175:
    def __init__(self,integration_time=200):
        print("----------\nEstablishing connection to Thorlabs CCS175 spectrometer.\n----------\n")
        self.ccs = instrument('CCS')
        self.integration_time = integration_time # in ms
        self.integration_time_test = integration_time//10
        #self.wavelengths = self.ccs.get_wavelengths()
        self.wavelengths = self.ccs._wavelength_array


    def get_ideal_scan(self, target_intensity = 0.8, test_time = None):
        self.set_ideal_integration_time(target_intensity = 0.8, test_time = None)
        return self.get_scan(), self.integration_time
        
    def get_scan(self,test=False):
        self.ccs.reset()
        if test:
            self.ccs.set_integration_time('{} ms'.format(self.integration_time_test))
        else:
            self.ccs.set_integration_time('{} ms'.format(self.integration_time))

        self.ccs.start_single_scan()
        while not self.ccs.is_data_ready():
            time.sleep(0.01)
        data = self.ccs.get_scan_data()
        if max(data) > 0.95:
            raise Warning('Raw data is saturated')
        return data

    def set_integration_time(self,integration_time):
        self.integration_time = integration_time

    def set_integration_time_test(self,integration_time_test):
        self.integration_time_test = integration_time_test


    def set_ideal_integration_time(self, target_intensity = 0.8, test_time = None):
        self.integration_time_test = test_time if test_time is not None else self.integration_time_test
        while 1:
            try:
                data = self.get_scan(test=True)
            except:
                self.integration_time_test //= 2
                print("Reducing integration time to:\t{}ms".format(
                    self.integration_time_test))
                continue
            max_intensity = max(data)
            if max_intensity < 0.05:
                self.integration_time_test *= 2
                print("Increasing integration time to:\t{}ms".format(
                    self.integration_time_test))
                continue
            elif max_intensity > 0.5:
                self.integration_time_test //= 10
                print("Reducing integration time to:\t{}ms".format(
                    self.integration_time_test))
                continue
            self.set_integration_time(int(self.integration_time_test*
                                        target_intensity/max_intensity))
            break


class ThorData():

    def __init__(self,path_to_dataset):
        self.class_ids = []
        self.current_campaign_nrs = {}
        self._file_paths = {}
        self._path_to_dataset = Path(path_to_dataset)
        if not self._path_to_dataset.exists():
            try:
                os.mkdir(path_to_dataset)
            except:
                print("Folder {} could not be created".format(path_to_dataset))
                raise
        self.data = {}
      


    def load_class_data(self, class_id, campaign_nr = None):
        if class_id in self.class_ids:
            raise ValueError("This class ID has already been loaded and initialized!")
        self.class_ids.append(class_id)

        file_path = self._path_to_dataset / "{}.json".format(class_id)
        self._file_paths[class_id] = file_path

        if file_path.exists():
            class_data, automatic_campaign_nr = self.load_previous_measurements(file_path)
        else:
            class_data = {}
            automatic_campaign_nr = 1

        class_campaign_nr = automatic_campaign_nr if campaign_nr is None else campaign_nr

        self.data[class_id] = class_data
        self.current_campaign_nrs[class_id] = class_campaign_nr


    def load_previous_measurements(self,file_path):
        try:
            with open(file_path, 'r') as json_file:
                data = json.load(json_file)
            try:
                campaign_nr = max(int(k) for k, _ in data.items()) + 1
            except:
                campaign_nr = datetime.datetime.now().strftime("%Y%m%d-%H%M")
        except:
            print("JSON file could not be loaded correctly! Check correct syntax. Empty dictonary will be returned.")
            data = {}
            campaign_nr = 1
            raise
        return data, campaign_nr

    def update_class_data(self,class_id,class_data):
        self.data[class_id][self.current_campaign_nrs[class_id]] = class_data

    def save_all_data(self):
        for class_id in self.class_ids:
            self.save_class_data(class_id)

    def save_class_data(self,class_id):
        try:
            with open(self._file_paths[class_id],'w') as json_file:
                json.dump(self.data[class_id], json_file, indent=2)
        except:
            print("HELP") # To be made more meaningful
            raise SystemError("Save did not work")

    # Can probably be removed
    def load_all_measurements(self):
        if os.path.exists(self._file_path):
            with open(self._file_path, 'r') as json_file:
                data = {self._class_id: json.load(json_file)}
        else:
            data = {self._class_id: {}}
        return data

    def save_measurements(self,measurements):
        with open(self._file_path, 'w') as json_file:
            json.dump(measurements, json_file, indent=2)

