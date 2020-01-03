from . import Thor
import configparser
from pathlib import Path

import datetime
import json
import os

import time


known_ref_spectra = {'spec_60': {'name': "Spectralon 60%"}} # To be expanded into json and loaded

class SpectraGenerator:

    def __init__(self, class_id, campaign_nr, config_file='config.ini'):
        self.measurements = {}
        self.class_id = class_id
        self._config = configparser.ConfigParser()
        self._config.read(config_file)
        self._ingest_path = Path(self._config['Spectroscopy']['sample_ingest_dir_path'])
        self._ref_sample = self._config['Spectroscopy']['reference_spectra']

        self.thor_ccs = Thor.Thor_CCS175()

        class_campaign_nr = None if campaign_nr < 0 else campaign_nr
        self.thor_data = Thor.ThorData(self._ingest_path)
        self.thor_data.load_class_data(class_id, class_campaign_nr)

        

    def measure_spectra(self,sample_size):
        wavelengths = self.thor_ccs.wavelengths.tolist()
        data = {'reference': {}}
        valid_ref_measurement = False

        input("Prepare {} for measurement and press any key to start measurement.".format(
                known_ref_spectra[self._ref_sample]['name']))
        while not valid_ref_measurement:
            try:
                measurement_data, measurement_time = self.thor_ccs.get_ideal_scan()
                valid_ref_measurement = True
            except:
                print("Reference measurement was NOT successful.")
                input("Reset and press any key to repeat.")
                self.thor_ccs.set_integration_time_test(self,20)
        epoch_id = datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")[:-3]
        data['reference'][epoch_id] = (wavelengths, measurement_data.tolist(), 
                                       measurement_time, self._ref_sample,)
        print("Reference measured and set. \nPrepare sample:")
        for i in range(sample_size):
            valid_measurement = False
            input("Press any key to start sample measurement {}".format(i+1))
            while not valid_measurement:
                try:
                    measurement_data, measurement_time = self.thor_ccs.get_ideal_scan()
                    valid_measurement = True
                except:
                    print("Sample measurement was NOT successful.")
                    input("Reset and press any key to repeat sample {}.".format(i+1))
                    self.thor_ccs.set_integration_time_test(self,20)
            epoch_id = datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")[:-3]
            data[epoch_id] = (wavelengths, measurement_data.tolist(), measurement_time,)

        input("Prepare {} for second measurement and press any key to start measurement.".format(
                 known_ref_spectra[self._ref_sample]['name']))
        while not valid_ref_measurement:
            try:
                measurement_data, measurement_time = self.thor_ccs.get_ideal_scan()
                valid_ref_measurement = True
            except:
                print("Reference measurement was NOT successful.")
                input("Reset and press any key to repeat.")
                self.thor_ccs.set_integration_time_test(self,20)
        epoch_id = datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")[:-3]
        data['reference'][epoch_id] = (wavelengths, measurement_data.tolist(), 
                                       measurement_time, self._ref_sample,)

        self.thor_data.update_class_data(self.class_id,data)

        self.thor_data.save_class_data(self.class_id)
        print("Samples saved.")