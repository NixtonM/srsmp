from abc import ABC, abstractmethod

from instrumental.drivers import instrument as inst_instrument
import os, json, datetime, time
from pathlib import Path

from seabreeze.spectrometers import Spectrometer as sb_Spectrometer
from seabreeze.spectrometers import list_devices as sb_list_devices


class Spectrometer(ABC):


    @abstractmethod
    def get_scan(self):
        pass

    @abstractmethod
    def set_integration_time(self, integration_time):
        pass


class ThorCCS175(Spectrometer):

    def __init__(self, integration_time=200):
        print("Establishing connection to Thorlabs CCS175 spectrometer.\n")
        self.instrument = inst_instrument('CCS')
        self.integration_time = integration_time  # in ms
        self.integration_time_test = integration_time // 10
        # self.wavelengths = self.ccs.get_wavelengths()
        self.wavelengths = self.instrument._wavelength_array
        print("Connection successful")

    def get_ideal_scan(self, target_intensity=0.8, test_time=None):
        self.set_ideal_integration_time(target_intensity=0.8, test_time=None)
        return self.get_scan(), self.integration_time

    def get_scan(self, test=False):
        self.instrument.reset()
        if test:
            self.instrument.set_integration_time('{} ms'.format(self.integration_time_test))
        else:
            self.instrument.set_integration_time('{} ms'.format(self.integration_time))

        self.instrument.start_single_scan()
        while not self.instrument.is_data_ready():
            time.sleep(0.01)
        data = self.instrument.get_scan_data()
        if max(data) > 0.95:
            raise Warning('Raw data is saturated')
        return data

    def set_integration_time(self, integration_time):
        self.integration_time = integration_time

    def set_integration_time_test(self, integration_time_test):
        self.integration_time_test = integration_time_test

    def set_ideal_integration_time(self, target_intensity=0.8, test_time=None):
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
            self.set_integration_time(int(self.integration_time_test *
                                          target_intensity / max_intensity))
            break


class FlameNIR(Spectrometer):

    def __init__(self, integration_time = 200):
        self.instrument = sb_Spectrometer.from_first_available()
        self.instrument.integration_time_micros(integration_time * 1000)

    def get_scan(self):
        return self.instrument.intensities()

    def set_integration_time(self, integration_time):
        self.instrument.integration_time_micros(integration_time * 1000)


