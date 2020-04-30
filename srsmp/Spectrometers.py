from abc import ABC, abstractmethod

import


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
        self.ccs = instrument('CCS')
        self.integration_time = integration_time  # in ms
        self.integration_time_test = integration_time // 10
        # self.wavelengths = self.ccs.get_wavelengths()
        self.wavelengths = self.ccs._wavelength_array
        print("Connection successful")

    def get_ideal_scan(self, target_intensity=0.8, test_time=None):
        self.set_ideal_integration_time(target_intensity=0.8, test_time=None)
        return self.get_scan(), self.integration_time

    def get_scan(self, test=False):
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

    def __init__(self, integration_time):
        pass

    def get_scan(self):
        pass

    def set_integration_time(self, integration_time):
        pass


