from abc import ABC, abstractmethod

from pathlib import Path
import json
import os
from datetime import datetime


class DataHandler(ABC):

    @abstractmethod
    def load_all_measurements(self):
        pass


class ThorData(DataHandler):

    def __init__(self, path_to_dataset):
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

    def load_class_data(self, class_id, campaign_nr=None):
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

    def load_previous_measurements(self, file_path):
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

    def update_class_data(self, class_id, class_data):
        self.data[class_id][self.current_campaign_nrs[class_id]] = class_data

    def save_all_data(self):
        for class_id in self.class_ids:
            self.save_class_data(class_id)

    def save_class_data(self, class_id):
        try:
            with open(self._file_paths[class_id], 'w') as json_file:
                json.dump(self.data[class_id], json_file, indent=2)
        except:
            print("HELP")  # To be made more meaningful
            raise SystemError("Save did not work")

    # Can probably be removed
    def load_all_measurements(self):
        if os.path.exists(self._file_path):
            with open(self._file_path, 'r') as json_file:
                data = {self._class_id: json.load(json_file)}
        else:
            data = {self._class_id: {}}
        return data

    def save_measurements(self, measurements):
        with open(self._file_path, 'w') as json_file:
            json.dump(measurements, json_file, indent=2)


class FlameNIRData(DataHandler):

    def load_all_measurements(self):
        pass
