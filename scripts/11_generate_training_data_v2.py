import configparser
import logging
from pathlib import Path
import sys

from srsmp.SpectraGenerator import SpectraGenerator
from srsmp.utils import check_and_init_all_dir

config_file = "config.ini"


def input_string(prompt_text):
    user_input = input(prompt_text)
    text = user_input.replace(" ","_")
    if not text: # Can be made more robust
        print("Empty strings not allowed!")
        text = input_string(prompt_text)
    return text


def input_int(prompt_text, default=None):
    if default is not None:
        user_input = input(prompt_text) or default
    else:
        user_input = input(prompt_text)
    try:
        val = int(user_input)
    except ValueError:
        print("Input must be an integer!")
        val = input_int(prompt_text, default)
    return val


def input_int_range(prompt_text,lower,upper, default=None):
    val = input_int(prompt_text, default)
    if val > upper or val < lower:
        print("Input must be within {:d} and {:d}".format(lower,upper))
        val = input_int_range(prompt_text,lower,upper,default)
    return val


if __name__ == "__main__":
    input("Make sure all settings are correct in 'config.ini' and press [ENTER] to continue.")
    print("---------------------------------------------")
    config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
    config.read(config_file)
    check_and_init_all_dir(config)

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(filename=Path(config['Spectroscopy']['sample_ingest_dir']) / 'ingest.log',
                        level=logging.INFO)
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.WARNING)
    logging.getLogger().addHandler(sh)

    material_id = input_string("Provide material identifier (avoid whitespaces): ")
    print("---------------------------------------------")
    class_dict = dict(config['ClassLookup'])
    rev_class_dict = {value: int(key) for key, value in class_dict.items()}

    class_prompt_str = ("The following prediction class identifiers were definied in 'config.ini':\n")
    for key, value in class_dict.items():
        class_prompt_str = class_prompt_str + "{:d}:\t".format(int(key)) + value + "\n"
    class_prompt_str = class_prompt_str+ "Enter the corresponding number: "

    property_id = str(input_int_range(class_prompt_str, int(min(class_dict.keys())),
                                      int(max(class_dict.keys()))))

    class_id = material_id + "_" + class_dict[property_id]
    file_path = Path(config['Spectroscopy']['sample_ingest_dir']) / (class_id + '.json')
    print("\nThe collected spectra will be saved in " + str(file_path))
    print("---------------------------------------------")
    # campaign_nr = input_int("Enter campaign number or -1 for automatic increase [-1]: ", -1)
    # print("---------------------------------------------")
    spg = SpectraGenerator(class_id, config_file=config_file)
    print("---------------------------------------------")
    nb_points = input_int("Number of points to be measured [10]: ", 10)
    print("---------------------------------------------")
    nb_samples = input_int("Number of samples per point [1]: ", 1)
    print("---------------------------------------------")
    integration_time = input_int("Integration time [0 for automatic]: ", 0)
    print("---------------------------------------------")

    spg.measure_spectra_set(nb_points, nb_samples, integration_time)


