import sys
import configparser
from pathlib import Path

def check_and_init_all_dir(config):
    sections = config.sections()

    paths = list()
    for sec in sections:
        for key in config[sec]:
            if key.endswith("_dir"):
                paths.append(Path(config[sec][key]))
            if key == "results_dir":
                paths.append(Path(config[sec][key])/'output')
                paths.append(Path(config[sec][key])/'model')
    
    for p in paths:
        p.mkdir(parents=True,exist_ok=True)

def make_base_datashare(config,scripts_loc):
    SA_config_file = Path(config['Common']['base_dir']) / "sa_config.ascii"
    
    com_link_loc = Path(config['PredictApp']['com_link_dir']) / Path(
        config['PredictApp']['com_link_file'])
    python_loc = sys.executable

    SA_datashare = ("<ASCII>\n"
                    "<S:com_link>\n"
                    "{}\n"
                    "<S:client_script>\n"
                    "{}\n"
                    "<S:python>\n"
                    "{}\n"
                    .format(com_link_loc.absolute(),scripts_loc,python_loc,)
                    )
    SA_config_file.write_bytes(SA_datashare.encode('ascii'))

    com_link_setup = ("<ASCII>\n")
    com_link_loc.write_bytes(com_link_setup.encode('ascii'))
