from srsmp import *
#from Thor import Thor_CCS175, ThorData
#from Predictor.Predictor import Hydrophobic_Predictor

import os
import socket, time, re
import struct, random, datetime
import numpy as np

from pathlib import Path
import configparser
import datetime

from scipy.spatial.transform import Rotation as R


config_file = "config.ini"


def read_com_link(com_link_file):

    datashare_regex = r"<([I|D|S]):(\w+)>\s+(\S*)\s*"
    data = com_link_file.read_text()
    datashare_matches = re.findall(datashare_regex, data)
    datashare_dict = dict()

    for m in datashare_matches:
        if m[0] == 'I':
            datashare_dict[m[1]] = int(m[2])
        elif m[0] == 'D':
            datashare_dict[m[1]] = float(m[2])
        elif m[0] == 'S':
            datashare_dict[m[1]] = m[2]
        else:
            raise ValueError("Unknown DataShare data format")
    return datashare_dict

def transform_probe(probe):
    r = R.from_euler('xyz',np.asarray([probe['Rx'],probe['Ry'],probe['Rz']]),degrees=True)
    offset_world = r.apply(OFFSET)
    return np.asarray([probe['X_TP'],probe['Y_TP'],probe['Z_TP']]) + offset_world


if __name__ == "__main__":
    print("---------------------------------------------")
    config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
    config.read(config_file)
    check_and_init_all_dir(config)
    make_base_datashare(config,os.path.dirname(os.path.realpath(__file__)))

    folder_path = Path(config['PredictApp']['spectra_dir'])
    com_link_location = Path(config['PredictApp']['com_link_dir'])/Path(
        config['PredictApp']['com_link_file'])
    class_id = datetime.datetime.now().strftime("%Y%m%d%H%M")
    
    class_predicition_dict = dict(config['ClassLookup'])
    
    dx = float(config['ToolOffset']['dx'])
    dy = float(config['ToolOffset']['dy'])
    dz = float(config['ToolOffset']['dz'])

    OFFSET = np.asarray([dx,dy,dz],dtype=np.float64)
    print("Tool offset used:\tdx = {:+.4f}\n\t\t\tdy = {:+.4f}\n\t\t\tdz = {:+.4f}".format(dx,dy,dz))
    print("---------------------------------------------")

    HOST = '127.0.0.1'  # Standard loopback interface address (localhost)
    PORT = 65432        # Port to listen on (non-privileged ports are > 1023)


    myCCS = Thor_CCS175()
    wavelengths = myCCS.wavelengths.tolist()
    myDataHandler = ThorData(folder_path)
    measurement_data = {'reference': {}}
    print("---------------------------------------------")
   
    # unique class_id --> will init empty data
    myDataHandler.load_class_data(class_id)

    hydro_pred = Hydrophobic_Predictor()
    ref_set = False
    
    pt_nr = int(class_id[-2:])*100
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, PORT))
        s.listen()
        print("---------------------------------------------")
        print('Ready for reference measurement')
        while True:
            exit_order = False
            conn, addr = s.accept()
            with conn:
                epoch = datetime.datetime.now()
                key_time = epoch.strftime("%Y%m%d%H%M%S%f")[:-3]
                print('Connected at', epoch.strftime("%d.%m.%Y %H:%M:%S"))
                while True:
                    data = conn.recv(1024)
                    if not data:
                        break
                    data_s = str(data,'utf8')
                    comm_key = int(data_s)
                    if comm_key == -99:
                        exit_order = True
                    else:
                        try:
                            measurement, measurement_time = myCCS.get_ideal_scan()
                        except:
                            print("Spectral scan was unsuccessful.")
                        else:
                            if comm_key == -1:
                                measurement_data['reference'][key_time] = (wavelengths,
                                    measurement.tolist(),measurement_time,"spec_60",)
                                hydro_pred.set_spectralon(measurement,myCCS.wavelengths,60)
                                print("Reference 'Spectralon 60%' set")
                                ref_set = True
                            elif comm_key == -2:
                                measurement_data['reference'][key_time] = (wavelengths,
                                    measurement.tolist(),measurement_time,"spec_25",)
                                hydro_pred.set_spectralon(measurement,myCCS.wavelengths,25)
                                print("Reference 'Spectralon 25%' set")
                                ref_set = True
                            else:
                                if not ref_set:
                                    print("Measure a reference spectralon first.")
                                    break

                                measurement_data[pt_nr] = (wavelengths,
                                    measurement.tolist(),measurement_time,)
                                class_prediction = hydro_pred.predict(measurement,myCCS.wavelengths)
                                class_prediction_str = class_predicition_dict[str(class_prediction)]

                    
                                ## Do 3D transform
                                probe = read_com_link(com_link_location)
                                pt_coord = transform_probe(probe)

                                ## Write results
                                SA_datashare = ("<ASCII>\n"
                                                    "<I:pt_nr>\n"
                                                    "{:d}\n"
                                                    "<S:class_pred>\n"
                                                    "{}\n"
                                                    "<D:x>\n"
                                                    "{:.16e}\n"
                                                    "<D:y>\n"
                                                    "{:.16e}\n"
                                                    "<D:z>\n"
                                                    "{:.16e}\n"
                                                    .format(pt_nr,class_prediction_str,pt_coord[0],pt_coord[1],pt_coord[2])
                                                    )
                                com_link_location.write_bytes(SA_datashare.encode('ascii'))
                                
                                print("Measurement {:04d} taken and predicted as {}".format(pt_nr,class_prediction_str))
                                pt_nr += 1
                    exit_code = 0
                    conn.sendall(bytes(str(exit_code), 'utf8'))
                    break
                if exit_order:
                    myDataHandler.update_class_data(class_id,measurement_data)
                    myDataHandler.save_all_data()
                    break
            
            print("-----------------------")
    print("Saving and closing for business")