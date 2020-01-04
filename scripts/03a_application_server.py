from srsmp import *
#from Thor import Thor_CCS175, ThorData
#from Predictor.Predictor import Hydrophobic_Predictor

import socket, time, re
import struct, random, datetime
import numpy as np

from pathlib import Path

from scipy.spatial.transform import Rotation as R

def read_com_link(regex_pattern):
    with open(com_link_location, 'r') as f:
        data = f.read()
        res = [float(x) for x in re.findall(regex_pattern,data)]
    return {'X':res[0],'Y':res[1],'Z':res[2],'Rx':res[3],'Ry':res[4],'Rz':res[5]}

def transform_probe(probe):
    r = R.from_euler('xyz',np.asarray([probe['Rx'],probe['Ry'],probe['Rz']]),degrees=True)
    offset_world = r.apply(OFFSET)
    return np.asarray([probe['X'],probe['Y'],probe['Z']]) + offset_world


if __name__ == "__main__":

    ## TODO: Load config
    folder_path = "C:\\apps\\IPA\\data"
    class_id = "specimen"
    com_link_location = "C:\\apps\\IPA\\com_link.txt"
    OFFSET = np.asarray([-0.000792585000000,0.065944655000000,-0.131226600000000],dtype=np.float64)

    HOST = '127.0.0.1'  # Standard loopback interface address (localhost)
    PORT = 65432        # Port to listen on (non-privileged ports are > 1023)


    myCCS = Thor_CCS175()
    myDataHandler = ThorData(folder_path,class_id)
    #measurements = myDataHandler.load_measurements()
    cmp_nr, measurements = myDataHandler.init_empty_measurements()
    current_reference = (None,None,)

    hydro_pred = Hydrophobic_Predictor()

    # Definition of scientific pattern regex
    sci_not_regex = re.compile('[+|-]?\ *[0-9]+\.?[0-9]*(?:[Ee]\ *[+|-]?\ *[0-9]+)?')

    pt_nr = 0
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, PORT))
        s.listen()
        print('Ready for measurements')
        while True:
            exit_order = False
            conn, addr = s.accept()
            with conn:
                epoch = datetime.datetime.now()
                key_time = epoch.strftime("%Y%m%d-%H%M%S%f")
                print('Connected at', key_time)
                while True:
                    data = conn.recv(1024)
                    if not data:
                        break
                    data_s = str(data,'utf8')
                    key = int(data_s)
                    if key == -1:
                        myCCS.set_ideal_integration_time()
                        thor_meas = myCCS.get_scan()
                        current_reference = (60,thor_meas,)
                        measurements[cmp_nr]['reference'][key_time] = (myCCS.wavelengths.tolist(),
                            thor_meas.tolist(),myCCS.integration_time,"spec_60",)
                        hydro_pred.set_spectralon(thor_meas,myCCS.wavelengths,60)
                        print("Reference 'Spectralon 60%' set")
                    elif key == -2:
                        myCCS.set_ideal_integration_time()
                        thor_meas = myCCS.get_scan()
                        current_reference = (25,thor_meas,)
                        measurements[cmp_nr]['reference'][key_time] = (myCCS.wavelengths.tolist(),
                            thor_meas.tolist(),myCCS.integration_time,"spec_25",)
                        hydro_pred.set_spectralon(thor_meas,myCCS.wavelengths,25)
                        print("Reference 'Spectralon 25%' set")
                    elif key == -99:
                        exit_order =  True
                    else:
                        ## Do spectrometer
                        myCCS.set_ideal_integration_time()
                        thor_meas = myCCS.get_scan()
                        measurements[cmp_nr][key] = (myCCS.wavelengths.tolist(),
                            thor_meas.tolist(),myCCS.integration_time,key_time,)
                        class_prediction = hydro_pred.predict(thor_meas,myCCS.wavelengths)
                    
                        ## Do 3D transform
                        probe = read_com_link(sci_not_regex)
                        pt_coord = transform_probe(probe)

                        ## Write results
                        with open(com_link_location, 'wb') as f:
                            SA_datashare = ("<ASCII>\n"
                                            "<I:pt_nr>\n"
                                            "\t{:d}\n"
                                            "<I:class_pred>\n"
                                            "\t{:d}\n"
                                            "<D:x>\n"
                                            "\t{:.16e}\n"
                                            "<D:y>\n"
                                            "\t{:.16e}\n"
                                            "<D:z>\n"
                                            "\t{:.16e}\n"
                                            .format(pt_nr,class_prediction,pt_coord[0],pt_coord[1],pt_coord[2])
                                            )
                            f.write(SA_datashare.encode('ascii'))

                        print("Measurement {} taken and predicted as {}".format(key,class_prediction))

                        pt_nr += 1
                    exit_code = 0
                    conn.sendall(bytes(str(exit_code), 'utf8'))
                    break
                if exit_order:
                    myDataHandler.save_measurements(measurements)
                    break
    print("Closed for business")