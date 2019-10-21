# Create your own Python Script.
# Make use of the objects _gui (of type IUserInterface), _deviceMgr (of type IDeviceManager) and _script (of type IScriptEngine) 
import clr
import ThorlabsSpectrum
import time
from ThorlabsSpectrum import *
from System import *



# spectrometer definitions
ourCCS = _deviceMgr.GetDevice[CCS]('M00411244')
_script.WriteLine( ourCCS.GetDeviceModel() )
_gui.SetActiveTrace(DisplayMode.Spectrum,0)
myTrace = _gui.GetSpectrumTrace(0)
myTrace.SetToWrite('M00411244')


# acquisition details
acquisition_time = 100 #[100, 150, 200] ms
spectralons = [25, 60] #%
#number_of_samples = 100
#classes = [1, 2, 3, 4, 5] #mm

k=0
_gui.ShowErrorMessage('Set spectralon: {0}'.format(spectralons[k]))
ret = ourCCS.StartSingleAcquisition()

while ourCCS.IsRunningSingleAcquisition():
    _script.Sleep(acquisition_time)

myTrace.SaveToFile("P:/gsg/gsgstud/07_Interdisciplinary_Project/IPA_Meyer_Weiss/05_NeuralNet/data_191014_PC/spectralon_atime{:03d}_spec25_post.csv".format(acquisition_time),SpectrumFileFormat.CSV)






_gui.ShowErrorMessage('Check acquisition time: {0}ms'.format(acquisition_time))

# measure samples
for i in range(len(classes)):
    
    # pre-spectralon measurements
    for k in range(len(spectralons)):
        _gui.ShowErrorMessage('Set spectralon: {0}'.format(spectralons[k]))
        ret = ourCCS.StartSingleAcquisition()

        while ourCCS.IsRunningSingleAcquisition():
            _script.Sleep(acquisition_time)
        
        myTrace.SaveToFile("P:/gsg/gsgstud/07_Interdisciplinary_Project/IPA_Meyer_Weiss/05_NeuralNet/data_191014_PC/spectralon_atime{:03d}_spec{:03d}_preClass{:03d}.csv".format(acquisition_time,spectralons[k],classes[i]),SpectrumFileFormat.CSV)



    _gui.ShowErrorMessage('Measuring class: {0}mm'.format(classes[i]))
    for k in range(number_of_samples):
        _gui.ShowErrorMessage('Press to measure sample {0}'.format(k+1))
    
        ret = ourCCS.StartSingleAcquisition()

        while ourCCS.IsRunningSingleAcquisition():
            _script.Sleep(acquisition_time)
        
        myTrace.SaveToFile("P:/gsg/gsgstud/07_Interdisciplinary_Project/IPA_Meyer_Weiss/05_NeuralNet/data_191014_PC/class{:03d}mm_atime{:03d}_{:03d}.csv".format(classes[i],acquisition_time,k),SpectrumFileFormat.CSV)

        
# post-spectralon measurements
for i in range(len(spectralons)):
    _gui.ShowErrorMessage('Set spectralon: {0}'.format(spectralons[i]))
    ret = ourCCS.StartSingleAcquisition()

    while ourCCS.IsRunningSingleAcquisition():
        _script.Sleep(acquisition_time)
    
    myTrace.SaveToFile("P:/gsg/gsgstud/07_Interdisciplinary_Project/IPA_Meyer_Weiss/05_NeuralNet/data_191014_PC/spectralon_atime{:03d}_spec{:03d}_post.csv".format(acquisition_time,spectralons[i]),SpectrumFileFormat.CSV)


