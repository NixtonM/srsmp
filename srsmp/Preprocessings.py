import numpy as np
import warnings
from sklearn.preprocessing import MinMaxScaler, RobustScaler


# preprocess the data
def pre_process(measurements,ref_curve,ref_scale,method):
    method = int(method)
    if method==1: #5
        measurements = np.asarray(measurements)
        cur_x = measurements * ref_scale
        cur_x = cur_x / max(abs(cur_x))
    elif method==2: #3
        measurements = np.asarray(measurements)
        abs_x = np.absolute(measurements)
        cur_x = measurements / abs_x.max(axis=0)
        cur_x = cur_x - ref_curve
    elif method==3: #6
        measurements = np.asarray(measurements)
        cur_x = measurements / max(abs(measurements))
        cur_x = cur_x - ref_curve
        cur_x = np.expand_dims(cur_x,axis=2)
        mms = MinMaxScaler(feature_range=(-1,1))
        mms.fit_transform(cur_x)
        cur_x = np.squeeze(cur_x, axis=1)
    elif method==4: #8
        measurements = np.asarray(measurements)
        cur_x = measurements / max(abs(measurements))
        cur_x = cur_x - ref_curve
        cur_x = np.expand_dims(cur_x,axis=2)
        rs = RobustScaler()
        rs.fit_transform(cur_x)
        cur_x = np.squeeze(cur_x, axis=1)
    else:
        warnings.warn('No valid preprocessing method enterred. Continue with method 3.')
        pre_process(measurements,ref_curve,ref_scale,method=3)

    return cur_x


# calculates scaling factor and reference curve according to sampled spectralons
def CalcRefScale(s25,s60):
    if (len(s25) != 0) & (len(s60) != 0):
        fact_25 = 0.25 / (sum(s25)/len(s25))
        fact_60 = 0.6 / (sum(s60)/len(s60))
        ref_scale = (fact_25 + fact_60) / 2
        ref_curve = (sum(s60)/len(s60)) / max(abs(sum(s60)/len(s60)))
    elif s25:
        ref_scale = 0.25 / (sum(s25)/len(s25))
        ref_curve = (sum(s25)/len(s25)) / max(abs(sum(s25)/len(s25)))
    elif s60:
        ref_scale = 0.6 / (sum(s60)/len(s60))
        ref_curve = (sum(s60)/len(s60)) / max(abs(sum(s60)/len(s60)))
    else:
        print('No reference spectralon given.')

    return ref_scale, ref_curve
