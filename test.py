#!/home/maxx/cse572/bin/python3.6
###############################################################################
#
#        File: test.py
#      Author: Maxx Rodriguez
#.      ASUID: 1204885197
#    Creation: 4/13/2021
#
###############################################################################

import pandas as pd
import pickle
import numpy as np

def main():
    cgm_data_file = 'test.csv'
    feature_extractions = pd.DataFrame()
    # Read CSV Files
    cgmFrame = pd.read_csv(cgm_data_file, low_memory=False, header=None)

    model = pickle.load(open('model.sav', 'rb'))

    for index, row in cgmFrame.iterrows():
        feature_extractions = feature_extractions.append(test_feature_extraction(row), ignore_index=True)

    # print(feature_extractions)
    pred = model.predict(feature_extractions)
    # print("Accuracy:",metrics.accuracy_score(labelVector, pred))
    resultsDf = pd.DataFrame(pred)
    resultsDf.to_csv('./Result.csv', index=False, header=False)

def test_feature_extraction(cgmRow):
    cgmData = cgmRow.T
    cgmData = cgmData.iloc[::-1]
    cgmData.reset_index(drop=True, inplace=True)
    cgmMaxIndex = None
    # feature 1 basic
    cgmMaxIndex = cgmData.idxmax()
    dG = cgmData.loc[cgmMaxIndex] - cgmData.iloc[0]
    # feature 2 time delta
    timeDiff = cgmMaxIndex*5*60

    # feature 3 dG Normalized
    try:
        dGNorm = dG / cgmData.iloc[0]
    except:
        dGNorm = 0

    # feature 4, 5, 6, 7 FFT
    fftPower = np.abs(np.fft.fft(cgmData))**2
    fftPower = np.delete(fftPower, 0)
    index = np.argmax(fftPower)
    fftFreq = np.fft.fftfreq(len(cgmData))
    fftFreq = np.delete(fftFreq, 0)
    p1 = fftPower[0]
    f1 = fftFreq[0]
    p2 = fftPower[index]
    f2 = fftFreq[index]

    # feature 8 first derivative
    # feature 9 second derivative
    firstD = np.gradient([cgmRow.iloc[0],cgmRow.iloc[-1]], [0, (23*5*60)])
    secondD = np.gradient(firstD)
    featureDict = {
        'timeDiff': timeDiff,
        'dgNorm': dGNorm,
        'p1': p1,
        'f1': f1,
        'p2': p2,
        'f2': f2,
        'firstD': firstD[0],
        # 'secondD': secondD[0]
    }

    return featureDict

if __name__ == "__main__":
    main()