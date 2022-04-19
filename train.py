#!/home/maxx/cse572/bin/python3.6
###############################################################################
#
#        File: train.py
#      Author: Maxx Rodriguez
#.      ASUID: 1204885197
#    Creation: 4/13/2021
#
###############################################################################

import datetime
import pandas as pd
import numpy as np
from scipy import rand
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
import pickle
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score


_meal_features = pd.DataFrame()
_no_meal_features = pd.DataFrame()


def extractMealData(df):
    mealRows = df.loc[df['BWZ Carb Input (grams)'] > 0]
    mealRows['DateTime'] = pd.to_datetime(mealRows['Date'] + ' ' + mealRows['Time'])
    mealRows['Time'] = pd.to_datetime(mealRows['Time'], format='%H:%M:%S').dt.time
    mealRowsReversed = mealRows[::-1]
    mealRowsReversed = mealRowsReversed.sort_values(by='DateTime', ascending=True)
    mealRowsReversed.reset_index(drop=True, inplace=True)
    numberOfRows = len(mealRowsReversed)
    rowsToDelete = []
    for i in range(numberOfRows):
        if(i+1 < numberOfRows):
            timeDelta = mealRowsReversed.iloc[i+1]['DateTime']-mealRowsReversed.iloc[i]['DateTime']
            if(timeDelta.total_seconds() > 7200):
                # # print(mealRowsReversed.iloc[i]['DateTime'])
                # # print(mealRowsReversed.iloc[i+1]['DateTime'])
                val = 0
            else:
                rowsToDelete.append(i)            

    mealRowsReversed.drop(rowsToDelete, inplace=True)
    mealRowsReversed.reset_index(drop=True, inplace=True)
    return mealRowsReversed

def extractGlucoseData(cgmData, insulinDf):
    global _no_meal_features
    global _meal_features

    cgmData['DateTime'] = pd.to_datetime(cgmData['Date'] + ' ' + cgmData['Time'])
    cgmData['Time'] = pd.to_datetime(cgmData['Time'], format='%H:%M:%S').dt.time
    cgmData = cgmData.sort_values(by='DateTime', ascending=True)
    arrOfGlucoseFrames = []
    arrOfNoMealFrames = []
    for i in range(len(insulinDf)):
        glucoseTime = insulinDf.iloc[i]['DateTime']
        gTime30MinPrior = glucoseTime - datetime.timedelta(minutes=30)
        gTime2hoursAfter = glucoseTime + datetime.timedelta(minutes=120)
        noMealTime = gTime2hoursAfter + datetime.timedelta(minutes=120)
        glucoseSlice = cgmData[
            (cgmData['DateTime'] > gTime30MinPrior) & 
            (cgmData['DateTime'] < gTime2hoursAfter)
        ]
        glucoseSlice2 = cgmData[
            (cgmData['DateTime'] > gTime2hoursAfter) & 
            (cgmData['DateTime'] < noMealTime)
        ]
        glucoseSlice['meal_time'] = glucoseTime
        if(len(glucoseSlice) >= 24): 
            arrOfGlucoseFrames.append(glucoseSlice)
            _meal_features = _meal_features.append(feature_extractor(glucoseSlice),  ignore_index=True)
        if(len(glucoseSlice2) >= 18):
            _no_meal_features = _no_meal_features.append(feature_extractor(glucoseSlice2), ignore_index=True)
            arrOfNoMealFrames.append(glucoseSlice2)



def feature_extractor(data):
    cgmMealIndex = None
    cgmMaxIndex = None
    cgmMaxRow = None
    cgmMealTimeRow = None
    if('meal_time' in data):
        # feature 1 basic
        cgmMealIndex = data['DateTime'].sub(data['meal_time'].values[0]).abs().idxmin()
        cgmMaxIndex = data['Sensor Glucose (mg/dL)'].idxmax()
        cgmMaxRow = data.loc[cgmMaxIndex]
        cgmMealTimeRow = data.loc[cgmMealIndex]
        dG = cgmMaxRow['Sensor Glucose (mg/dL)'] - cgmMealTimeRow['Sensor Glucose (mg/dL)']
        
        # feature 2 time delta
        timeDiff = cgmMaxRow['DateTime'] - cgmMealTimeRow['DateTime']
        timeDiff = timeDiff.total_seconds()

        # feature 3 dG Normalized
        dGNorm = dG / cgmMealTimeRow['Sensor Glucose (mg/dL)']
    else:
        dG = data.iloc[-1]['Sensor Glucose (mg/dL)'] - data.iloc[0]['Sensor Glucose (mg/dL)']
        timeDiff = data.iloc[-1]['DateTime'] - data.iloc[0]['DateTime']
        timeDiff = timeDiff.total_seconds()
        dGNorm = dG / data.iloc[0]['Sensor Glucose (mg/dL)']


    # feature 4, 5, 6, 7 FFT
    fftPower = np.abs(np.fft.fft(data['Sensor Glucose (mg/dL)']))**2
    fftPowerTemp = fftPower
    fftPowerTemp = np.delete(fftPower, 0)
    index = np.argmax(fftPowerTemp)
    fftFreq = np.fft.fftfreq(len(data['Sensor Glucose (mg/dL)']))
    # fftFreq = np.delete(fftFreq, 0)
    p1 = fftPower[1]
    f1 = fftFreq[1]
    p2 = fftPower[index]
    f2 = fftFreq[index]

    # feature 8 first derivative
    # feature 9 second derivative
    times = data['DateTime'].astype('datetime64[s]').astype('int64')
    if('meal_time' in data):
        firstD = np.gradient([cgmMealTimeRow['Sensor Glucose (mg/dL)'],cgmMaxRow['Sensor Glucose (mg/dL)']], [times[cgmMealIndex], times[cgmMaxIndex]])
        secondD = np.gradient(firstD)
    else:
        firstD = np.gradient([data.iloc[0]['Sensor Glucose (mg/dL)'],data.iloc[-1]['Sensor Glucose (mg/dL)']], [times.iloc[0], times.iloc[-1]])
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

def main():
    global _no_meal_features
    global _meal_features
    cgm_data_file = 'CGMData.csv'
    insulin_data_file = 'InsulinData.csv'
    p2cgm_data_file = 'CGM_patient2.csv'
    p2insulin_data_file = 'Insulin_patient2.csv'

    # Read CSV Files
    cgmFrame = pd.read_csv(cgm_data_file, low_memory=False)
    p2cgmFrame = pd.read_csv(p2cgm_data_file, low_memory=False)
    insulinFrame = pd.read_csv(insulin_data_file, low_memory=False)
    p2insulinFrame = pd.read_csv(p2insulin_data_file, low_memory=False)

    # Remove unwanted columns
    cgmData = cgmFrame[['Index','Date', 'Time', 'Sensor Glucose (mg/dL)', 'ISIG Value']]
    p2cgmData = p2cgmFrame[['Index','Date', 'Time', 'Sensor Glucose (mg/dL)', 'ISIG Value']]
    insulinData = insulinFrame[['Index','Date', 'Time', 'BWZ Carb Input (grams)']]
    p2insulinData = p2insulinFrame[['Index','Date', 'Time', 'BWZ Carb Input (grams)']]

    # extract meal data, interpolate missing values
    mealRows = extractMealData(insulinData)
    mealRows2 = extractMealData(p2insulinData)
    cgmData.interpolate(inplace=True)
    p2cgmData.interpolate(inplace=True)

    # extract glucose data and the features for each matrix
    extractGlucoseData(cgmData, mealRows)
    extractGlucoseData(p2cgmData, mealRows2)
    meal_features = pd.concat([_meal_features, _no_meal_features])
    meal_features.reset_index(drop=True, inplace=True)
    meal_features.fillna(0, inplace=True)

    # create label vector
    ones = np.ones(len(_meal_features))
    zeroes = np.zeros(len(_no_meal_features))
    labelVector = np.concatenate((ones, zeroes), axis=None)
    x_train, x_test, y_train, y_test = train_test_split(meal_features, labelVector, test_size=0.2, random_state=42)

    # run data through model
    model = LinearSVC(random_state=0, tol=1e-5)
    model.fit(x_train, y_train.ravel())
    pred = model.predict(x_test)

    # k fold cross validation with 10 splits
    kVal = KFold(n_splits=10, random_state=None)
    scores = cross_val_score(model, meal_features, labelVector, scoring='accuracy', cv=kVal, n_jobs=-1)
    print('Accuracy: %.3f (%.3f)' % (np.mean(scores), np.std(scores)))
    # print("Accuracy:",metrics.accuracy_score(y_test, pred))
    modelFile = 'model.sav'

    # save to pickle file
    pickle.dump(model, open(modelFile,'wb'))

if __name__ == "__main__":
    main()