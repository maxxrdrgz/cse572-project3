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
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
import math
import sys

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
        glucoseSlice['carb_intake'] = insulinDf.iloc[i]['BWZ Carb Input (grams)']
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

    # Read CSV Files
    cgmFrame = pd.read_csv(cgm_data_file, low_memory=False)
    insulinFrame = pd.read_csv(insulin_data_file, low_memory=False)

    # Remove unwanted columns
    cgmData = cgmFrame[['Index','Date', 'Time', 'Sensor Glucose (mg/dL)', 'ISIG Value']]
    insulinData = insulinFrame[['Index','Date', 'Time', 'BWZ Carb Input (grams)']]

    # extract meal data, interpolate missing values
    mealRows = extractMealData(insulinData)
    cgmData.interpolate(inplace=True)
    extractGlucoseData(cgmData, mealRows)
    
    minCarb = mealRows['BWZ Carb Input (grams)'].min()
    maxCarb = mealRows['BWZ Carb Input (grams)'].max()
    nBins = math.ceil((maxCarb-minCarb)/20)
    print(maxCarb)
    print(minCarb)
    print(nBins)
    cats = range(nBins)
    mealRows['bin'] = pd.qcut(mealRows['BWZ Carb Input (grams)'], q=nBins, labels=cats)
    gtAggregateDf = mealRows.groupby(pd.qcut(mealRows['BWZ Carb Input (grams)'], q=nBins, labels=cats)).count()
    groundTruthDf = pd.DataFrame({'points': gtAggregateDf['Index']})
    kmeans = KMeans(n_clusters=nBins)
    _meal_features = _meal_features.fillna(0)
    kmeansFit = kmeans.fit(_meal_features)
    kmeansPred = kmeansFit.predict(_meal_features)
    _meal_features['Cluster'] = kmeansPred
    kmeansGrouped = _meal_features.groupby('Cluster').apply(list)
    print(mealRows)
    sys.exit()


    _meal_features_scaled = StandardScaler().fit_transform(_meal_features)
    _meal_features_normalized = normalize(_meal_features_scaled)
    _meal_features_normalized = pd.DataFrame(_meal_features_normalized)

    pca = PCA(n_components = 2)
    _meal_features_principal = pca.fit_transform(_meal_features_normalized)
    _meal_features_principal = pd.DataFrame(_meal_features_principal)
    _meal_features_principal.columns = ['P1', 'P2']
    print(_meal_features_principal)

    # neigh = NearestNeighbors(n_neighbors=2)
    # nbrs = neigh.fit(_meal_features_principal)
    # distances, indices = nbrs.kneighbors(_meal_features_principal)
    # distances = np.sort(distances, axis=0)
    # # np.set_printoptions(threshold=np.inf)
    # distances = distances[:,1]
    # # tempDist = str(distances)

    # print(distances)
    defaultEps = 0.1
    n_clusters_ = 0
    while(n_clusters_ != nBins):
        db = DBSCAN(eps=defaultEps, min_samples=5).fit(_meal_features_principal)
        labels = db.labels_
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise_ = list(labels).count(-1)
        if(n_clusters_ > nBins):
            defaultEps = defaultEps + 0.0001
        elif(n_clusters_ < nBins):
            defaultEps = defaultEps - 0.0001
        print(defaultEps)
        print("Estimated number of clusters: %d" % n_clusters_)
        print("Estimated number of noise points: %d" % n_noise_)

    dbscanSSE = 0
    cluster_centers = {}


    for i in range(nBins):
        points_of_cluster = _meal_features_principal.to_numpy()[labels==i,:]
        print(points_of_cluster)
        sys.exit()
        centroid_of_cluster= np.mean(points_of_cluster, axis=0)
        cluster_centers[i] = centroid_of_cluster

        
        
    # cluster_centers = db.cluster_centers_
    for point, label in zip(_meal_features_principal.to_numpy(), labels):
        if(label != -1):
            dbscanSSE += np.square(point - cluster_centers[label]).sum()


    print(pd.Series(labels).value_counts())


    # resultsList = [kmeansFit.inertia_, dbscanSSE, 0, 0, 0, 0]
    
    # resultsDf = pd.DataFrame(resultsList).T
    # resultsDf.to_csv('./Result.csv', index=False, header=False)


    # Number of clusters in labels, ignoring noise if present.
    # n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    # n_noise_ = list(labels).count(-1)
    # # print(kmeansGrouped)
    # print("Estimated number of clusters: %d" % n_clusters_)
    # print("Estimated number of noise points: %d" % n_noise_)

    # meal_features = pd.concat([_meal_features, _no_meal_features])
    # meal_features.reset_index(drop=True, inplace=True)
    # meal_features.fillna(0, inplace=True)


if __name__ == "__main__":
    main()