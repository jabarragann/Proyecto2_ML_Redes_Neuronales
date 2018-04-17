import pandas as pd

def createDataSet():
    data = pd.read_csv('exerciseData/msd_genre_dataset.csv')

    metalPunkData = data[(data.genre == 'metal') | (data.genre == 'punk') ]
    metalPunkData.to_csv("exerciseData/Metal_Punk.csv")

    danceElectronicaData = data[(data.genre == 'dance and electronica')]
    danceElectronicaData.to_csv("exerciseData/Dance_Electronica.csv")

    # Remove Track_id, artist_name, title
    print("Dance data size:",danceElectronicaData.shape[0])
    print("Metal & Punk data size:",metalPunkData.shape[0])

    headers = list(danceElectronicaData.columns.values)
    danceElectronicaData = danceElectronicaData.drop(labels=[headers[1],headers[2],headers[3]],axis=1)
    metalPunkData = metalPunkData.drop(labels=[headers[1],headers[2],headers[3]],axis=1)

    #Create Training Set and Test set
    df1 = danceElectronicaData.sample(frac=0.2, replace=False)
    danceElectronicaData = danceElectronicaData.drop(df1.index.values)

    df2 = metalPunkData.sample(frac=0.2, replace=False)
    metalPunkData = metalPunkData.drop(df2.index.values)

    testSet = pd.concat([df1, df2])
    trainingSet = pd.concat([danceElectronicaData,metalPunkData])

    testSet = testSet.sample(frac=1, replace=False)
    testSet.index.name='data_base_index'
    trainingSet = trainingSet.sample(frac=1, replace=False)
    trainingSet.index.name="data_base_index"

    print("Training set size:", trainingSet.shape[0])
    print("Test set size:", testSet.shape[0])

    # Save Training Set And Test Set
    trainingSet.to_csv("exerciseData/trainingSetGenre.csv")
    testSet.to_csv("exerciseData/testSetGenre.csv")

def splitIntoFeatureMatrixAndLabels(dataSet):

    yData = dataSet['genre'].values
    xData = dataSet.drop(labels=['genre', "data_base_index"], axis=1).values

    for i in range(yData.shape[0]):
        if yData[i] == "dance and electronica":
            yData[i] = 1
        else:
            yData[i] = 0

    return xData,yData