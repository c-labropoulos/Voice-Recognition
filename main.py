import numpy as np
import os
import librosa
from sklearn.mixture import GaussianMixture
traindatapath = '/Voice Dataset/train/'
n_mfcc=39
n_components=35

trained_gaussian=[]
trained_labelname = []
for file in os.listdir(traindatapath):
    if file.endswith(".wav"):
        y, sr = librosa.load(traindatapath+file)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        gaussian = GaussianMixture(n_components=n_components,random_state=3)
        mfcc_t = np.transpose(mfcc)
        gaussian = gaussian.fit(mfcc_t)
        trained_gaussian.append(gaussian)
        trained_labelname.append(file[1])
testdatapath = '/Voice Dataset/test/'
test_labelname = []
actual_labelname = []
predicted_labelnames = []
for file in os.listdir(testdatapath):
    if file.endswith(".wav"):
        y, sr = librosa.load(testdatapath + file)

        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        mfcc = np.transpose(mfcc)

        actual_labelname.append(file[1])
        prediction=[]
        for gaussian in trained_gaussian:
            prediction.append(gaussian.score(mfcc))
        predicted_labelnames.append(trained_labelname[prediction.index(max(prediction))])
#print("Real labels:")
#rint(actual_labelname)
#print("predicted labels:")
#print(predicted_labelnames)

i = 0
count = 0
while (i < len(actual_labelname)):
    if actual_labelname[i] == predicted_labelnames[i]:
        print("Predicted Label <<"+str(predicted_labelnames[i])+">> matches with the real label <<"+str(actual_labelname[i])+">>")
        count += 1

    i += 1

print("Precision: " + str(count / len(actual_labelname)) + "%")
