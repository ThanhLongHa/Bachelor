#Import
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import time
import os
import pandas as pd

# ProzessId erheben
pid = os.getpid()
print("ProcessID: ",pid)

# Startzeit aufzeichnen
start_time= time.time()
print(time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()))

# Daten auslesen
df_train = pd.read_csv("D:/Uni/Bachelor/ProjectData/DigitRecognizer/train.csv")
df_test = pd.read_csv("D:/Uni/Bachelor/ProjectData/DigitRecognizer/test.csv")

# Daten vorbereiten
X_train = df_train.drop("label", axis=1)
y_train = df_train.label

# Algorithmus starten
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
pred = knn.predict(df_test)

# Ergebnisse in einer CSV Datei speichern
predictions = pd.Series(knn.predict(df_test))
ImageId = pd.Series(np.arange(1, len(df_test) + 1))
submission = pd.concat([ImageId, predictions], axis=1)
submission.columns = ('ImageId', 'Label')
#submission.to_csv('Python_KNN_Abgabe.csv', index=False)

# Endzeit aufzeichnen
end_time = time.time()
print(time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()))

#Ausführungsdauer ausgeben
print('runtime=',end_time-start_time)