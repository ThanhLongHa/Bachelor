#Importe von Bibliotheken
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score

# Daten laden
df = pd.read_csv("data.csv")

# Aufteilung in Trainings- und Tesdaten
X = df.drop('Label', axis=1)
y = df['Label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# Machine Learning Algorithmus anwenden
model = LogisticRegression()
model.fit(X_train, y_train)

# Klassifizierung durchführen
y_pred = model.predict(X_test)

# Konfusionsmatrix erstellen
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix: \n", conf_matrix)

# Genauigkeit berechnen
acc = accuracy_score(y_test, y_pred)
print("Accuracy: ", acc)