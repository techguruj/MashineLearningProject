# Einführung
![image](https://github.com/user-attachments/assets/9e96760c-f911-4d59-86f8-4302ed1ebe1f)

Dies ist unser Abschlussprojekt. 
Vielen Dank an die folgenden Mitglieder für ihren Beitrag: Jinwen, Antonio, Jad, Lina, Mohamed.

Das Thema dieses Projekts ist die Vorhersage, ob Mitarbeiter des Unternehmens kündigen werden, basierend auf statistischen Daten zu 1130 Mitarbeitern in den folgenden 15 Dimensionen und unter Verwendung von maschinellen Lernmodellen sowie der Bewertung der Vorhersageergebnisse.


15 Dimensionen: 
-stag, event, gender, age, industry, profession, traffic, coach, head_gender, greywage, way, extraversion, independ, selfcontrol, anxiety, novator

Verwendete Modelle:
-Bayes Gaussian, Decision Tree, Random Forest, Logistic Regression, KNN (k-nearest neighbors), SVC (Support Vector Classification), MLP (Multilayer perceptron)

Erforderliche Bibliotheken:
-Scikit-Learn, Numpy, Pandas, Matplotlib, Seaborn

# Ablauf

### 1. Datenanalyse
- Mögliche Zusammenhänge zwischen den Daten finden
- 
![image](https://github.com/user-attachments/assets/e43dd75d-5621-4a78-9f00-252172519db2)

### 2. Datenvorbereitung
- Ausreißer und fehlende Werte entfernen
- Mit PCA (Principal Component Analysis) prüfen, ob eine Dimensionreduzierung möglich ist
- ![image](https://github.com/user-attachments/assets/09e089b8-b4a5-4528-9014-be73a0682487)

- Dimensionen mit zu vieler Unterdimensionen möglichst auf zwei Unterdimensionen reduzieren
- Alle Dimensionen mit One-Hot-Encoding kodieren
- 
### 3. Vorhersagen mit verschiedenen Modellen des maschinellen Lernens

Bayes Gaussian, Decision Tree, Random Forest, Logistic Regression, KNN (k-nearest neighbors), SVC (Support Vector Classification), MLP (Multilayer perceptron)

- Iterieren und die optimalen Parameter finden
![image](https://github.com/user-attachments/assets/a11de663-cf3f-4453-9561-b1ba0f92e1d3)

### 4. Die optimalen Parameter aller oben genannten Modelle mit dem VotingClassifier zu einem Modell kombinieren und erneut vorhersagen

![image](https://github.com/user-attachments/assets/1d52d975-4997-4376-90e9-06c97ac7f433)

# Fazit

Aufgrund der Zeitbeschränkung (3 Tage) habe ich alle Unterdimensionen auf 2 Dimensionen reduziert und anschließend eine One-Hot-Codierung vorgenommen, was die Genauigkeit der Vorhersageergebnisse beeinträchtigt hat. Wir wissen jedoch, dass ohne Reduktion der Unterdimensionen die Genauigkeit der Vorhersageergebnisse signifikant auf 78 % erhöht werden kann. Wenn wir mehr Zeit hätten, könnten wir ein besseres Modell trainieren.
