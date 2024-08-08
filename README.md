#Einführung

Dies ist unser Abschlussprojekt. Vielen Dank an die folgenden Mitglieder für ihren Beitrag: Jinwen, Antonio, Jad, Lina, Mohamed.
Das Thema dieses Projekts ist die Vorhersage, ob Mitarbeiter des Unternehmens kündigen werden, basierend auf statistischen Daten zu 1130 Mitarbeitern in den folgenden 15 Dimensionen und unter Verwendung von maschinellen Lernmodellen sowie der Bewertung der Vorhersageergebnisse.

15 Dimensionen: stag, event, gender, age, industry, profession, traffic, coach, head_gender, greywage, way, extraversion, independ, selfcontrol, anxiety, novator

Verwendete Modelle:
Bayes Gaussian, Decision Tree, Random Forest, Logistic Regression, KNN (k-nearest neighbors), SVC (Support Vector Classification), MLP (Multilayer perceptron)

Erforderliche Bibliotheken:

Scikit-Learn, Numpy, Pandas, Matplotlib, Seaborn

#Ablauf

##1. Datenanalyse
- Mögliche Zusammenhänge zwischen den Daten finden
##2. Datenvorbereitung
- Ausreißer und fehlende Werte entfernen
- Mit PCA (Principal Component Analysis) prüfen, ob eine Dimensionreduzierung möglich ist
- Dimensionen mit unzähligen Unterdimensionen möglichst auf zwei Unterdimensionen reduzieren
- Alle Dimensionen mit One-Hot-Encoding kodieren
##3. Vorhersagen mit verschiedenen Modellen des maschinellen Lernens

Bayes Gaussian, Decision Tree, Random Forest, Logistic Regression, KNN (k-nearest neighbors), SVC (Support Vector Classification), MLP (Multilayer perceptron)

- Iterieren und die optimalen Parameter finden
##4. Die optimalen Parameter aller oben genannten Modelle mit dem VotingClassifier zu einem Modell kombinieren und erneut vorhersagen
