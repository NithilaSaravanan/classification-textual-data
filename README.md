# Classification of textual data on two benchmark datasets

The primary motive of this project is to implement and compare five major classification algorithms, Logistic Regression,
Decision Trees, Linear SVC, AdaBoost, and RandomForest. Not every model is suited to every set of data and their relative
strengths can be shown by comparing them on two different
types of datasets. These five algorithms, plus Multinomial
Naive Bayes, were employed on 2 different datasets. The first
dataset, 20 News Groups is a multi class problem, whereas
the second, IMDb reviews, is a binary classification problem.
- It is important to pre-process data in order to get accurate
results. It may have been more effective for the IMDB
database of reviews if words like actor, movie, film, were
stripped out to only focus on sentiment holding words.
- Some classification models are more suited to certain
types of problems. MultiNB was the most accurate for
multi-class, yet lagged a little compared to Linear SVC
for binary classification.
- Cross validation can substantially improve the training
model.

### Abstract from the report

The performance of various Machine
Learning (ML) algorithms, namely Logistic Regression (LR),
Decision Trees (DT), Support Vector Machine (SVM), AdaBoost
(AB), Random Forest (RF) and Multinomial Na¨ıve Bayes (MNB)
was investigated on two benchmark datasets. The text data was
preprocessed where whitespace, punctuation and stop words were
removed. The text was further processed through lemmatization.
POS tagging was introduced before vectorizing the dataset.
GridSearchCV was utilized to find the best hyperparameters and
the winning algorithms were identified.
