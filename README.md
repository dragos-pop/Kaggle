# Ended Kaggle Competitions

## 1) Tabular Playground Series - Oct 2022
- Period: October 1, 2022 - October 31, 2022
- Objective: predict the probability of each team scoring within the next 10 seconds of the game given a snapshot from a Rocket League match
- Data: sequences of snapshots of the state of a Rocket League match, including position and velocity of all players and the ball, as well as extra information
- Task: 2-target Regression
- Evaluation Metric: Mean log loss of the target variables
- Algorithms: LightGBM Regressor, LightGBM Classifier (predict_proba), CatBoost Regressor
- Result: 0.20210 (253/463 - Top 55%)
- Key insights: removing most of the snapshots (95%) saves space, improves fitting time, and does not have negative implications over the result; ball velocity and its distance from the y=0 axis are the most important features in predicting the probability of a team to score within the next 10 seconds

## 2) Playground Series Season 3, Episode 2
- Period: January 10, 2023 - January 16, 2023
- Objective: predict the probability of having a stroke given some clinical features about each patient
- Data: 11 clinical features, such as the the gender, BMI, average glucose level, etc
- Task: Regression
- Evaluation Metric: ROC AUC
- Algorithm: LightGBM Regressor, LightGBM Classifier (predict_proba), CatBoost Regressor, CatBoost Classifier (predict_proba)
- Result: 0.87851 (606/770 - Top 79%)
- Key insights: classifier.predict_proba(test) gives the probability of each sample to belong to the first class, which is not having a stroke -> it gives the probability of NOT having a stroke (1-predict_proba(test) gives the probability of having a stroke); SMOTE improves the results significantly given the large class imbalance, however, it only works when data is fully composed of numerical features

## 3) Goodreads Books Reviews
- Period: Mar 29, 2022 - Apr 2, 2023
- Objective: predict book review ratings which range from 0 to 5
- Data: approximately 0.9M book reviews, containing the book, the author, the review text, and its stats
- Task: Ordinal Regression
- Evaluation Metric: F1 score
- Algorithm: LightGBM Classifier (predict_proba), CatBoost Classifier (predict_proba)
- Result: 0.39318 (187/261 - Top 72%)
- Key insight: classifier.predict_proba() generally used for ordinal regression

## 4) Competition Name
- Period: Month Year - Month Year
- Objective: 
- Data: 
- Task: Regression/ Binary Classification/ Multi-class Classification/ Clustering
- Evaluation Metric: 
- Algorithm: 
- Result: 
- Key insights: 
