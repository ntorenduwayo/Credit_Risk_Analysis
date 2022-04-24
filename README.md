# Credit_Risk_Analysis
## Overview of the analysis
In this project, we used Python to analyze a credit card credit dataset from LendingClub a lending service company to predict credit risk. The goal is to apply Machine Learning algorithms to uncover the hidden patterns or trends and use them to make forecasts about how much risk a given loan carries.</br>
We evaluate and compared the performance of six different Machine Learning algorithms (i.e., *RandomOverSampler, *SMOTE, *ClusterCentroids, *SMOTEENN, *BalancedRandomForestClassifier, and *EasyEnsembleClassifier) to make conclusion and recommendations to the company about the best model that can improve the business efficiency by eliminating most of the high-risk loans.
## Results 
### Oversampling model via naive random oversampling algorithm
#### Table 1: Accuracy Score:
<img width="358" alt="An accuracy score model 1 - naive random oversampling " src="https://user-images.githubusercontent.com/34750363/164576322-7c5347d4-7195-4ff6-b814-6d20dca545cc.png">
#### Table 2: Confusion Matrix
<img width="273" alt="Confusion matrix model 1 - naive random oversampling " src="https://user-images.githubusercontent.com/34750363/164576386-24588989-43e1-4c61-acb5-b0d99587ef4f.png">
#### Table 3: Imbalanced Classification Report
<img width="381" alt="Imbalanced classification report model 1 - naive random oversampling" src="https://user-images.githubusercontent.com/34750363/164576441-542b9040-91b3-4fba-a4c4-2f5bbda72669.png">  </br>
* Accuracy score</br>
The Naïve Random Oversampling model had about 0.65 **accuracy score** meaning that the model classifier is approximately 65% accurate in general. However, because our dataset is not symmetric (i.e., class imbalance or where values of false positive and false negatives are almost the same), this measure may be misleading. Consequently, we used other parameters (e.g., Precision, recall, F scores, area under ROC curves) to further evaluate the model performance.
*	Precision Rate</br>
From the Imbalanced Classification Report Table 3 above, the model had a high-risk loan **precision rate** of about 0.01 or 1% meaning that out of 100 transactions classified as high risky loans, only 1 transaction was correctly classified as a high-risk loan. On the other hand, the **precision rate** of low-risk loans was 1.00 or 100% meaning that out of 100 transactions classified as low risk-loans, all 100 transactions were correctly classified as low-risk loans.
*	Recall (i.e., Sensitivity)</br>
The **sensitivity** was 0.63 or 63% for the high-risk loans meaning that of all the loans that were truly high-risk the model correctly labeled 63% as high-risk loans which is very good as it is above 50%.  The low-risk loans had 0.66 or 66% sensitivity which is a little bit better than the high-risk loan.
*	F-1 Score i.e., the harmonic mean between precision & recall</br>
The **F-1 Score** was 0.02 for the high-risk loans, and 0.80 for the low-risk loans.
*	Support i.e., the number of occurrences of the given class (High-risk loan, or low-risk loans) in our dataset.
The dataset had 87 High-risk loans class and 17,118 of low-risk loans class, which is a not a well-balanced dataset.
### Oversampling model via SMOTE (Synthetic Minority Oversampling Technique) 
#### Table 1: Accuracy Score:
<img width="292" alt="Accuracy score for the model 2 - SMOTE Oversampling" src="https://user-images.githubusercontent.com/34750363/164576517-cef9045a-170e-4d0c-97b0-9ad8adf41dc6.png">
#### Table 2: Confusion Matrix
<img width="227" alt="Confusion matrix model 2 - SMOTE Oversampling" src="https://user-images.githubusercontent.com/34750363/164576618-8c7e071d-ef2a-48e1-ba0d-2224251179d6.png">
#### Table 3: Imbalanced Classification Report
<img width="421" alt="Imbalanced classification report model 2 - SMOTE Oversampling" src="https://user-images.githubusercontent.com/34750363/164576666-7c2f2d85-74a0-42b6-9fb2-c360ba8b0b38.png">
*	Accuracy score
The SMOTE model had about 0.62 **accuracy score** meaning that the model classifier is approximately 62% accurate in general. However, because our dataset is not symmetric (i.e., class imbalance or where values of false positive and false negatives are almost the same), this measure may be misleading. Consequently, we used other parameters (e.g., Precision, recall, F scores, area under ROC curves) to further evaluate the model performance.
*	Precision Rate
From the Imbalanced Classification Report Table 3 above, the model had a high-risk loan **precision rate** of about 0.01 or 1% meaning that out of 100 transactions classified as high risky loans, only 1 transaction was correctly classified as a high-risk loan. On the other hand, the **precision rate** of low-risk loans was 1.00 or 100% meaning that out of 100 transactions classified as low risk-loans, all 100 transactions were correctly classified as low-risk loans.
*	Recall (i.e., Sensitivity)
The sensitivity was 0.62 or 62% for the high-risk loans meaning that of all the loans that were truly high-risk the model correctly labeled 62% as high-risk loans which is very good as it is above 50%.  The low-risk loans had 0.63 or 63% sensitivity which is lower than the high-risk loan.
*	F-1 Score i.e., the harmonic mean between precision & recall
The high-risk and low-risk loans had an **F-1 Score** of 0.02 and 0.77 respectively.
*	Support i.e., the number of occurrences of the given class (High-risk loan, or low-risk loans) in our dataset.
We had 87 of High-risk loans class and 17,118 of low-risk loans class, which is a not a well-balanced dataset.


### Undersampling model via Cluster Centroids algorithm
#### Table 1: Accuracy Score
<img width="298" alt="Accuracy score for the model 3 - Undersampling Cluster Centroids" src="https://user-images.githubusercontent.com/34750363/164576739-1b65b189-4245-4981-ac50-91a5c0fdd4c8.png">
#### Table 2: Confusion Matrix
<img width="278" alt="Confusion matrix model 3 - Undersampling Cluster Centroids" src="https://user-images.githubusercontent.com/34750363/164576815-3788d1c8-8ec3-41c0-b0b8-558f1d6e7611.png">
#### Table 3: Imbalanced Classification Report
<img width="368" alt="Imbalanced classification report model 3 - Undersampling Cluster Centroids" src="https://user-images.githubusercontent.com/34750363/164576874-d6c1e2bc-7be3-4d05-a5d9-0a6c0cbf73bb.png">
*	Accuracy score
The ClusterCentroids model had about 0.52 **accuracy score** meaning that the model classifier is approximately 52% accurate in general. However, because our dataset is not symmetric (i.e., class imbalance or where values of false positive and false negatives are almost the same), this measure may be misleading. Consequently, we used other parameters (e.g., Precision, recall, F scores, area under ROC curves) to further evaluate the model performance.
*	Precision Rate
From the Imbalanced Classification Report Table 3 above, the model had a high-risk loan **precision rate** of about 0.01 or 1% meaning that out of 100 transactions classified as high risky loans, only 1 transaction was correctly classified as a high-risk loan. On the other hand, the **precision rate** of low-risk loans was 1.00 or 100% meaning that out of 100 transactions classified as low risk-loans, all 100 transactions were correctly classified as low-risk loans.
*	Recall (i.e., Sensitivity)
The sensitivity was 0.57 or 57% for the high-risk loans meaning that of all the loans that were truly high-risk, the model correctly labeled 57% as high-risk loans which is very good as it is above 50%.  The low-risk loans had 0.47 or 47% sensitivity which is not good since is below 50%.
*	F-1 Score i.e., the harmonic mean between precision & recall
The high-risk and low-risk loans had an **F-1 Score** of 0.01 and 0.64 respectively.
*	Support i.e., the number of occurrences of the given class (High-risk loan, or low-risk loans) in our dataset.
We had 87 of High-risk loans class and 17,118 of low-risk loans class, which is a not a well-balanced dataset.

### Over-and undersampling model via SMOTEENN
#### Table 1: Accuracy Score
<img width="237" alt="Accuracy score model 4 - SMOTEENN" src="https://user-images.githubusercontent.com/34750363/164576933-452de2b2-3424-4ea7-a419-7583a7b094bc.png">
#### Table 2: Confusion Matrix
<img width="277" alt="Confusion matrix model 4 - SMOTEENN" src="https://user-images.githubusercontent.com/34750363/164577017-85cc9f20-608d-48c9-96cb-f098a7a0e5b9.png">
#### Table 3: Imbalanced Classification Report
<img width="421" alt="Imbalanced classification report model 4 -SMOTEENN" src="https://user-images.githubusercontent.com/34750363/164577079-cc49737f-a80a-4185-b193-f1700278d460.png">
*	Accuracy score
The SMOTEENN model had about 0.65 **accuracy score** meaning that the model classifier is approximately 65% accurate in general. However, because our dataset is not symmetric (i.e., class imbalance or where values of false positive and false negatives are almost the same), this measure may be misleading. Consequently, we used other parameters (e.g., Precision, recall, F scores, area under ROC curves) to further evaluate the model performance.
*	Precision Rate
From the Imbalanced Classification Report Table 3 above, the model had a high-risk loan **precision rate** of about 0.01 or 1% meaning that out of 100 transactions classified as high risky loans, only 1 transaction was correctly classified as a high-risk loan. On the other hand, the **precision rate** of low-risk loans was 1.00 or 100% meaning that out of 100 transactions classified as low risk-loans, all 100 transactions were correctly classified as low-risk loans.
*	Recall (i.e., Sensitivity)
The sensitivity was 0.69 or 69% for the high-risk loans meaning that of all the loans that were truly high-risk, the model correctly labeled 69% as high-risk loans which is very good as it is above 50%.  The low-risk loans had 0.61 or 61% sensitivity which is good since is above 50%. But it is a little bit less than the high-risk loans.
*	F-1 Score i.e., the harmonic mean between precision & recall
The high-risk and low-risk loans had an **F-1 Score** of 0.02 and 0.76 respectively.
*	Support i.e., the number of occurrences of the given class (High-risk loan, or low-risk loans) in our dataset.
We had 87 of High-risk loans class and 17,118 of low-risk loans class, which is a not a well-balanced dataset.

### BalancedRandomForestClassifier model 
#### Table 1: Accuracy Score:
<img width="254" alt="Accuracy score model 5 - BalancedRandomForestClassifier" src="https://user-images.githubusercontent.com/34750363/164577125-d19a445c-86b4-46ca-9a1e-333e77fc9c31.png">
#### Table 2: Confusion Matrix
<img width="244" alt="Confusion matrix model 5 - BalancedRandomForestClassifier" src="https://user-images.githubusercontent.com/34750363/164577173-838d0cd2-5231-485a-9d88-651b5471cfa7.png">
#### Table 3: Imbalanced Classification Report
<img width="398" alt="Imbalanced classification report model 5 - BalancedRandomForestClassifier" src="https://user-images.githubusercontent.com/34750363/164577229-1cb56776-6970-468c-960d-4fe946d4637c.png">
*	Accuracy score
The BalancedRandomForestClassifier model had about 0.78 **accuracy score** meaning that the model classifier is approximately 78% accurate in general. However, because our dataset is not symmetric (i.e., class imbalance or where values of false positive and false negatives are almost the same), this measure may be misleading. Consequently, we used other parameters (e.g., Precision, recall, F scores, area under ROC curves) to further evaluate the model performance.
*	Precision Rate
From the Imbalanced Classification Report Table 3 above, the model had a high-risk loan **precision rate** of about 0.04 or 4% meaning that out of 100 transactions classified as high risky loans, only 4 transactions were correctly classified as a high-risk loans. On the other hand, the **precision rate** of low-risk loans was 1.00 or 100% meaning that out of 100 transactions classified as low risk-loans, all 100 transactions were correctly classified as low-risk loans.
*	Recall (i.e., Sensitivity)
The sensitivity was 0.67 or 67% for the high-risk loans meaning that of all the loans that were truly high-risk, the model correctly labeled 67% as high-risk loans which is very good as it is above 50%.  The low-risk loans had 0.91 or 91% sensitivity which is good since is above 50%. This was much better than the high-risk loans.
*	F-1 Score i.e., the harmonic mean between precision & recall
The high-risk and low-risk loans had an **F-1 Score** of 0.07 and 0.95 respectively.
*	Support i.e., the number of occurrences of the given class (High-risk loan, or low-risk loans) in our dataset.
We had 87 of High-risk loans class and 17,118 of low-risk loans class, which is a not a well-balanced dataset.

### EasyEnsembleClassifier model
#### Table 1: Accuracy Score
<img width="238" alt="Accuracy score model 6 - EasyEnsembleClassifier" src="https://user-images.githubusercontent.com/34750363/164577325-ef2f56f0-5c05-417b-b3a1-04061862924c.png">
#### Table 2: Confusion Matrix
<img width="243" alt="Confusion matrix model 6 - EasyEnsembleClassifier" src="https://user-images.githubusercontent.com/34750363/164577397-1dc71f24-2087-4254-8175-5af423e58b27.png">
#### Table 3: Imbalanced Classification Report
<img width="406" alt="Imbalanced classification report model 6 - EasyEnsembleClassifier" src="https://user-images.githubusercontent.com/34750363/164577463-2f7874c4-c141-4c10-bbde-69b30e82c15b.png">
*	Accuracy score
The EasyEnsembleClassifiermodel had about 0.92 **accuracy score** meaning that the model classifier is approximately 92% accurate in general. However, because our dataset is not symmetric (i.e., class imbalance or where values of false positive and false negatives are almost the same), this measure may be misleading. Consequently, we used other parameters (e.g., Precision, recall, F scores, area under ROC curves) to further evaluate the model performance.
*	Precision Rate
From the Imbalanced Classification Report Table 3 above, the model had a high-risk loan **precision rate** of about 0.07 or 7% meaning that out of 100 transactions classified as high risky loans, only 7 transactions were correctly classified as a high-risk loans. On the other hand, the **precision rate** of low-risk loans was 1.00 or 100% meaning that out of 100 transactions classified as low risk-loans, all 100 transactions were correctly classified as low-risk loans.
*	Recall (i.e., Sensitivity)
The sensitivity was 0.91 or 91% for the high-risk loans meaning that of all the loans that were truly high-risk, the model correctly labeled 91% as high-risk loans which is very good as it is above 50%.  The low-risk loans had 0.94 or 94% sensitivity which is good since is above 50%. This was better than the high-risk loans.
*	F-1 Score i.e., the harmonic mean between precision & recall
The high-risk and low-risk loans had an **F-1 Score** of 0.14 and 0.97 respectively.
*	Support i.e., the number of occurrences of the given class (High-risk loan, or low-risk loans) in our dataset.
We had 87 of High-risk loans class and 17,118 of low-risk loans class, which is a not a well-balanced dataset.

## Summary
We found that the Naïve Random Oversampling model, SMOTE model, ClusterCentroids model, SMOTEENN model, BalancedRandomForestClassifier model, and EasyEnsembleClassifier model had an accuracy scores of 65%, 62%, 52%, 65%, 79%, and 93% respectively. Thus, using the accuracy score the EasyEnsembleClassifier model (93% correct overall) followed by the BalancedRandomForestClassifier model (79% correct overall) performed better than the other models, while the ClusterCentroids model (Only 52% correct overall) was the worst. However, due to the dataset class imbalance, accuracy alone could not be used to sufficiently evaluate these models. To that fact, we used the precision score, the recall or sensitivity score, and the F-1 score. Although, both the EasyEnsembleClassifier and BalancedRandomForestClassifier models had 100% precision score for the low-risk loans (i.e., Out of 100 transactions classified as low risk-loans, all 100 transactions were correctly classified as low-risk loans), their high-risk loans precision scores were low (7% and 4% respectively). The sensitivity score for EasyEnsembleClassifier model was very good for both high-risk and low-risk loans (0.94, and 0.91 respectively) compared to the BalancedRandomForestClassifier model with a 0.67 and 0.91 sensitivity for high-risk and low-risk loans correspondingly. The low-risk loan’s F-1 scores for both the EasyEnsembleClassifier and BalancedRandomForestClassifier models were good (0.97, and 0.95 respectively) in comparison to all other models, but their high-risk loan’s F-1 scores were bad (0.14 and 0.07 respectively) even though they were improved compared to the other models.</br>
In the light of these results, all the models were weak in determining the high-risk loans. Both the EasyEnsembleClassifier and BalancedRandomForestClassifier models produced good improvements, especially the EasyEnsembleClassifier model. However, they had a high-risk loan low precision score meaning that a lot of low-risk loans would be falsely classified as high-risk loans which would adversely impact the bank’s loan strategy by missing good opportunities to increase revenues and business.
Therefore, I would not recommend any of these models to the bank to predict the loan risk because their low f-1 score in detecting high-risk loans.
