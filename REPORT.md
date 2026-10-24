# Wine Quality Classification

**Name:** Ryan Truong
**Course:** CS 271  
**Date:** 10/25/26

## 1. Objective
A classifier model experiment for determining wine quality. The evaluated models are logistic regression, decision tree, support vector machine, and gradient boost.

## 3. Data Exploration
### Class Distribution

High: 50

Low: 163

Medium: 2,016

Medium-High: 620

Medium-Low: 2,151

## 4. Preprocessing

### Training and Test Split

I used a 80-10-10 test, validation, and test split. Given that there are only 5000 samples, I wanted as many training samples as possible. 

### Invalid Value Handling

The first step taken in preprocessing is checking for invalid values. Each value is verified to be a float. Afterwards, the quality column was checked to see if it matches one of the five valid labels (Low, Medium-Low, Medium, Medium-High, and High). Any rows that have irregularities would be removed. In this dataset, no invalid rows were found out of the 5000 samples.

### Feature Scaling
For the logistic regression and SVM models, I used a standard scaler for regularization. Decision trees and gradient boosting did not need standardization because they dont make distance-based comparisons.

### Target Encoding
The Scikit-learn classifier models automatically handles encoding the strings for the target labels. It internally maps each string to the numeric class indices each one represents. If I wasn't using Scikit, then each class (high, medium, etc.) would have to be represented by a certain index in a size 5 vector. 
 
## 5. Model Training
First, a logistic regression model was tested to establish a baseline. The results are 

Training accuracy: 0.4370
Training error: 0.5630
Validation accuracy: 0.4380
Validation error: 0.5620
Test accuracy: 0.4480
Test error: 0.5520

Classification report:
              precision    recall  f1-score   support

        High       0.00      0.00      0.00         5
         Low       0.00      0.00      0.00        17
      Medium       0.45      0.29      0.36       201
 Medium-High       0.00      0.00      0.00        62
  Medium-Low       0.45      0.77      0.56       215

    accuracy                           0.45       500
   macro avg       0.18      0.21      0.18       500
weighted avg       0.37      0.45      0.39       500

The results are decent with 44.8% accuracy. The dataset is small, so a logistic regression model may work the best due to its simplicity. The logistic regression model does not show any signs of overfitting. 

Second a decision tree model was used to introduce more complexity. The results are 

Training accuracy: 1.0000
Training error: 0.0000
Validation accuracy: 0.3540
Validation error: 0.6460
Test accuracy: 0.3860
Test error: 0.6140

Classification report:
              precision    recall  f1-score   support

        High       0.00      0.00      0.00         5
         Low       0.08      0.06      0.07        17
      Medium       0.40      0.40      0.40       201
 Medium-High       0.21      0.24      0.23        62
  Medium-Low       0.45      0.45      0.45       215

    accuracy                           0.39       500
   macro avg       0.23      0.23      0.23       500
weighted avg       0.38      0.39      0.39       500

The results here indicate an overfitting problem, as there is 0 training error. The decision tree model was able to memorize the data. To combat this, I changed the max_depth hyperparameter to 7. After experementing with many different values from range 1-15, I found 7 to give the consistently best results. I also tuned min_samples_split and min_samples_leaf, however these appeard to have minimal effect on the results, so they are left at default.

The results of the tuned decision tree are

Training accuracy: 0.4948
Training error: 0.5052
Validation accuracy: 0.4100
Validation error: 0.5900
Test accuracy: 0.3960
Test error: 0.6040

Classification report:
              precision    recall  f1-score   support

        High       0.00      0.00      0.00         5
         Low       0.00      0.00      0.00        17
      Medium       0.31      0.13      0.18       201
 Medium-High       0.33      0.03      0.06        62
  Medium-Low       0.42      0.79      0.55       215

    accuracy                           0.40       500
   macro avg       0.21      0.19      0.16       500
weighted avg       0.34      0.40      0.32       500

The changed max_depth appears to have limited the overfitting, overall improving model generalization. However, it is still outperformed by the logistic regression model.

Next a support vector machine model was tried. The results are 

Training accuracy: 0.5600
Training error: 0.4400
Validation accuracy: 0.4340
Validation error: 0.5660
Test accuracy: 0.4420
Test error: 0.5580

Classification report:
              precision    recall  f1-score   support

        High       0.00      0.00      0.00         5
         Low       0.00      0.00      0.00        17
      Medium       0.43      0.35      0.39       201
 Medium-High       0.00      0.00      0.00        62
  Medium-Low       0.45      0.70      0.55       215

    accuracy                           0.44       500
   macro avg       0.18      0.21      0.19       500
weighted avg       0.37      0.44      0.39       500

The SVM model also performs around the average.

Then svm tuned for overfitting by changing C=0.1 and gamma=0.001

Training accuracy: 0.4303
Training error: 0.5697
Validation accuracy: 0.4300
Validation error: 0.5700
Test accuracy: 0.4300
Test error: 0.5700

Classification report:
              precision    recall  f1-score   support

        High       0.00      0.00      0.00         5
         Low       0.00      0.00      0.00        17
      Medium       0.00      0.00      0.00       201
 Medium-High       0.00      0.00      0.00        62
  Medium-Low       0.43      1.00      0.60       215

    accuracy                           0.43       500
   macro avg       0.09      0.20      0.12       500
weighted avg       0.18      0.43      0.26       500

Although overfitting is reduced since the training accuracy and test accuracy are much closer, the test accuracy is actually lower than the original version.

Next, a gradient boost model

Training accuracy: 0.6907
Training error: 0.3093
Validation accuracy: 0.4080
Validation error: 0.5920
Test accuracy: 0.3920
Test error: 0.6080

Classification report:
              precision    recall  f1-score   support

        High       0.00      0.00      0.00         5
         Low       0.00      0.00      0.00        17
      Medium       0.38      0.32      0.35       201
 Medium-High       0.11      0.02      0.03        62
  Medium-Low       0.41      0.61      0.49       215

    accuracy                           0.39       500
   macro avg       0.18      0.19      0.17       500
weighted avg       0.34      0.39      0.35       500

After some tuning. Introduced patience up to 10 iterations. Reduced max_depth to 2 and min_samples_leaf increased to 15. These are all done to reduce overfitting.

Training accuracy: 0.4745
Training error: 0.5255
Validation accuracy: 0.4220
Validation error: 0.5780
Test accuracy: 0.4280
Test error: 0.5720

Classification report:
              precision    recall  f1-score   support

        High       0.00      0.00      0.00         5
         Low       0.00      0.00      0.00        17
      Medium       0.40      0.16      0.23       201
 Medium-High       0.00      0.00      0.00        62
  Medium-Low       0.43      0.85      0.57       215

    accuracy                           0.43       500
   macro avg       0.17      0.20      0.16       500
weighted avg       0.35      0.43      0.34       500

The adjusted GradientBoost model performs better after implementing the changes to reduce overfitting. The best change was introducing patience, which prevented the training accuracy from going too high, while not improving the validation accuracy. 

## 6. Evaluation
### Test Accuracy Summary

| Model | Test Accuracy |
|---|---:|
| Logistic Regression | 44.8% |
| Tuned Decision Tree | 39.6% |
| Default Decision Tree | 38.6% |
| Default SVM | 44.2% |
| Tuned SVM | 43.0% |
| Tuned Gradient Boosting | 42.8% |
| Default Gradient Boosting | 39.2% |

Four model types were trained on the wine_quality_dataset: logistic regression, decision tree, support vector machine, and gradient boost. Logistic regression achieved the highest test accuracy, followed by the default SVM. For decision tree, svm, and gradient boosting, the default hyperparmeters showed signs of overfitting. 

#### Reducing overfitting for decision trees
To reduce overfitting for decision trees, the max_depth, min_samples_split and min_samples_leaf were changed from the default values. After experimenting with combinations, it seemed like max_depth had the most noticable effect on reducing overfitting. The max_depth chosen was 7. Since min_samples_split and min_samples_leaf had little effect on the results, they were left at default values. Overall, the tuned model only performed marginally better, even though it had siginifcantly less overfitting.

#### Reducing overfitting for SVM
To reduce overfitting for SVM the C (regularization strength), gamma, and kernel type were adjusted. A smaller C value, meaning stronger regularization, was found to have the best results. The optimal kernel type seemed to be rbf with a gamma of 0.001 to reduce overfitting and produce smoother boundaries. The rbf kernel type made the model non-linear, which may be a reason for the degradation in performance. 

#### Reduce overfitting for gradient boost
To reduce overfitting for the gradient boost model the max_depth and min_samples_leaf were adjusted, along with introducing patience. For max_depth, a lower value of 2 was found to have the best effects in reducing overfitting. Min_samples_leaf was chosen to be 15, however this hyperparameter did not seem to have a large effect on the results. Patience was implemented by reserving 10% of the training data for validation to check for early stoppage. The model would tolerate 10 iterations of no change before initiating an early stoppage. This seemed to have a reasonable increase in performance in the model.

## 7. Discussion and Conclusions
The two best performing models are logistic regression and linear SVM. It seems like the simpler models had an easier time with this dataset because there are only 11 features. The relation between the features may be more linear in this dataset, which gives an advantage to the simpler logistic regression and SVM models. The decision tree and gradient boost models seemed to overfit to the training data. However, even after tuning to reduce overfitting, the models still performed worse than the linear models. 

