# **Wine Quality Classification**

**Name:** Ryan Truong
**Course:** CS 271  
**Date:** 10/25/26

## **Objective**
A classifier model experiment for determining wine quality. The evaluated models are logistic regression, decision tree, support vector machine, and gradient boost. This report focuses entirely on accuracy as the performance metric. F1 score, precision, and recall may offer more insights in future work.

## **Data**
### **Class Distribution**
The quality class distribution in the wine quality dataset is as follows:

High: 50

Low: 163

Medium: 2,016

Medium-High: 620

Medium-Low: 2,151

## **Preprocessing**

### **Training and Test Split**

I used a 80-10-10 test, validation, and test split. Given that there are only 5000 samples, this split maximizes the amount of training samples. 

### **Invalid Value Handling**

The first step taken in preprocessing is checking for invalid values. Each value is verified to be a float. Afterwards, the quality column was checked to see if it matches one of the five valid labels (Low, Medium-Low, Medium, Medium-High, and High). Any rows that have irregularities would be removed. In this dataset, no invalid rows were found out of the 5000 samples.

### **Feature Scaling**
For the logistic regression and SVM models, a standard scaler is used for regularization. Decision trees and gradient boosting did not need standardization because they dont make distance-based comparisons.

### **Target Encoding**
The Scikit-learn classifier models automatically handles encoding the strings for the target labels. It internally maps each string to the numeric class indices each one represents. If an alternative method was used, then each class (high, medium, etc.) would have to be represented by a certain index in a size 5 vector. 
 
## **Model Training**
#### Logistic Regression

| Split | Accuracy |
|---|---:|
| Training | 43.70% |
| Validation | 43.80% |
| Test | 44.80% |

The results are reasonable with 44.8% accuracy. The dataset is small, so a logistic regression model may work the best due to its simplicity. The logistic regression model does not show any signs of overfitting. 

#### Decision Tree

| Split | Accuracy |
|---|---:|
| Training | 100.00% |
| Validation | 35.40% |
| Test | 38.60% |

The results here indicate an overfitting problem, as the training accuracy is 100%. The decision tree model was able to memorize the data. To combat this, the max_depth hyperparameter is set to 7. After experimenting with many different values from range 1-15, value 7 was found to give the consistently best results. min_samples_split and min_samples_leaf were also tuned, however these appeared to have minimal effect on the results, so they are left at default.

#### Tuned Decision Tree

| Split | Accuracy |
|---|---:|
| Training | 49.48% |
| Validation | 41.00% |
| Test | 39.60% |

The changed max_depth appears to have limited the overfitting, overall improving model generalization. However, it is still outperformed by the logistic regression model.

#### Support Vector Machine

| Split | Accuracy |
|---|---:|
| Training | 56.00% |
| Validation | 43.40% |
| Test | 44.20% |

The SVM model performs reasonably well, almost matching the logistic regression model. However, the default SVM shows signs of overfitting. To combat this, the hyperparameters C=0.1 and gamma=0.001 are set. C represents the strength of regularization, with smaller values resulting in stronger regularization. Lower gammas result in smoother models, which can help overfitting.

#### Tuned Support Vector Machine

| Split | Accuracy |
|---|---:|
| Training | 43.03% |
| Validation | 43.00% |
| Test | 43.00% |

Although overfitting is reduced since the training accuracy and test accuracy are much closer, the test accuracy is actually lower than the original version.

#### Gradient Boost
| Split | Accuracy |
|---|---:|
| Training | 69.07% |
| Validation | 40.80% |
| Test | 39.20% |

After some tuning patience up to 10 iterations is introduced. The max_depth is reduced to 2 and min_samples_leaf increased to 15. These are all done to reduce overfitting by limiting the size of the tree, so the model can not simply memorize the data set.

#### Tuned Gradient Boost (Patience introduced)
| Split | Accuracy |
|---|---:|
| Training | 47.45% |
| Validation | 42.20% |
| Test | 42.80% |

The adjusted GradientBoost model performs better after implementing the changes to reduce overfitting. The best change was introducing patience, which prevented the training accuracy from going too high, while not improving the validation accuracy. 

## **Evaluation**
### **Test Accuracy Summary**

| Model | Test Accuracy |
|---|---:|
| Logistic Regression | 44.80% |
| Tuned Decision Tree | 39.60% |
| Default Decision Tree | 38.60% |
| Default SVM | 44.20% |
| Tuned SVM | 43.00% |
| Tuned Gradient Boosting | 42.80% |
| Default Gradient Boosting | 39.20% |

Four model types were trained on the wine_quality_dataset: logistic regression, decision tree, support vector machine, and gradient boost. Logistic regression achieved the highest test accuracy, followed by the default SVM. For decision tree, svm, and gradient boosting, the default hyperparmeters showed signs of overfitting. 

#### **Reducing overfitting for decision trees**
To reduce overfitting for decision trees, the max_depth, min_samples_split and min_samples_leaf were changed from the default values. After experimenting with combinations, it seemed like max_depth had the most noticable effect on reducing overfitting. The max_depth chosen was 7. Since min_samples_split and min_samples_leaf had little effect on the results, they were left at default values. Overall, the tuned model only performed marginally better, even though it had siginifcantly less overfitting.

#### **Reducing overfitting for SVM**
To reduce overfitting for SVM the C (regularization strength), gamma, and kernel type were adjusted. A smaller C value, meaning stronger regularization, was found to have the best results. The optimal kernel type seemed to be rbf with a gamma of 0.001 to reduce overfitting and produce smoother boundaries. The rbf kernel type made the model non-linear, which may be a reason for the degradation in performance. 

#### **Reduce overfitting for gradient boost**
To reduce overfitting for the gradient boost model the max_depth and min_samples_leaf were adjusted, along with introducing patience. For max_depth, a lower value of 2 was found to have the best effects in reducing overfitting. Min_samples_leaf was chosen to be 15, however this hyperparameter did not seem to have a large effect on the results. Patience was implemented by reserving 10% of the training data for validation to check for early stoppage. The model would tolerate 10 iterations of no change before initiating an early stoppage. This seemed to have a reasonable increase in performance in the model.

## **Discussion and Conclusions**
The two best performing models are logistic regression and linear SVM. It seems like the simpler models had an easier time with this dataset because there are only 11 features. The relation between the features may be more linear in this dataset, which gives an advantage to the simpler logistic regression and SVM models. The decision tree and gradient boost models seemed to overfit to the training data. However, even after tuning to reduce overfitting, the models still performed worse than the linear models. 

