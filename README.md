### HeartFailure_Classification

### Keywords: data exploration, feature selection, machine learning, cardiovascular diseases

The study aimed to propose a data processing pipeline that could classify patients into groups of high and low risk of heart disease. To achieve this goal, a range of machine learning algorithms were tested: logistic regression, decision trees, random forests, support vector machine, XGBoost and the Naive Bayes classifier.

The processed dataset exhibited class imbalance in the target feature. This provided an opportunity to investigate the influence of different dataset balancing techniques on the classification results. Model performance was evaluated using several key metrics, including accuracy, specificity, sensitivity, and the area under the ROC curve (AUC). Notably, the most favorable results were obtained when applying the SMOTE oversampling technique and a combination of oversampling and undersampling known as SMOTETomek.

The experimental phase of the study focused on the assessment of imputation techniques impact on models' efficiency and quality. Different approaches, such as mean, median and iterative imputation methods, as well as the kNN algorithm were tested for handling missing data points. However, no impact of the selected methods  on the results was observed.
