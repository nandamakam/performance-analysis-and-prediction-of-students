# performance-analysis-and-prediction-of-students
ml

* OBJECTIVE:
  My objective was to build a model that would predict whether or not a student would fail the math course that was being tracked. I focused on failure rates as I believed that metric to be more valuable in terms of flagging struggling students who may need more help.
To be able to preemptively assess which students may need the most attention is, in my opinion, an important step to personalized education.

* PROCESS
The target value is 'G3', which, according to the accompanying paper of the dataset, can be binned into a passing or failing classification. If 'G3' is greater than or equal to 10, then the student passes. Otherwise, she fails. Likewise, the 'G1' and 'G2' features are binned in the same manner.

The data can be reduced to 4 fundamental features, in order of importance:
1. 'G2' score
2. 'G1' score
3. 'School'
4. 'Absences'

When no grade knowledge is known, 'School' and 'Absences' capture most of the predictive basis. As grade knowledge becomes available, 'G1' and 'G2' scores alone are enough to achieve over 90% accuracy. I experimentally discovered that the model performs best when it uses only 2 features at a time for each experiment.

The model is a linear support vector machine with a regularization factor of 100. This model performed the best when compared to other models, such as naive bayes, logistic regression, and random forest classifiers.


* RESULT
The following results have been averaged over 5 trials.

| Features Considered 	| G1 & G2 	| G1 & School 	| School & Absences 	|
|---------------------	|:-------:	|:-----------:	|:-----------------:	|
| Paper Accuracy      	|   0.919 	|       0.838 	|             0.706 	|
| My Model Accuracy   	|  0.9165 	|      0.8285 	|            0.6847 	|
| False Pass Rate     	|   0.096 	|        0.12 	|             0.544 	|
| False Fail Rate     	|   0.074 	|      0.1481 	|            0.2185 	|
