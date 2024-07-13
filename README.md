# Forest_Cover_Classification
Implemented some popular classification algorithms to predict the type of forest cover

## Dataset Description
The dataset contains data from 4 areas of the Roosevelt National Forest in Colorado. It includes information on tree type, shadow coverage, distance to nearby landmarks (roads etc.), soil type, and local topography for 7 different types of forest cover.
* Title: Forest Cover Type Dataset
* URL: https://www.kaggle.com/datasets/uciml/forest-cover-type-dataset
<table>
<thead>
  <tr>
    <th align="center" colspan="2">Generic Description</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td align="center">Number of samples</td>
    <td align="center">5,81,012</td>
  </tr>
  <tr>
    <td align="center">Number of attributes</td>
    <td align="center">55</td>
  </tr>
  <tr>
    <td align="center">Input features</td>
    <td align="center">54</td>
  </tr>
  <tr>
    <td align="center">Target variable</td>
    <td align="center">Cover_Type (1 to 7)</td>
  </tr>
</tbody>
</table>

The dataset has been split into a training subset and test subset, with 80 % being used for training and 20 % being used for testing. Before being used, the data in both subsets have been normalized using StandardScaler.

## K Nearest Neighbours (KNN Classifier)
For the KNN classifier, the accuracy rates were computed for different values of K for the training dataset, and confusion matrices and bar plots were plotted for some of these values to visualize the frequency distribution of actual and predicted labels for the test dataset.
* Best Value of K (= 1)
  <table>
  <thead>
    <tr>
      <th align="center"></th>
      <th align="center">Accuracy (in %)</th>
      <th align="center">Precision (in %)</th>
      <th align="center">Recall (in %)</th>
      <th align="center">F1 Score (in %)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="center"><strong>Training Subset</strong></td>
      <td align="center">100.000</td>
      <td align="center">100.000</td>
      <td align="center">100.000</td>
      <td align="center">100.000</td>
    </tr>
    <tr>
      <td align="center"><strong>Test Subset</strong></td>
      <td align="center">93.554</td>
      <td align="center">93.552</td>
      <td align="center">93.554</td>
      <td align="center">93.553</td>
    </tr>
  </tbody>
  </table>

* Other Values of K
  <table>
  <thead>
    <tr>
      <th align="center">Value of K</th>
      <th align="center"></th>
      <th align="center">Accuracy (in %)</th>
      <th align="center">Precision (in %)</th>
      <th align="center">Recall (in %)</th>
      <th align="center">F1 Score (in %)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="center" rowspan="2"><strong>3</strong></td>
      <td align="center"><strong>Training Subset</strong></td>
      <td align="center">96.924</td>
      <td align="center">96.920</td>
      <td align="center">96.924</td>
      <td align="center">96.921</td>
    </tr>
    <tr>
      <td align="center"><strong>Test Subset</strong></td>
      <td align="center">93.244</td>
      <td align="center">93.224</td>
      <td align="center">93.244</td>
      <td align="center">93.229</td>
    </tr>
    <tr>
      <td align="center" colspan="6"></td>
    </tr>
    <tr>
      <td align="center" rowspan="2"><strong>5</strong></td>
      <td align="center"><strong>Training Subset</strong></td>
      <td align="center">95.624</td>
      <td align="center">95.616</td>
      <td align="center">95.624</td>
      <td align="center">95.616</td>
    </tr>
    <tr>
      <td align="center"><strong>Test Subset</strong></td>
      <td align="center">92.802</td>
      <td align="center">92.776</td>
      <td align="center">92.802</td>
      <td align="center">92.778</td>
    </tr>
    <tr>
      <td align="center" colspan="6"></td>
    </tr>
    <tr>
      <td align="center" rowspan="2"><strong>10</strong></td>
      <td align="center"><strong>Training Subset</strong></td>
      <td align="center">93.585</td>
      <td align="center">93.567</td>
      <td align="center">93.585</td>
      <td align="center">93.557</td>
    </tr>
    <tr>
      <td align="center"><strong>Test Subset</strong></td>
      <td align="center">91.640</td>
      <td align="center">91.596</td>
      <td align="center">91.640</td>
      <td align="center">91.593</td>
    </tr>
    <tr>
      <td align="center" colspan="6"></td>
    </tr>
    <tr>
      <td align="center" rowspan="2"><strong>20</strong></td>
      <td align="center"><strong>Training Subset</strong></td>
      <td align="center">91.298</td>
      <td align="center">91.257</td>
      <td align="center">91.298</td>
      <td align="center">91.245</td>
    </tr>
    <tr>
      <td align="center"><strong>Test Subset</strong></td>
      <td align="center">90.102</td>
      <td align="center">90.046</td>
      <td align="center">90.102</td>
      <td align="center">90.033</td>
    </tr>
  </tbody>
  </table>

## Decision Tree Classifier
A decision tree classifier was implemented using various impurity measures, and confusion matrices and bar plots were plotted for the corresponding results.
  <table>
  <thead>
    <tr>
      <th align="center">Criterion</th>
      <th align="center"></th>
      <th align="center">Accuracy (in %)</th>
      <th align="center">Precision (in %)</th>
      <th align="center">Recall (in %)</th>
      <th align="center">F1 Score (in %)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="center" rowspan="2"><strong>Entropy</strong></td>
      <td align="center"><strong>Training Subset</strong></td>
      <td align="center">100.000</td>
      <td align="center">100.000</td>
      <td align="center">100.000</td>
      <td align="center">100.000</td>
    </tr>
    <tr>
      <td align="center"><strong>Test Subset</strong></td>
      <td align="center">94.381</td>
      <td align="center">94.378</td>
      <td align="center">94.381</td>
      <td align="center">94.379</td>
    </tr>
    <tr>
      <td align="center" colspan="6"></td>
    </tr>
    <tr>
      <td align="center" rowspan="2"><strong>Gini Index</strong></td>
      <td align="center"><strong>Training Subset</strong></td>
      <td align="center">100.000</td>
      <td align="center">100.000</td>
      <td align="center">100.000</td>
      <td align="center">100.000</td>
    </tr>
    <tr>
      <td align="center"><strong>Test Subset</strong></td>
      <td align="center">93.879</td>
      <td align="center">93.876</td>
      <td align="center">93.879</td>
      <td align="center">93.877</td>
    </tr>
  </tbody>
  </table>

## Random Forest Classifier
A random forest classifier with 200 estimators was implemented using various impurity measures, and confusion matrices and bar plots were plotted for the corresponding results.
  <table>
  <thead>
    <tr>
      <th align="center">Criterion</th>
      <th align="center"></th>
      <th align="center">Accuracy (in %)</th>
      <th align="center">Precision (in %)</th>
      <th align="center">Recall (in %)</th>
      <th align="center">F1 Score (in %)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="center" rowspan="2"><strong>Entropy</strong></td>
      <td align="center"><strong>Training Subset</strong></td>
      <td align="center">100.000</td>
      <td align="center">100.000</td>
      <td align="center">100.000</td>
      <td align="center">100.000</td>
    </tr>
    <tr>
      <td align="center"><strong>Test Subset</strong></td>
      <td align="center">95.770</td>
      <td align="center">95.779</td>
      <td align="center">95.770</td>
      <td align="center">95.751</td>
    </tr>
    <tr>
      <td align="center" colspan="6"></td>
    </tr>
    <tr>
      <td align="center" rowspan="2"><strong>Gini Index</strong></td>
      <td align="center"><strong>Training Subset</strong></td>
      <td align="center">100.000</td>
      <td align="center">100.000</td>
      <td align="center">100.000</td>
      <td align="center">100.000</td>
    </tr>
    <tr>
      <td align="center"><strong>Test Subset</strong></td>
      <td align="center">95.594</td>
      <td align="center">95.607</td>
      <td align="center">95.594</td>
      <td align="center">95.571</td>
    </tr>
  </tbody>
  </table>

## Gaussian Naive Bayes Classifier
A Gaussian Naive Bayes classifier was implemented with prior probabilities as the class probabilities of the dataset, and confusion matrices and bar plots were plotted for the corresponding results.
  <table>
  <thead>
    <tr>
      <th align="center"></th>
      <th align="center">Accuracy (in %)</th>
      <th align="center">Precision (in %)</th>
      <th align="center">Recall (in %)</th>
      <th align="center">F1 Score (in %)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="center"><strong>Training Subset</strong></td>
      <td align="center">8.818</td>
      <td align="center">49.597</td>
      <td align="center">8.818</td>
      <td align="center">5.596</td>
    </tr>
    <tr>
      <td align="center"><strong>Test Subset</strong></td>
      <td align="center">8.729</td>
      <td align="center">49.808</td>
      <td align="center">8.729</td>
      <td align="center">5.543</td>
    </tr>
  </tbody>
  </table>

## Linear SVC (Support Vector Classifier)
For the linear SVC model, the optimal hyperparameters was found using grid search with cross validation, and confusion matrices and bar plots were plotted for the corresponding results.
  <table>
  <thead>
    <tr>
      <th align="center"></th>
      <th align="center">Accuracy (in %)</th>
      <th align="center">Precision (in %)</th>
      <th align="center">Recall (in %)</th>
      <th align="center">F1 Score (in %)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="center"><strong>Training Subset</strong></td>
      <td align="center">71.274</td>
      <td align="center">70.398</td>
      <td align="center">71.274</td>
      <td align="center">69.713</td>
    </tr>
    <tr>
      <td align="center"><strong>Test Subset</strong></td>
      <td align="center">71.272</td>
      <td align="center">70.354</td>
      <td align="center">71.272</td>
      <td align="center">69.663</td>
    </tr>
  </tbody>
  </table>

## Logistic Regression
A logistic regression model was implemented using the L2 norm and the 'saga' solver, and confusion matrices and bar plots were plotted for the corresponding results.
  <table>
  <thead>
    <tr>
      <th align="center"></th>
      <th align="center">Accuracy (in %)</th>
      <th align="center">Precision (in %)</th>
      <th align="center">Recall (in %)</th>
      <th align="center">F1 Score (in %)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="center"><strong>Training Subset</strong></td>
      <td align="center">72.514</td>
      <td align="center">71.309</td>
      <td align="center">72.514</td>
      <td align="center">71.496</td>
    </tr>
    <tr>
      <td align="center"><strong>Test Subset</strong></td>
      <td align="center">72.261</td>
      <td align="center">70.979</td>
      <td align="center">72.261</td>
      <td align="center">71.220</td>
    </tr>
  </tbody>
  </table>

## Linear Discriminant Analysis (LDA)
A LDA model was implemented using different solvers, and confusion matrices and bar plots were plotted for the corresponding results.
  <table>
  <thead>
    <tr>
      <th align="center">Solver</th>
      <th align="center"></th>
      <th align="center">Accuracy (in %)</th>
      <th align="center">Precision (in %)</th>
      <th align="center">Recall (in %)</th>
      <th align="center">F1 Score (in %)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="center" rowspan="2"><strong>SVD</strong></td>
      <td align="center"><strong>Training Subset</strong></td>
      <td align="center">67.973</td>
      <td align="center">69.481</td>
      <td align="center">67.973</td>
      <td align="center">68.294</td>
    </tr>
    <tr>
      <td align="center"><strong>Test Subset</strong></td>
      <td align="center">67.828</td>
      <td align="center">69.365</td>
      <td align="center">67.828</td>
      <td align="center">68.160</td>
    </tr>
    <tr>
      <td align="center" colspan="6"></td>
    </tr>
    <tr>
      <td align="center" rowspan="2"><strong>LSQR</strong></td>
      <td align="center"><strong>Training Subset</strong></td>
      <td align="center">67.973</td>
      <td align="center">69.481</td>
      <td align="center">67.973</td>
      <td align="center">68.294</td>
    </tr>
    <tr>
      <td align="center"><strong>Test Subset</strong></td>
      <td align="center">67.828</td>
      <td align="center">69.365</td>
      <td align="center">67.828</td>
      <td align="center">68.160</td>
    </tr>
    <tr>
      <td align="center" colspan="6"></td>
    </tr>
    <tr>
      <td align="center" rowspan="2"><strong>Eigen</strong></td>
      <td align="center"><strong>Training Subset</strong></td>
      <td align="center">67.973</td>
      <td align="center">69.481</td>
      <td align="center">67.973</td>
      <td align="center">68.294</td>
    </tr>
    <tr>
      <td align="center"><strong>Test Subset</strong></td>
      <td align="center">67.828</td>
      <td align="center">69.365</td>
      <td align="center">67.828</td>
      <td align="center">68.160</td>
    </tr>
  </tbody>
  </table>

## Perceptron
A linear perceptron classifier was implemented using different penalty/regularization terms, and confusion matrices and bar plots were plotted for the corresponding results.
  <table>
  <thead>
    <tr>
      <th align="center">Penalty</th>
      <th align="center"></th>
      <th align="center">Accuracy (in %)</th>
      <th align="center">Precision (in %)</th>
      <th align="center">Recall (in %)</th>
      <th align="center">F1 Score (in %)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="center" rowspan="2"><strong>L1</strong></td>
      <td align="center"><strong>Training Subset</strong></td>
      <td align="center">63.530</td>
      <td align="center">63.101</td>
      <td align="center">63.530</td>
      <td align="center">62.958</td>
    </tr>
    <tr>
      <td align="center"><strong>Test Subset</strong></td>
      <td align="center">63.440</td>
      <td align="center">62.884</td>
      <td align="center">63.440</td>
      <td align="center">62.828</td>
    </tr>
    <tr>
      <td align="center" colspan="6"></td>
    </tr>
    <tr>
      <td align="center" rowspan="2"><strong>L2</strong></td>
      <td align="center"><strong>Training Subset</strong></td>
      <td align="center">62.370</td>
      <td align="center">61.299</td>
      <td align="center">62.370</td>
      <td align="center">61.637</td>
    </tr>
    <tr>
      <td align="center"><strong>Test Subset</strong></td>
      <td align="center">62.484</td>
      <td align="center">61.362</td>
      <td align="center">62.484</td>
      <td align="center">61.746</td>
    </tr>
    <tr>
      <td align="center" colspan="6"></td>
    </tr>
    <tr>
      <td align="center" rowspan="2"><strong>Elastic Net</strong></td>
      <td align="center"><strong>Training Subset</strong></td>
      <td align="center">65.987</td>
      <td align="center">66.695</td>
      <td align="center">65.987</td>
      <td align="center">64.444</td>
    </tr>
    <tr>
      <td align="center"><strong>Test Subset</strong></td>
      <td align="center">65.786</td>
      <td align="center">66.406</td>
      <td align="center">65.786</td>
      <td align="center">64.206</td>
    </tr>
  </tbody>
  </table>

## Neural Network
A neural network with 5 hidden layers was implemented using different activation functions, and confusion matrices and bar plots were plotted for the corresponding results. Before using the class labels as the target variable, they have be zero-indexed -> 1 to 7 has been converted to 0 to 6.
  <table>
  <thead>
    <tr>
      <th align="center">Activation Function</th>
      <th align="center"></th>
      <th align="center">Accuracy (in %)</th>
      <th align="center">Precision (in %)</th>
      <th align="center">Recall (in %)</th>
      <th align="center">F1 Score (in %)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="center" rowspan="2"><strong>ReLU</strong></td>
      <td align="center"><strong>Training Subset</strong></td>
      <td align="center">97.535</td>
      <td align="center">96.529</td>
      <td align="center">95.054</td>
      <td align="center">95.780</td>
    </tr>
    <tr>
      <td align="center"><strong>Test Subset</strong></td>
      <td align="center">95.872</td>
      <td align="center">94.200</td>
      <td align="center">92.102</td>
      <td align="center">93.123</td>
    </tr>
    <tr>
      <td align="center" colspan="6"></td>
    </tr>
    <tr>
      <td align="center" rowspan="2"><strong>Tanh</strong></td>
      <td align="center"><strong>Training Subset</strong></td>
      <td align="center">97.742</td>
      <td align="center">95.612</td>
      <td align="center">96.139</td>
      <td align="center">95.869</td>
    </tr>
    <tr>
      <td align="center"><strong>Test Subset</strong></td>
      <td align="center">95.540</td>
      <td align="center">92.041</td>
      <td align="center">92.279</td>
      <td align="center">92.157</td>
    </tr>
    <tr>
      <td align="center" colspan="6"></td>
    </tr>
    <tr>
      <td align="center" rowspan="2"><strong>Sigmoid</strong></td>
      <td align="center"><strong>Training Subset</strong></td>
      <td align="center">97.432</td>
      <td align="center">96.447</td>
      <td align="center">94.942</td>
      <td align="center">95.653</td>
    </tr>
    <tr>
      <td align="center"><strong>Test Subset</strong></td>
      <td align="center">96.012</td>
      <td align="center">94.470</td>
      <td align="center">91.959</td>
      <td align="center">93.090</td>
    </tr>
  </tbody>
  </table>

## AdaBoost Classifier
An Adaboost classifier was implemented using different base estimators, and confusion matrices and bar plots were plotted for the corresponding results.
  <table>
  <thead>
    <tr>
      <th align="center">Base Estimator</th>
      <th align="center"></th>
      <th align="center">Accuracy (in %)</th>
      <th align="center">Precision (in %)</th>
      <th align="center">Recall (in %)</th>
      <th align="center">F1 Score (in %)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="center" rowspan="2"><strong>Decision Tree</strong></td>
      <td align="center"><strong>Training Subset</strong></td>
      <td align="center">100.000</td>
      <td align="center">100.000</td>
      <td align="center">100.000</td>
      <td align="center">100.000</td>
    </tr>
    <tr>
      <td align="center"><strong>Test Subset</strong></td>
      <td align="center">93.880</td>
      <td align="center">93.877</td>
      <td align="center">93.880</td>
      <td align="center">93.878</td>
    </tr>
    <tr>
      <td align="center" colspan="6"></td>
    </tr>
    <tr>
      <td align="center" rowspan="2"><strong>Linear SVC</strong></td>
      <td align="center"><strong>Training Subset</strong></td>
      <td align="center">52.969</td>
      <td align="center">56.406</td>
      <td align="center">52.969</td>
      <td align="center">53.169</td>
    </tr>
    <tr>
      <td align="center"><strong>Test Subset</strong></td>
      <td align="center">53.018</td>
      <td align="center">56.373</td>
      <td align="center">53.018</td>
      <td align="center">53.235</td>
    </tr>
    <tr>
      <td align="center" colspan="6"></td>
    </tr>
    <tr>
      <td align="center" rowspan="2"><strong>Logistic Regression</strong></td>
      <td align="center"><strong>Training Subset</strong></td>
      <td align="center">50.793</td>
      <td align="center">56.051</td>
      <td align="center">50.793</td>
      <td align="center">52.616</td>
    </tr>
    <tr>
      <td align="center"><strong>Test Subset</strong></td>
      <td align="center">50.578</td>
      <td align="center">55.833</td>
      <td align="center">50.578</td>
      <td align="center">52.440</td>
    </tr>
  </tbody>
  </table>

## SGD (Stochastic Gradient Descent) Classifier
A SGD classifier was implemented using different loss functions, and confusion matrices and bar plots were plotted for the corresponding results.
  <table>
  <thead>
    <tr>
      <th align="center">Loss Function</th>
      <th align="center"></th>
      <th align="center">Accuracy (in %)</th>
      <th align="center">Precision (in %)</th>
      <th align="center">Recall (in %)</th>
      <th align="center">F1 Score (in %)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="center" rowspan="2"><strong>Hinge</strong></td>
      <td align="center"><strong>Training Subset</strong></td>
      <td align="center">71.444</td>
      <td align="center">70.384</td>
      <td align="center">71.444</td>
      <td align="center">70.015</td>
    </tr>
    <tr>
      <td align="center"><strong>Test Subset</strong></td>
      <td align="center">71.526</td>
      <td align="center">70.486</td>
      <td align="center">71.526</td>
      <td align="center">70.069/td>
    </tr>
    <tr>
      <td align="center" colspan="6"></td>
    </tr>
    <tr>
      <td align="center" rowspan="2"><strong>Log_Loss</strong></td>
      <td align="center"><strong>Training Subset</strong></td>
      <td align="center">71.499</td>
      <td align="center">70.135</td>
      <td align="center">71.499</td>
      <td align="center">70.304</td>
    </tr>
    <tr>
      <td align="center"><strong>Test Subset</strong></td>
      <td align="center">71.540</td>
      <td align="center">70.096</td>
      <td align="center">71.540</td>
      <td align="center">70.319</td>
    </tr>
    <tr>
      <td align="center" colspan="6"></td>
    </tr>
    <tr>
      <td align="center" rowspan="2"><strong>Modified Huber</strong></td>
      <td align="center"><strong>Training Subset</strong></td>
      <td align="center">70.603</td>
      <td align="center">68.941</td>
      <td align="center">70.603</td>
      <td align="center">69.030</td>
    </tr>
    <tr>
      <td align="center"><strong>Test Subset</strong></td>
      <td align="center">70.558</td>
      <td align="center">68.798</td>
      <td align="center">70.558</td>
      <td align="center">69.941</td>
    </tr>
  </tbody>
  </table>

## Histogram-based Gradient Boosting Classifier
A histogram-based gradient boosting classifier was implemented using L2 regurlarization, and confusion matrices and bar plots were plotted for the corresponding results.
  <table>
  <thead>
    <tr>
      <th align="center"></th>
      <th align="center">Accuracy (in %)</th>
      <th align="center">Precision (in %)</th>
      <th align="center">Recall (in %)</th>
      <th align="center">F1 Score (in %)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="center"><strong>Training Subset</strong></td>
      <td align="center">97.826</td>
      <td align="center">97.828</td>
      <td align="center">97.826</td>
      <td align="center">97.824</td>
    </tr>
    <tr>
      <td align="center"><strong>Training Subset</strong></td>
      <td align="center">95.460</td>
      <td align="center">95.459</td>
      <td align="center">95.460</td>
      <td align="center">95.450</td>
    </tr>
  </tbody>
  </table>

## Bagging Classifier
A bagging classifier with 20 estimators was implemented using a decision tree as the base estimator, and confusion matrices and bar plots were plotted for the corresponding results.
  <table>
  <thead>
    <tr>
      <th align="center"></th>
      <th align="center">Accuracy (in %)</th>
      <th align="center">Precision (in %)</th>
      <th align="center">Recall (in %)</th>
      <th align="center">F1 Score (in %)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="center"><strong>Training Subset</strong></td>
      <td align="center">99.952</td>
      <td align="center">99.952</td>
      <td align="center">99.952</td>
      <td align="center">99.952</td>
    </tr>
    <tr>
      <td align="center"><strong>Training Subset</strong></td>
      <td align="center">96.511</td>
      <td align="center">96.504</td>
      <td align="center">96.511</td>
      <td align="center">96.504</td>
    </tr>
  </tbody>
  </table>

## XGBoost Classifier
For the XGBoost classifier model, the optimal hyperparameters was found using grid search with 5-fold cross validation, and confusion matrices and bar plots were plotted for the corresponding results.
  <table>
  <thead>
    <tr>
      <th align="center"></th>
      <th align="center">Accuracy (in %)</th>
      <th align="center">Precision (in %)</th>
      <th align="center">Recall (in %)</th>
      <th align="center">F1 Score (in %)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="center"><strong>Training Subset</strong></td>
      <td align="center">99.992</td>
      <td align="center">99.992</td>
      <td align="center">99.992</td>
      <td align="center">99.992</td>
    </tr>
    <tr>
      <td align="center"><strong>Training Subset</strong></td>
      <td align="center">96.938</td>
      <td align="center">96.933</td>
      <td align="center">96.938</td>
      <td align="center">96.933</td>
    </tr>
  </tbody>
  </table>
