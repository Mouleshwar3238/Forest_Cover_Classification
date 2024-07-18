# Forest_Cover_Classification
Implemented some popular classification algorithms to predict the type of forest cover. (Extensive improvements were carried out later)

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

The dataset was split into a training subset and test subset, with 80 % being used for training and 20 % being used for testing. Before being used, the data in both subsets was normalized using StandardScaler.

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
For the decision tree model, the optimal hyperparameters were found using halving grid search with 5-fold cross validation, and confusion matrices and bar plots were plotted for the corresponding results.
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
      <td align="center">94.381</td>
      <td align="center">94.378</td>
      <td align="center">94.381</td>
      <td align="center">94.379</td>
    </tr>
  </tbody>
  </table>

## Random Forest Classifier
For the random forest model, the optimal hyperparameters were found using halving grid search with 5-fold cross validation, and confusion matrices and bar plots were plotted for the corresponding results.
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
      <td align="center">96.295</td>
      <td align="center">96.296</td>
      <td align="center">96.295</td>
      <td align="center">96.283</td>
    </tr>
  </tbody>
  </table>

## Gaussian Naive Bayes Classifier
For the Gaussian Naive Bayes model, the optimal hyperparameters were found using halving grid search with 5-fold cross validation, and confusion matrices and bar plots were plotted for the corresponding results.
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
      <td align="center">46.094</td>
      <td align="center">65.928</td>
      <td align="center">46.094</td>
      <td align="center">39.067</td>
    </tr>
    <tr>
      <td align="center"><strong>Test Subset</strong></td>
      <td align="center">46.010</td>
      <td align="center">65.791</td>
      <td align="center">46.010</td>
      <td align="center">35.943</td>
    </tr>
  </tbody>
  </table>

## Linear SVC (Support Vector Classifier)
For the linear SVC model, the optimal hyperparameters were found using halving grid search with 5-fold cross validation, and confusion matrices and bar plots were plotted for the corresponding results.
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
For the logistic regression model, the optimal hyperparameters were found using halving grid search with 5-fold cross validation, and confusion matrices and bar plots were plotted for the corresponding results.
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
      <td align="center">72.173</td>
      <td align="center">70.839</td>
      <td align="center">72.173</td>
      <td align="center">71.070</td>
    </tr>
    <tr>
      <td align="center"><strong>Test Subset</strong></td>
      <td align="center">72.169</td>
      <td align="center">70.796</td>
      <td align="center">72.169</td>
      <td align="center">71.017</td>
    </tr>
  </tbody>
  </table>

## Linear Discriminant Analysis (LDA)
For the LDA model, the optimal hyperparameters were found using halving grid search with 5-fold cross validation, and confusion matrices and bar plots were plotted for the corresponding results.
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
For the perceptron model, the optimal hyperparameters were found using halving grid search with 5-fold cross validation, and confusion matrices and bar plots were plotted for the corresponding results.
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
      <td align="center">60.319</td>
      <td align="center">60.491</td>
      <td align="center">60.319</td>
      <td align="center">60.191</td>
    </tr>
    <tr>
      <td align="center"><strong>Test Subset</strong></td>
      <td align="center">60.214</td>
      <td align="center">60.366</td>
      <td align="center">60.214</td>
      <td align="center">60.058</td>
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
For the AdaBoost classifier model, the optimal hyperparameters were found using halving grid search with 5-fold cross validation, and confusion matrices and bar plots were plotted for the corresponding results.
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
      <td align="center">97.369</td>
      <td align="center">97.365</td>
      <td align="center">97.369</td>
      <td align="center">97.366</td>
    </tr>
  </tbody>
  </table>

## SGD (Stochastic Gradient Descent) Classifier
For the SGD classifier, the optimal hyperparameters were found using halving grid search with 5-fold cross validation, and confusion matrices and bar plots were plotted for the corresponding results.
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
      <td align="center">71.428</td>
      <td align="center">70.039</td>
      <td align="center">71.428</td>
      <td align="center">69.948</td>
    </tr>
    <tr>
      <td align="center"><strong>Test Subset</strong></td>
      <td align="center">71.472</td>
      <td align="center">70.018</td>
      <td align="center">71.472</td>
      <td align="center">69.951</td>
    </tr>
  </tbody>
  </table>

## Histogram-based Gradient Boosting Classifier
For the gradient boosting classifier, the optimal hyperparameters were found using halving grid search with 5-fold cross validation, and confusion matrices and bar plots were plotted for the corresponding results.
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
      <td align="center">97.918</td>
      <td align="center">97.921</td>
      <td align="center">97.918</td>
      <td align="center">97.917</td>
    </tr>
    <tr>
      <td align="center"><strong>Test Subset</strong></td>
      <td align="center">95.608</td>
      <td align="center">95.608</td>
      <td align="center">95.608</td>
      <td align="center">95.599</td>
    </tr>
  </tbody>
  </table>

## Bagging Classifier
For the bagging classifier, the optimal hyperparameters were found using halving grid search with 5-fold cross validation, and confusion matrices and bar plots were plotted for the corresponding results.
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
      <td align="center">96.917</td>
      <td align="center">96.912</td>
      <td align="center">96.917</td>
      <td align="center">96.909</td>
    </tr>
  </tbody>
  </table>

## XGBoost Classifier
For the XGBoost classifier, the optimal hyperparameters were found using halving grid search with 5-fold cross validation, and confusion matrices and bar plots were plotted for the corresponding results.
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
      <td align="center">97.283</td>
      <td align="center">97.279</td>
      <td align="center">97.283</td>
      <td align="center">97.279</td>
    </tr>
  </tbody>
  </table>
