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

The dataset was split into a training set and a test set, with 75 % being used for training and 25 % being used for testing. Before being used, the data in both subsets was normalized using StandardScaler. Additionally, for the neural network model, the training set was further split into training and validation subsets, with 75 % being used for training and 25 % being used for validation.

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
      <td align="center">93.446</td>
      <td align="center">93.469</td>
      <td align="center">93.466</td>
      <td align="center">93.467</td>
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
      <td align="center">96.811</td>
      <td align="center">96.807</td>
      <td align="center">96.811</td>
      <td align="center">96.807</td>
    </tr>
    <tr>
      <td align="center"><strong>Test Subset</strong></td>
      <td align="center">93.151</td>
      <td align="center">93.134</td>
      <td align="center">93.151</td>
      <td align="center">93.139</td>
    </tr>
    <tr>
      <td align="center" colspan="6"></td>
    </tr>
    <tr>
      <td align="center" rowspan="2"><strong>5</strong></td>
      <td align="center"><strong>Training Subset</strong></td>
      <td align="center">95.475</td>
      <td align="center">95.466</td>
      <td align="center">95.475</td>
      <td align="center">95.467</td>
    </tr>
    <tr>
      <td align="center"><strong>Test Subset</strong></td>
      <td align="center">92.723</td>
      <td align="center">92.702</td>
      <td align="center">92.723</td>
      <td align="center">92.705</td>
    </tr>
    <tr>
      <td align="center" colspan="6"></td>
    </tr>
    <tr>
      <td align="center" rowspan="2"><strong>10</strong></td>
      <td align="center"><strong>Training Subset</strong></td>
      <td align="center">93.418</td>
      <td align="center">93.398</td>
      <td align="center">93.418</td>
      <td align="center">93.387</td>
    </tr>
    <tr>
      <td align="center"><strong>Test Subset</strong></td>
      <td align="center">91.605</td>
      <td align="center">91.569</td>
      <td align="center">91.605</td>
      <td align="center">91.566</td>
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
      <td align="center">94.303</td>
      <td align="center">94.302</td>
      <td align="center">94.303</td>
      <td align="center">94.302</td>
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
      <td align="center">95.691</td>
      <td align="center">95.701</td>
      <td align="center">95.691</td>
      <td align="center">95.672</td>
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
      <td align="center">46.011</td>
      <td align="center">66.014</td>
      <td align="center">46.011</td>
      <td align="center">38.916</td>
    </tr>
    <tr>
      <td align="center"><strong>Test Subset</strong></td>
      <td align="center">45.765</td>
      <td align="center">65.648</td>
      <td align="center">45.765</td>
      <td align="center">38.583</td>
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
      <td align="center">71.269</td>
      <td align="center">70.407</td>
      <td align="center">71.269</td>
      <td align="center">69.673</td>
    </tr>
    <tr>
      <td align="center"><strong>Test Subset</strong></td>
      <td align="center">71.276</td>
      <td align="center">70.299</td>
      <td align="center">71.276</td>
      <td align="center">69.688</td>
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
