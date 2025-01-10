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

For all the models except the K Nearest Neighbour Classifier and the neural network, the optimal hyperparameters were found using halving grid search with 5-fold cross validation.

Finally, confusion matrices and bar graphs were plotted to visualize the results for each model.

## K Nearest Neighbours (KNN Classifier)
For the KNN classifier, the accuracy rates were computed for different values of K for the training as well as test datasets.
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
      <td align="center">72.448</td>
      <td align="center">71.295</td>
      <td align="center">72.448</td>
      <td align="center">71.484</td>
    </tr>
    <tr>
      <td align="center"><strong>Test Subset</strong></td>
      <td align="center">72.507</td>
      <td align="center">71.152</td>
      <td align="center">72.507</td>
      <td align="center">71.552</td>
    </tr>
  </tbody>
  </table>

## SGD (Stochastic Gradient Descent) Classifier
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
      <td align="center">71.412</td>
      <td align="center">69.955</td>
      <td align="center">71.412</td>
      <td align="center">69.913</td>
    </tr>
    <tr>
      <td align="center"><strong>Test Subset</strong></td>
      <td align="center">71.364</td>
      <td align="center">70.015</td>
      <td align="center">71.364</td>
      <td align="center">69.879</td>
    </tr>
  </tbody>
  </table>

## Neural Network
A neural network with 5 hidden layers was implemented using different activation functions. Before using the class labels as the target variable, they have been zero-indexed -> 1 to 7 has been converted to 0 to 6.
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
      <td align="center">95.010</td>
      <td align="center">92.683</td>
      <td align="center">91.618</td>
      <td align="center">92.078</td>
    </tr>
    <tr>
      <td align="center"><strong>Test Subset</strong></td>
      <td align="center">93.997</td>
      <td align="center">90.922</td>
      <td align="center">89.501</td>
      <td align="center">90.138</td>
    </tr>
    <tr>
      <td align="center" colspan="6"></td>
    </tr>
    <tr>
      <td align="center" rowspan="2"><strong>Tanh</strong></td>
      <td align="center"><strong>Training Subset</strong></td>
      <td align="center">96.864</td>
      <td align="center">95.259</td>
      <td align="center">93.779</td>
      <td align="center">91.204</td>
    </tr>
    <tr>
      <td align="center"><strong>Test Subset</strong></td>
      <td align="center">95.014</td>
      <td align="center">91.932</td>
      <td align="center">90.576</td>
      <td align="center">91.204</td>
    </tr>
    <tr>
      <td align="center" colspan="6"></td>
    </tr>
    <tr>
      <td align="center" rowspan="2"><strong>Sigmoid</strong></td>
      <td align="center"><strong>Training Subset</strong></td>
      <td align="center">94.290</td>
      <td align="center">92.100</td>
      <td align="center">90.649</td>
      <td align="center">91.356</td>
    </tr>
    <tr>
      <td align="center"><strong>Test Subset</strong></td>
      <td align="center">93.471</td>
      <td align="center">90.658</td>
      <td align="center">89.175</td>
      <td align="center">89.896</td>
    </tr>
  </tbody>
  </table>

## AdaBoost Classifier
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
      <td align="center">94.308</td>
      <td align="center">94.305</td>
      <td align="center">94.308</td>
      <td align="center">94.306</td>
    </tr>
  </tbody>
  </table>

## Histogram-based Gradient Boosting Classifier
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
      <td align="center">95.398</td>
      <td align="center">95.403</td>
      <td align="center">95.398</td>
      <td align="center">95.391</td>
    </tr>
    <tr>
      <td align="center"><strong>Test Subset</strong></td>
      <td align="center">93.390</td>
      <td align="center">93.392</td>
      <td align="center">93.390</td>
      <td align="center">93.378</td>
    </tr>
  </tbody>
  </table>

## Bagging Classifier
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
      <td align="center">99.995</td>
      <td align="center">99.995</td>
      <td align="center">99.995</td>
      <td align="center">99.995</td>
    </tr>
    <tr>
      <td align="center"><strong>Test Subset</strong></td>
      <td align="center">96.938</td>
      <td align="center">96.933</td>
      <td align="center">96.938</td>
      <td align="center">96.932</td>
    </tr>
  </tbody>
  </table>
