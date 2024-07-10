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
      <td align="center">100.00</td>
      <td align="center">100.000</td>
      <td align="center">100.00</td>
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
      <td align="center">100.00</td>
      <td align="center">100.000</td>
      <td align="center">100.00</td>
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
      <td align="center">100.00</td>
      <td align="center">100.000</td>
      <td align="center">100.00</td>
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
      <td align="center">100.00</td>
      <td align="center">100.000</td>
      <td align="center">100.00</td>
    </tr>
    <tr>
      <td align="center"><strong>Test Subset</strong></td>
      <td align="center">95.594</td>
      <td align="center">95.609</td>
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
      <td align="center">8.762</td>
      <td align="center">49.628</td>
      <td align="center">8.762</td>
      <td align="center">5.633</td>
    </tr>
    <tr>
      <td align="center"><strong>Test Subset</strong></td>
      <td align="center">8.676</td>
      <td align="center">49.848</td>
      <td align="center">8.676</td>
      <td align="center">5.580</td>
    </tr>
  </tbody>
  </table>

## Linear SVC (Support Vector Classifier)
A linear SVC was implemented using the L2 norm, and confusion matrices and bar plots were plotted for the corresponding results.
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
      <td align="center">71.272</td>
      <td align="center">70.391</td>
      <td align="center">71.272</td>
      <td align="center">69.710</td>
    </tr>
    <tr>
      <td align="center"><strong>Test Subset</strong></td>
      <td align="center">71.271</td>
      <td align="center">70.353</td>
      <td align="center">71.271</td>
      <td align="center">69.662</td>
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
