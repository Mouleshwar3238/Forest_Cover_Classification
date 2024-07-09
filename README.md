# Forest_Cover_Classification
Implemented some popular classification algorithms to predict the type of forest cover

## Dataset Description
The dataset contains data from 4 areas of the Roosevelt National Forest in Colorado. It includes information on tree type, shadow coverage, distance to nearby landmarks (roads etcetera), soil type, and local topography for 7 different types of forest cover.
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
    <td align="center">Cover_Type (0 to 6)</td>
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
      <td align="center">93.239</td>
      <td align="center">93.239</td>
      <td align="center">93.239</td>
      <td align="center">93.239</td>
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
      <td align="center">96.748</td>
      <td align="center">96.744</td>
      <td align="center">96.748</td>
      <td align="center">96.745</td>
    </tr>
    <tr>
      <td align="center"><strong>Test Subset</strong></td>
      <td align="center">92.914</td>
      <td align="center">92.893</td>
      <td align="center">92.914</td>
      <td align="center">92.899</td>
    </tr>
    <tr>
      <td align="center" colspan="6"></td>
    </tr>
    <tr>
      <td align="center" rowspan="2"><strong>5</strong></td>
      <td align="center"><strong>Training Subset</strong></td>
      <td align="center">95.393</td>
      <td align="center">95.384</td>
      <td align="center">95.393</td>
      <td align="center">95.384</td>
    </tr>
    <tr>
      <td align="center"><strong>Test Subset</strong></td>
      <td align="center">92.446</td>
      <td align="center">92.419</td>
      <td align="center">92.446</td>
      <td align="center">92.423</td>
    </tr>
    <tr>
      <td align="center" colspan="6"></td>
    </tr>
    <tr>
      <td align="center" rowspan="2"><strong>10</strong></td>
      <td align="center"><strong>Training Subset</strong></td>
      <td align="center">93.271</td>
      <td align="center">93.253</td>
      <td align="center">93.271</td>
      <td align="center">93.241</td>
    </tr>
    <tr>
      <td align="center"><strong>Test Subset</strong></td>
      <td align="center">91.169</td>
      <td align="center">91.128</td>
      <td align="center">91.169</td>
      <td align="center">91.123</td>
    </tr>
    <tr>
      <td align="center" colspan="6"></td>
    </tr>
    <tr>
      <td align="center" rowspan="2"><strong>20</strong></td>
      <td align="center"><strong>Training Subset</strong></td>
      <td align="center">90.861</td>
      <td align="center">90.815</td>
      <td align="center">90.861</td>
      <td align="center">90.798</td>
    </tr>
    <tr>
      <td align="center"><strong>Test Subset</strong></td>
      <td align="center">89.552</td>
      <td align="center">89.490</td>
      <td align="center">89.552</td>
      <td align="center">89.476</td>
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
      <td align="center">8.7676</td>
      <td align="center">5.580</td>
    </tr>
  </tbody>
  </table>

