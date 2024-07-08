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
    <td align="center">Cover_Type (1 to 7)</td>
  </tr>
</tbody>
</table>

The dataset has been split into a training subset and test subset, with 80 % being used for training and 20 % being used for testing. Before being used, the data in both subsets have been normalized using StandardScaler.

## K Nearest Neighbours (KNN Classifier)
