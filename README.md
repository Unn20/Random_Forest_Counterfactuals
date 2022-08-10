# Random forest counterfactuals

---
The RFC is a library specialised to be use in the domain of Explainable Artificial Intelligence (XAI).
It allows to search for an explanation for pretrained Random Forest Classifier model (from scikit-learn library) in the form of counterfactual examples.
The library also allow to specify constraints (frozen attribute, monotonnicity) for attributes in the specified dataset.
An algorithm is an extended version of Gabriele Tolomei et al. algorithm ([link to github](https://github.com/gtolomei/ml-feature-tweaking)). 

An extended version has:
- multiclass classification problem's explainability,
- gives new methods of selecting counterfactual examples, 
- provides new distance functions (specially HOEM)
- it is optimized to provide explanation as fast as possible with multiprocessing calculation

This work is done for purpose of my master's thesis.


## Installation

This package was tested on Windows 10 and Ubuntu 20.04 platforms.

To install package:

`pip install rf_counterfactuals`


## Manual installation (for developers)

To run unittests:

```
cd tests
python -m pytest
```

To build up a pywheel file:

`python setup.py bdist_whee`

## Simple example on iris dataset

```
from rf_counterfactuals import RandomForestExplainer, visualize

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

### Load iris dataset as Pandas Dataframe (other formats aren't supported yet) and split. Then train RF classifier on it
X, y = load_iris(return_X_y=True, as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.67, random_state=420, stratify=y)
rf = RandomForestClassifier()
rf.fit(X_train, y_train)

### Make an RandomForestExplainer object with input of: RandomForest model, training data
rfe = RandomForestExplainer(rf, X_train, y_train)

### Look for counterfactual examples (max/limit 1) in test data, which lead to change label's value from '0'('setosa') to '2'('virginica')
### Counterfactual examples are selected based on 'hoem' metric's value
X_test_label_0 = X_test[y_test==0]
counterfactuals = rfe.explain_with_single_metric(X_test_label_0, 2, metric='hoem', limit=1)

### Visualize an example row (row_no = 0) of data with its counterfactual
row_index_to_visualize = 0

row = X_test_label_0.iloc[row_index_to_visualize]

# First counterfactual found for row 0th
cf = counterfactuals[row_index_to_visualize].iloc[0]

print(f"row label: {rf.predict(row.to_frame(0).T)[0]} |\t cf label: {rf.predict(cf.to_frame(0).T)[0]}")
print(visualize(rfe, row, cf))
```
output (it may vary because of used random seed to split dataset):
```
row label: 0 |	 cf label: 2
                     X     X'  difference constraints
sepal length (cm)  5.8  6.333       0.533            
sepal width (cm)   4.0  4.000       0.000            
petal length (cm)  1.2  5.076       3.876            
petal width (cm)   0.2  0.200       0.000    
```

### Author
Maciej Leszczyk, Poznan University of Technology 

maciej.leszczyk98(at)gmail.com