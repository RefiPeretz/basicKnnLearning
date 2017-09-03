import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

cancer = load_breast_cancer()
# Description of dataset..
print(cancer.DESCR)

# Convert to data frame
cancerDF = pd.DataFrame(data= np.c_[cancer['data'], cancer['target']],
                     columns= np.append(cancer['feature_names'],['target']))
# Target is the label of each instance.

# Create series represting the distribution of the data set between ['benign','malignant']
# target = pd.Series(data=cancerDF.target)
target = pd.Series(data=cancerDF.target.value_counts().values, index=['malignant', 'benign'],name='target')
# With out 'target prepare dataset
X = cancerDF.iloc[:,:30]
y = cancerDF['target'].astype(float)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

k = 5
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train,y_train)

print('Accuracy for k = '+str(k)+' :' + str(knn.score(X_test,y_test)))

k_range = range(1,20)
scores = []
# Check accuracy by k choosing
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors = k)
    knn.fit(X_train, y_train)
    scores.append(knn.score(X_test, y_test))

plt.figure()
plt.xlabel('k')
plt.ylabel('accuracy')
plt.scatter(k_range, scores)
plt.xticks([0,5,10,15,20]);
plt.title('k - accuracy graph')
plt.show()

# --------------------------------------------------------------------------------------------

knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train,y_train)
# Find the training and testing accuracies by target value (i.e. malignant, benign)
mal_train_X = X_train[y_train == 0]
mal_train_y = y_train[y_train == 0]
ben_train_X = X_train[y_train == 1]
ben_train_y = y_train[y_train == 1]

mal_test_X = X_test[y_test == 0]
mal_test_y = y_test[y_test == 0]
ben_test_X = X_test[y_test == 1]
ben_test_y = y_test[y_test == 1]


scores = [knn.score(mal_train_X, mal_train_y), knn.score(ben_train_X, ben_train_y),
          knn.score(mal_test_X, mal_test_y), knn.score(ben_test_X, ben_test_y)]

plt.figure()

# Plot the scores as a bar chart
bars = plt.bar(np.arange(4), scores, color=['#4c72b0', '#4c72b0', '#55a868', '#55a868'])

# directly label the score onto the bars
for bar in bars:
    height = bar.get_height()
    plt.gca().text(bar.get_x() + bar.get_width() / 2, height * .90, '{0:.{1}f}'.format(height, 2),
                   ha='center', color='w', fontsize=11)

# remove all the ticks (both axes), and tick labels on the Y axis
plt.tick_params(top='off', bottom='off', left='off', right='off', labelleft='off', labelbottom='on')

# remove the frame of the chart
for spine in plt.gca().spines.values():
    spine.set_visible(False)

plt.xticks([0, 1, 2, 3], ['Malignant\nTraining', 'Benign\nTraining', 'Malignant\nTest', 'Benign\nTest'], alpha=0.8);
plt.title('Training and Test Accuracies for Malignant and Benign Cells', alpha=0.8)
plt.show()
print('Done')
