from FormatDataset import read_and_format_data as read
from DecesionTreeClassifier import DecisionTreeClassifier as Classifier
dataset = read()
classifier = Classifier()
decision_tree = classifier.decisionTreeTrain(dataset[0],
                                             dataset[1:],
                                             [label[0] for label in dataset[1:]])
accuracy = classifier.measure_accuracy(decision_tree,
                                  [[value for index, value in enumerate(feature) if index != 0] for feature in dataset[1:]],
                                  dataset[0])
print('Accuracy of decision tree model is: %f' %accuracy)
