from Tree import Node


class DecisionTreeClassifier:
    def decisionTreeTrain(self, data, remainingFeatures, labels):
        guess = self.mostFrequentData(data)          # guess = most frequent answer in data
        if len(remainingFeatures) == 1:              # base case: cannot split further
            return Node(guess, True)
        if self.checkSameData(data):                      # base case: no need to split further
            return Node(guess, True)
        else:                                        # we need to query more features
            score = 0
            yes = []
            no = []
            featureIndex = -1
            removeIndex = -1
            yesSelectedRemainingFeatures = []
            noSelectedRemainingFeatures = []
            for index, feature in enumerate(remainingFeatures):
                tempNo, tempNoSelectedRemainingFeatures = self.dataSubset('n', feature, remainingFeatures, data)
                tempYes, tempYesSelectedRemainingFeatures = self.dataSubset('y', feature, remainingFeatures, data)
                tempScore = self.majorityVote(tempYes)+self.majorityVote(tempNo)
                if tempScore > score:
                    score = tempScore
                    yes = tempYes
                    no = tempNo
                    removeIndex = index
                    featureIndex = labels.index(feature[0])
                    yesSelectedRemainingFeatures = tempYesSelectedRemainingFeatures
                    noSelectedRemainingFeatures = tempNoSelectedRemainingFeatures
            left = self.decisionTreeTrain(no, self.remove_selected_feature(noSelectedRemainingFeatures, removeIndex), labels)
            right = self.decisionTreeTrain(yes, self.remove_selected_feature(yesSelectedRemainingFeatures, removeIndex), labels)
            return Node(score, False, left, right, featureIndex)

    def checkSameData(self, data):
        for i in data:
            if data[0] != i:
                return False
        return True


    def mostFrequentData(self, data):
        arrayCount = [0]*2
        for i in data:
            arrayCount[i] += 1
        return arrayCount.index(max(arrayCount))

    def dataSubset(self, polarity, feature, remainingFeatures, result_set):
        remaining_indices = []
        for index, item in enumerate(feature):
            if item == polarity:
                remaining_indices.append(index)
        return [result for i, result in enumerate(result_set) if i+1 in remaining_indices ],\
               [[element for idx, element in enumerate(feature) if idx in remaining_indices or idx == 0]
                for index, feature in enumerate(remainingFeatures)]

    def majorityVote(self, feature):
        arrayCount = [0] * 2
        for i in feature:
            arrayCount[i] += 1
        return max(arrayCount)

    def remove_selected_feature(self, selected_remaining_features, feature_index):
        return [feature for index, feature in enumerate(selected_remaining_features) if index != feature_index]



    def decision_tree_test(self, tree, test_point):
        if tree.isLeaf:
            return tree.score
        else:
            if test_point[tree.featureIndex] == 'n':
                return self.decision_tree_test(tree.left, test_point)
            else:
                return self.decision_tree_test(tree.right, test_point)

    def measure_accuracy(self, tree, test_data, results):
        correct_results = 0
        for index, data in enumerate(results):
            if data == self.decision_tree_test(tree, [feature[index] for feature in test_data]):
                correct_results += 1
        return float(correct_results/len(results))

