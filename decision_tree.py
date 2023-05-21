import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree
import graphviz
import pydot

def partition(x):
    """
    Partition the column vector x into subsets indexed by its unique values (v1, ... vk)
    Returns a dictionary of the form
    { v1: indices of x == v1,
      v2: indices of x == v2,
      ...
      vk: indices of x == vk }, where [v1, ... vk] are all the unique values in the vector z.
    """

    # INSERT YOUR CODE HERE
    partitions = {}
    i = 0
    for x_i in x:
        if x_i in partitions:
            partitions[x_i].append(i)
        else:
            partitions[x_i] = [i]
        i += 1
    return partitions


def entropy(y):
    """
    Compute the entropy of a vector y by considering the counts of the unique values (v1, ... vk), in z
    Returns the entropy of z: H(z) = p(z=v1) log2(p(z=v1)) + ... + p(z=vk) log2(p(z=vk))
    """

    # INSERT YOUR CODE HERE
    entropy=0
    
    y_part=partition(y)

    for key in y_part.keys():
        fraction = len(y_part[key])/len(y)
        entropy += -(fraction)*np.log2(fraction)

    return entropy

def mutual_information(x, y):
    """
    Compute the mutual information between a data column (x) and the labels (y). The data column is a single attribute
    over all the examples (n x 1). Mutual information is the difference between the entropy BEFORE the split set, and
    the weighted-average entropy of EACH possible split.
    Returns the mutual information: I(x, y) = H(y) - H(y | x)
    """

    # INSERT YOUR CODE HERE
    h_y = entropy(y)
    p_x = partition(x)

    h_y_x = 0
    for v_i in p_x:
        # selecting new vector from y for each unique value of x
        vec_vxi = y[[i for i in p_x[v_i]]]
        h_y_x += len(p_x[v_i])/len(x)*entropy(vec_vxi)

    # Compute the mutual information
    I_xy = h_y - h_y_x

    return I_xy

def id3(x, y, attribute_value_pairs=None, depth=0, max_depth=5):
    """
    Implements the classical ID3 algorithm given training data (x), training labels (y) and an array of
    attribute-value pairs to consider. This is a recursive algorithm that depends on three termination conditions
        1. If the entire set of labels (y) is pure (all y = only 0 or only 1), then return that label
        2. If the set of attribute-value pairs is empty (there is nothing to split on), then return the most common
           value of y (majority label)
        3. If the max_depth is reached (pre-pruning bias), then return the most common value of y (majority label)
    Otherwise the algorithm selects the next best attribute-value pair using INFORMATION GAIN as the splitting criterion
    and partitions the data set based on the values of that attribute before the next recursive call to ID3.
    The tree we learn is a BINARY tree, which means that every node has only two branches. The splitting criterion has
    to be chosen from among all possible attribute-value pairs. That is, for a problem with two features/attributes x1
    (taking values a, b, c) and x2 (taking values d, e), the initial attribute value pair list is a list of all pairs of
    attributes with their corresponding values:
    [(x1, a),
     (x1, b),
     (x1, c),
     (x2, d),
     (x2, e)]
     If we select (x2, d) as the best attribute-value pair, then the new decision node becomes: [ (x2 == d)? ] and
     the attribute-value pair (x2, d) is removed from the list of attribute_value_pairs.
    The tree is stored as a nested dictionary, where each entry is of the form
                    (attribute_index, attribute_value, True/False): subtree
    * The (attribute_index, attribute_value) determines the splitting criterion of the current node. For example, (4, 2)
    indicates that we test if (x4 == 2) at the current node.
    * The subtree itself can be nested dictionary, or a single label (leaf node).
    * Leaf nodes are (majority) class labels
    Returns a decision tree represented as a nested dictionary, for example
    {(4, 1, False):
        {(0, 1, False):
            {(1, 1, False): 1,
             (1, 1, True): 0},
         (0, 1, True):
            {(1, 1, False): 0,
             (1, 1, True): 1}},
     (4, 1, True): 1}
    """

    # INSERT YOUR CODE HERE. NOTE: THIS IS A RECURSIVE FUNCTION
    
    dtree = {}

    if attribute_value_pairs is None:
        attribute_value_pairs = []
        for index in range (len(x[0])):
            for value in np.unique(x[index]):
            # for val in np.unique(np.array([item[idx] for item in x])):
                attribute_value_pairs.append((index, value))

    attribute_value_pairs = np.array(attribute_value_pairs)

    # check for pure splits
    unique_values_of_y, count_y = np.unique(y, return_counts=True)
    if len(unique_values_of_y) == 1:
        return unique_values_of_y[0]

    if len(attribute_value_pairs) == 0 or depth == max_depth:
        return unique_values_of_y[np.argmax(count_y)]

    info_gain = []

    for feat, val in attribute_value_pairs:
        info_gain.append(mutual_information(np.array((x[:, feat] == val).astype(int)), y))

    info_gain = np.array(info_gain)
    (feat, val) = attribute_value_pairs[np.argmax(info_gain)]

    partitions = partition(np.array((x[:, feat] == val).astype(int)))

    attribute_value_pairs = np.delete(attribute_value_pairs, np.argmax(info_gain), 0)
    for value, indices in partitions.items():
        x_new = x.take(np.array(indices), axis=0)
        y_new = y.take(np.array(indices), axis=0)
        output = bool(value)

        dtree[(feat, val, output)] = id3(x_new, y_new, attribute_value_pairs=attribute_value_pairs, depth=depth+1, max_depth=max_depth)

    return dtree



def predict_example(x, tree):
    """
    Predicts the classification label for a single example x using tree by recursively descending the tree until
    a label/leaf node is reached.
    Returns the predicted label of x according to tree
    """

    # INSERT YOUR CODE HERE. NOTE: THIS IS A RECURSIVE FUNCTION.
    # Node structure: (attribute_index, attribute_value, True/False)
    # Leaf node : { (): class_label, ():class_label }

    for decision_node, child_tree in tree.items():
        index = decision_node[0]
        value = decision_node[1]
        decision = decision_node[2]

        if decision == (x[index] == value):
            if type(child_tree) is not dict:
                class_label = child_tree
            else:
                class_label = predict_example(x, child_tree)

            return class_label


def compute_error(y_true, y_pred):
    """
    Computes the average error between the true labels (y_true) and the predicted labels (y_pred)
    Returns the error = (1/n) * sum(y_true != y_pred)
    """

    # INSERT YOUR CODE HERE
    return (1/len(y_true)) * sum(y_true != y_pred)


def visualize(tree, depth=0):
    """
    Pretty prints (kinda ugly, but hey, it's better than nothing) the decision tree to the console. Use print(tree) to
    print the raw nested dictionary representation.
    DO NOT MODIFY THIS FUNCTION!
    """

    if depth == 0:
        print('TREE')

    for inx, split_criterion in enumerate(tree):
        sub_trees = tree[split_criterion]

        # Print the current node: split criterion
        print('|\t' * depth, end='')
        print('+-- [SPLIT: x{0} = {1}]'.format(split_criterion[0], split_criterion[1]))

        # Print the children
        if type(sub_trees) is dict:
            visualize(sub_trees, depth + 1)
        else:
            print('|\t' * (depth + 1), end='')
            print('+-- [LABEL = {0}]'.format(sub_trees))


def find_depth(tree, tree_depth=1):
    '''
    To calculate the depth of the tree
    '''
    for key in tree:
        if isinstance(tree[key], dict):
            tree_depth = find_depth(tree[key], tree_depth + 1)
    return tree_depth


def confusion_matrix(y_pred, y_true):
    '''
    Function to print the CONFUSION MATRIX for given y_pred and y_true:
    '''
    
    confusion_matrix = [[0, 0],[0, 0]]
    for label_indx in range(len(y_true)):
        if y_pred[label_indx]==y_true[label_indx]:
            if y_pred[label_indx]==1:
                confusion_matrix[0][0] += 1
            else:
                confusion_matrix[1][1] += 1
        else:
            if y_pred[label_indx]==1:
                confusion_matrix[1][0] += 1
            else:
                confusion_matrix[0][1] += 1

    for row in confusion_matrix:
        print(row)


if __name__ == '__main__':

    '''
    Part A: Learn the Decision Tree for depth = (1, 10)
    Compute Average test error and train error and make plots 
    
    '''

    for i in [1,2,3]:

        # Load the training data
        M = np.genfromtxt(f'./data/monks-{i}.train', missing_values=0, skip_header=0, delimiter=',', dtype=int)
        ytrn = M[:, 0]
        Xtrn = M[:, 1:]

        # Load the test data
        M = np.genfromtxt(f'./data/monks-{i}.test', missing_values=0, skip_header=0, delimiter=',', dtype=int)
        ytst = M[:, 0]
        Xtst = M[:, 1:]

        depth = []
        trn_error = []
        tst_error = []

        for d in range(1, 11):
            
            depth.append(d)
            decision_tree = id3(Xtrn, ytrn,max_depth=d)
            
            # compute the test error 
            y_pred = [predict_example(x, decision_tree) for x in Xtrn]
            trn_err = compute_error(ytrn, y_pred)
            trn_error.append(trn_err)

            # Compute the test error
            y_pred = [predict_example(x, decision_tree) for x in Xtst]
            tst_err = compute_error(ytst, y_pred)
            tst_error.append(tst_err)

            # Print Test and Train Error
            print(f'\nTree Depth: {d}')
            print('Train Error = {0:4.2f}%.'.format(trn_err * 100))
            print('Test Error = {0:4.2f}%.'.format(tst_err * 100))  
        

            '''
            Part B: 
            Report the learned decision tree and the confusion matrix on the test set for depth=1 and depth=2 for Monk Data set 1
            '''

            if i==1 and d in [1, 2]:
                visualize(decision_tree, depth=find_depth(decision_tree))
                print('Confusion matrix:')
                confusion_matrix(y_pred, ytst)


        # Plot graphs 
        # X - axis : Tree Depth 
        # Y - axis : Error
        plt.title(f"Monk Data - {i}")
        plt.plot(trn_error, label=f"Train Error")
        plt.plot(tst_error, label=f"Test Error")
        
        # Add labels and legend to the plot
        plt.xlabel("Depth")
        plt.ylabel("Error")
        plt.legend()
        plt.show() 

        print(f'\nFor Monk Data - {i}')
        print(f'Average Train Error is = {np.sum(trn_error)/len(trn_error)}')
        print(f'Average Test Error is = {np.sum(tst_error)/len(tst_error)}')



        '''
        Part C:
        For monks-1, use scikit-learns default decision tree algorithm2 to learn a decision tree. 
        Visualize the learned decision tree using graphviz3. 
        Report the visualized decision tree and the confusion matrix on the test set
        '''

        if i==1:
            clf = tree.DecisionTreeClassifier()
            clf.fit(Xtrn, ytrn)

            # Predicting the results for the test data
            sci_y_pred = clf.predict(Xtst)
            dot_data = tree.export_graphviz(clf, f'Monks-{i}.dot') 
            graph = graphviz.Source(dot_data) 
            (graph,) = pydot.graph_from_dot_file(f'Monks-{i}.dot')
            graph.write_png(f'Monks-{i}-output.png')
            
            # confusion matrix
            print("Confusion matrix for monks-1 learned using sci-kit learn's DecisionTreeClassifier")
            confusion_matrix(sci_y_pred, ytst)
    
    """
    Problem D
    Data set used - Breast Cancer Wisconsin (Original) Data Set
    
    From UCI Machine Learning repository
    Class labels are 2 - benign, 4 - malignant
    
    source: https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Original%29
    
    """
    
    # Load the data
    M = np.genfromtxt(f'./data/breast-cancer-wisconsin.data', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    
    # Shuffling data points before splitting into training and testing sets to ensure no bias
    np.random.shuffle(M)
    
    # Replacing class labels 2 (benign) as 0 and 4 (malignant) as 1 for uniformity

    # Class Label present in the last column
    Y = M[:, -1]            
    Y[Y==2] = 0
    Y[Y==4] = 1

    # splitting dataset into train 80% and test 20%
    # exclusing first column of id numbers and the last column (class labels) for Xtrn and Xtst data set

    splt_point = int(0.8*len(M))
    ytrn = Y[:splt_point]         
    Xtrn = M[:splt_point, 1:-1]       
    ytst = Y[splt_point:]   
    Xtst = M[splt_point:, 1:-1]

    for d in range(1, 3):    
        decision_tree = id3(Xtrn, ytrn, max_depth=d)
        print(decision_tree)
        visualize(decision_tree, depth=find_depth(decision_tree))
        y_pred = [predict_example(x, decision_tree) for x in Xtst]
        print('Confusion matrix:')
        confusion_matrix(y_pred, ytst)
            
    # learning through scikit learn's DecisionTreeClassifier
    clf = tree.DecisionTreeClassifier()
    clf.fit(Xtrn, ytrn)
    y_pred = clf.predict(Xtst)
    dot_data = tree.export_graphviz(clf, 'breast_cancer.dot') 
    graph = graphviz.Source(dot_data) 
    (graph,) = pydot.graph_from_dot_file('breast_cancer.dot')
    graph.write_png(f'breast_cancer-output.png')

    print("Confusion matrix for breast cancer wisconsin learned using sci-kit learn's DecisionTreeClassifier")
    confusion_matrix(y_pred, ytst)

    

        

