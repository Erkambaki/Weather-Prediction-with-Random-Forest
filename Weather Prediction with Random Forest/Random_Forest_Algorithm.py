# Imports for decision tree
import numpy as np
import pandas as pd
import random
from Decision_Tree_Algorithm import decision_tree_algorithm, decision_tree_predictions


# Reduced error tree pruning
def filter_df(df, question):
    feature, comparison_operator, value = question.split()
    
    # Continuous feature
    if comparison_operator == "<=":
        df_yes = df[df[feature] <= float(value)]
        df_no =  df[df[feature] >  float(value)]
        
    # Categorical feature
    else:
        df_yes = df[df[feature].astype(str) == value]
        df_no  = df[df[feature].astype(str) != value]
    
    return df_yes, df_no

def determine_leaf(df_train, ml_task):
    
    if ml_task == "regression":
        return df_train.label.mean()
    
    # Classification
    else:
        return df_train.label.value_counts().index[0]

def determine_errors(df_val, tree, ml_task):
    predictions = decision_tree_predictions(df_val, tree)
    actual_values = df_val.label
    
    if ml_task == "regression":
        # Mean squared error (mse)
        return ((predictions - actual_values) **2).mean()
    else:
        # Number of errors
        return sum(predictions != actual_values)

def pruning_result(tree, df_train, df_val, ml_task):
    
    leaf = determine_leaf(df_train, ml_task)
    errors_leaf = determine_errors(df_val, leaf, ml_task)
    errors_decision_node = determine_errors(df_val, tree, ml_task)

    if errors_leaf <= errors_decision_node:
        return leaf
    else:
        return tree

def post_pruning(tree, df_train, df_val, ml_task):
    
    question = list(tree.keys())[0]
    yes_answer, no_answer = tree[question]

    # base case
    if not isinstance(yes_answer, dict) and not isinstance(no_answer, dict):
        return pruning_result(tree, df_train, df_val, ml_task)
        
    # recursive part
    else:
        df_train_yes, df_train_no = filter_df(df_train, question)
        df_val_yes, df_val_no = filter_df(df_val, question)
        
        if isinstance(yes_answer, dict):
            yes_answer = post_pruning(yes_answer, df_train_yes, df_val_yes, ml_task)
            
        if isinstance(no_answer, dict):
            no_answer = post_pruning(no_answer, df_train_no, df_val_no, ml_task)
        
        tree = {question: [yes_answer, no_answer]}
    
        return pruning_result(tree, df_train, df_val, ml_task)
    

def bootstrapping(train_df, n_bootstrap):
    bootstrap_indices = np.random.randint(low=0, high=len(train_df), size=n_bootstrap)
    df_bootstrapped = train_df.iloc[bootstrap_indices]
    
    return df_bootstrapped

# Random forest algorithm
def random_forest_algorithm(train_df, val_df, n_trees, n_bootstrap, n_features, dt_max_depth, ml_task):
    forest = []
    for i in range(n_trees):
        print("Decision-Tree # ", i+1)
        df_bootstrapped = bootstrapping(train_df, n_bootstrap)
        tree = decision_tree_algorithm(df_bootstrapped, ml_task=ml_task ,max_depth=dt_max_depth, random_subspace=n_features)
        tree_pruned = post_pruning(tree, train_df, val_df, ml_task)
        forest.append(tree_pruned)
    
    return forest

# Predictions
def random_forest_predictions(test_df, forest, ml_task):
    df_predictions = {}
    for i in range(len(forest)):
        column_name = "tree_{}".format(i)
        predictions = decision_tree_predictions(test_df, tree=forest[i])
        df_predictions[column_name] = predictions

    df_predictions = pd.DataFrame(df_predictions)
    if ml_task == "regression":
      random_forest_predictions = df_predictions.mean(axis=1)
    else:
      random_forest_predictions = df_predictions.mode(axis=1)[0]
    
    return random_forest_predictions