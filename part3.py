import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

np.random.seed(2024)

from closedForm import LinearRegressionClosedForm

def transform_input(x):
    '''
    This function transforms the input to generate new features.

    Args:
      x: 2D numpy array of input values. Dimensions (n' x 1)

    Returns:
      2D numpy array of transformed input. Dimensions (n' x K+1)
      
    '''
    X_transformed = np.ones((x.shape[0], 4))
    X_transformed[:, 1] = x.flatten()
    X_transformed[:, 2] = x.flatten() ** 2
    X_transformed[:, 3] = np.cos(x.flatten())

    return X_transformed



    raise NotImplementedError()
    
def read_dataset(filepath):
    '''
    This function reads the dataset and creates train and test splits.
    
    n = 500
    n' = 0.9*n

    Args:
      filename: string containing the path of the csv file

    Returns:
      X_train: 2D numpy array of input values for training. Dimensions (n' x 1)
      y_train: 2D numpy array of target values for training. Dimensions (n' x 1)
      
      X_test: 2D numpy array of input values for testing. Dimensions ((n-n') x 1)
      y_test: 2D numpy array of target values for testing. Dimensions ((n-n') x 1)
      
    '''
    # Write your code here

    dataset = pd.read_csv(filepath)
    split_index =  int(len(dataset) * 0.8)
    x = dataset.drop(['ID', 'score'], axis=1).values
    y = dataset['score'].values

    x_train = x[:split_index]
    y_train = y[:split_index]


    x_test = x[split_index:]
    y_test = y[split_index:]
    # y_test = np.array(dataset['y'][split_index:][1:].values.reshape(-1, 64))

    return x_train, y_train, x_test, y_test
    raise NotImplementedError()


############################################
#####        Helper functions          #####
############################################

def plot_dataset(X, y):
    '''
    This function generates the plot to visualize the dataset  

    Args:
      X : 2D numpy array of data points. Dimensions (n x 64)
      y : 2D numpy array of target values. Dimensions (n x 1)

    Returns:
      None
    '''
    plt.title('Plot of the unknown dataset')
    cmap = plt.cm.get_cmap('tab20', 64)
    
    for i in range(0, 64):
        plt.scatter(X[:,0], y, color=cmap(i), label=f'Feature {i}')
  
    plt.xlabel('x', fontsize=16)
    plt.ylabel('y', fontsize=16)
    plt.show()

# Terminal text coloring
RESET = '\033[0m'
GREEN = '\033[32m'
RED = '\033[31m'

if __name__ == '__main__':
    
    print(RED + "##### Starting experiment #####")
    
    print(RESET +  "Loading dataset: ",end="")
    try:
        X_train, y_train, X_validation, y_validation = read_dataset('/home/surajp2909/Sem1/fml/assgn1/assgmt1/train.csv')
        print(GREEN + "done")
    except Exception as e:
        print(RED + "failed")
        print(e)
        exit()
    
    print(RESET +  "Plotting dataset: ",end="")
    try:
        plot_dataset(X_train, y_train)
        print(GREEN + "done")
    except Exception as e:
        print(RED + "failed")
        print(e)
        exit()
    
    print(RESET + "Performing input transformation: ", end="")
    try:
        X_train = transform_input(X_train)
        X_test = transform_input(X_validation)
        print(GREEN + "done")
    except Exception as e:
        print(RED + "failed")
        print(e)
        exit()
        
    print(RESET + "Caclulating weights: ", end="")
    try:
        linear_reg = LinearRegressionClosedForm()
        linear_reg.fit(X_train,y_train)
        print(GREEN + "done")
    except Exception as e:
        print(RED + "failed")
        print(e)
        exit()
    
    print(RESET + "Checking closeness: ", end="")
    try:
        y_hat = linear_reg.predict(X_test)
        if np.allclose(y_hat, y_validation, atol=1e-02):
          print(GREEN + "done")
        else:
          print(RED + "failed")
    except Exception as e:
        print(RED + "failed")
        print(e)
        exit()