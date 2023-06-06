import numpy as np

def split_data(X,labels,train_percentage, random_seed=42):
    
    X_heart=X[labels==1]
    X_noheart=X[labels==0]
    
    mult=train_percentage/100
    lim_h=int(mult*X_heart.shape[0])
    lim_noh=int(mult*X_noheart.shape[0])
    
    X_train=np.row_stack(( X_heart[:lim_h],X_noheart[:lim_noh] ))
    train_ones=np.ones( (X_heart[:lim_h]).shape[0] ).reshape(-1,1)
    train_zeros=np.zeros( (X_noheart[:lim_noh]).shape[0] ).reshape(-1,1)
    y_train=np.row_stack ((train_ones,train_zeros)).squeeze()
    
    
    X_test=np.row_stack(( X_heart[lim_h:],X_noheart[lim_noh:] ))
    test_ones=np.ones( (X_heart[lim_h:]).shape[0] ).reshape(-1,1)
    test_zeros=np.zeros( (X_noheart[lim_noh:]).shape[0] ).reshape(-1,1)
    y_test=np.row_stack ((test_ones,test_zeros)).squeeze()
    
    #Shuffle training set
    np.random.seed(random_seed)
    indices=np.random.permutation(X_train.shape[0])
    X_train = X_train[indices]
    y_train = y_train[indices]
        
    return X_train, y_train, X_test, y_test