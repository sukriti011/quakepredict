
import matplotlib.pyplot as plt
import numpy as np
from time import time

# Utility function to report best scores
def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")
            
def LinearRegression(train_X, train_y, test_X):
    from sklearn.datasets import load_iris
    from sklearn.linear_model import ElasticNet
    regr = ElasticNet(random_state=0,max_iter=10000).fit(train_X, train_y)
    test_y = regr.predict(test_X)
    train_y = regr.predict(train_X)
    return test_y, train_y

def LinearRegressionWithFeatureSelection(train_X, train_y, test_X):
    from sklearn.linear_model import LassoCV
    from sklearn.feature_selection import SelectFromModel
    fs = LassoCV(cv=5)
    model = SelectFromModel(fs, threshold='median')
    model.fit(train_X, train_y)
    
    print(train_X.shape)
    train_X = model.transform(train_X)
    print(train_X.shape)
    test_X = model.transform(test_X)

    test_y, train_y = LinearRegression(train_X, train_y, test_X)    
    return test_y, train_y

def SupportVectorRegression(train_X, train_y, test_X):
    from sklearn.svm import SVR
    regr = SVR(gamma='scale', C=1.0, epsilon=0.2).fit(train_X, train_y)
    test_y = regr.predict(test_X)    
    train_y = regr.predict(train_X)
    return test_y, train_y

def LinearSupportVectorRegression(train_X, train_y, test_X):
    from sklearn.svm import LinearSVR
    regr = LinearSVR(C=1.0, epsilon=0.1, tol=0.01, max_iter=10000).fit(train_X, train_y)
    test_y = regr.predict(test_X) 
    train_y = regr.predict(train_X)
    return test_y, train_y

def LinearSupportVectorRegressionGridSearch(train_X, train_y):
    from sklearn.model_selection import GridSearchCV
    from sklearn.svm import LinearSVR
    regr = SVR()
    search_params = {'dual': [True, False], 'tol':[0.01, 0.001], 'max_iter':[5000, 10000]}
    gs = GridSearchCV(regr, param_grid=search_params)
    start = time()
    gs.fit(train_X, train_y)
    print("GridSearchCV took %.2f seconds for %d candidate parameter settings."
          % (time() - start, len(gs.cv_results_['params'])))
    report(gs.cv_results_)

def RandomForestRegression(train_X, train_y, test_X):
    from sklearn.ensemble import RandomForestRegressor
    regr = RandomForestRegressor(random_state=1,
                                 n_estimators=250, n_jobs=20, verbose=1).fit(train_X, train_y)
                                 #n_estimators=5000, n_jobs=20, verbose=1).fit(train_X, train_y)
    test_y = regr.predict(test_X)
    train_y = regr.predict(train_X)

    # importances = regr.feature_importances_
    # std = np.std([tree.feature_importances_ for tree in regr.estimators_],
    #              axis=0)
    # indices = np.argsort(importances)
    
    # # Plot the feature importances of the forest
    # plt.figure()
    # plt.title("Feature importances")
    # plt.barh(range(train_X.shape[1]), importances[indices],
    #        color="r", xerr=std[indices], align="center")
    # # If you want to define your own labels,
    # # change indices to a list of labels on the following line.
    # plt.yticks(range(train_X.shape[1]), indices)
    # plt.ylim([-1, train_X.shape[1]])
    # plt.show()
    
    return test_y, train_y, regr

def RandomForestRegressionGridSearch(train_X, train_y):
    from sklearn.model_selection import GridSearchCV
    from sklearn.ensemble import RandomForestRegressor
    regr = RandomForestRegressor(n_jobs=10)
    search_params = {'n_estimators': [50, 100, 200, 250, 500]}
    # search_params = {'n_estimators': [50]}
    gs = GridSearchCV(regr, param_grid=search_params)
    gs.fit(train_X, train_y)
    start = time()
    print("GridSearchCV took %.2f seconds for %d candidate parameter settings."
          % (time() - start, len(gs.cv_results_['params'])))
    report(gs.cv_results_)

def GradientBoostRegression(train_X, train_y, test_X):
    from sklearn.ensemble import GradientBoostingRegressor
    regr = GradientBoostingRegressor(random_state=0,
                                     n_estimators=250).fit(train_X, train_y)
    test_y = regr.predict(test_X)
    train_y = regr.predict(train_X)
    return test_y, train_y

def GradientBoostRegressionGridSearch(train_X, train_y):
    from sklearn.model_selection import GridSearchCV
    from sklearn.ensemble import GradientBoostingRegressor
    regr = GradientBoostingRegressor()
    search_params = {'n_estimators': [100, 200, 300, 500],
                     'loss': ['ls', 'lad', 'huber']}
    gs = GridSearchCV(regr, param_grid=search_params)
    start = time()
    gs.fit(train_X, train_y)
    print("GridSearchCV took %.2f seconds for %d candidate parameter settings."
          % (time() - start, len(gs.cv_results_['params'])))
    report(gs.cv_results_)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, default='gpa')
    parser.add_argument('-m', '--model', type=str, default='elastic')
    parser.add_argument('-o', '--output', type=str, required=False)
    parser.add_argument('-n', '--normalize', action="store_true", default=False)
    args = parser.parse_args()

    if args.output:
        outputPrefix = args.output
    else:
        outputPrefix = args.input + '_' + args.model + '_'
    
    train_X = np.load(args.input + '_train_X.npy')
    train_y = np.load(args.input + '_train_y.npy')
    test_X = np.load(args.input + '_test_X.npy')
    test_y = np.load(args.input + '_test_y.npy')

    if args.normalize:
        print('normalizing data')
        print('before {} {}'.format(train_X.min(), train_X.max()))
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        scaler.fit(train_X)
        train_X = scaler.transform(train_X)
        print('after {} {}'.format(train_X.min(), train_X.max()))
        test_X = scaler.transform(test_X)
    
    if args.model == 'elastic':
        predict_test_y, predict_train_y = LinearRegression(train_X, train_y, test_X)
    elif args.model == 'elastic_fs':
        predict_test_y, predict_train_y = LinearRegressionWithFeatureSelection(train_X, train_y, test_X)
    elif args.model == 'svr_linear':
        predict_test_y, predict_train_y = LinearSupportVectorRegression(train_X, train_y, test_X)
    elif args.model == 'svr_rbf':
        predict_test_y, predict_train_y = SupportVectorRegression(train_X, train_y, test_X)
    elif args.model == 'rfr':
        predict_test_y, predict_train_y, regr = RandomForestRegression(train_X, train_y, test_X)
    elif args.model == 'xgb':
        predict_test_y, predict_train_y = GradientBoostRegression(train_X, train_y, test_X)
    elif args.model == 'svr_linear_gs':
        LinearSupportVectorRegressionGridSearch(train_X, train_y)
        exit(1)        
    elif args.model == 'rfr_gs':
        RandomForestRegressionGridSearch(train_X, train_y)
        exit(1)
    elif args.model == 'xgb_gs':
        GradientBoostRegressionGridSearch(train_X, train_y)
        exit(1)
    else:
        print('Unrecognized model')
        exit(1)
    
    # mse = ((predict_y - test_y)**2).mean(axis=0)
    mae_test = (np.abs(predict_test_y - test_y)).mean()
    mae_train = (np.abs(train_y - predict_train_y)).mean()
    err = np.abs(predict_test_y - test_y)
    with open(outputPrefix + 'scores.txt', "w") as f:
        f.write('{}\nmae_test = {}, mae_train = {}'.format(args.model, mae_test, mae_train))
        f.write('min: {}\n'.format(err.min()))
        f.write('max: {}\n'.format(err.max()))
        f.write('50: {}\n'.format(np.percentile(err, 50)))
        f.write('80: {}\n'.format(np.percentile(err, 80)))
        f.write('90: {}\n'.format(np.percentile(err, 90)))
        f.write('95: {}\n'.format(np.percentile(err, 95)))
        f.write('99: {}\n'.format(np.percentile(err, 99)))

    plt.figure()
    plt.plot(predict_test_y, color='blue', label='Prediction')
    plt.plot(test_y, color='darkorange', label='Ground Truth')
    # if args.input[0]=='g':
    plt.grid(True)
    plt.xlabel('Sample')
    plt.ylabel('Time to next event')
    plt.title(outputPrefix)
    plt.legend(loc="lower right")
    plt.savefig(outputPrefix + 'overlay.png')
    
    plt.figure()
    plt.plot([0, 20], [0, 20], color='navy', linestyle='--')
    plt.scatter(train_y, predict_train_y, color='blue', label='Train')
    plt.scatter(test_y, predict_test_y, color='darkorange', label='Test')
    # if args.input[0]=='g':
    plt.xlim([0.0, 20.0])
    plt.ylim([0.0, 20.0])
    plt.grid(True)
    plt.xlabel('Ground truth')
    plt.ylabel('Prediction')
    plt.title(outputPrefix)
    # plt.legend(loc="lower right")
    plt.savefig(outputPrefix + 'plot.png')

    #plt.figure()
    #plt.hist(np.abs(predict_y - test_y), range=(0,4))
    #plt.savefig(outputPrefix + 'hist.png')
    
    plt.close('all')

    if args.model == 'rfr':
        importances = regr.feature_importances_
        std = np.std([tree.feature_importances_ for tree in regr.estimators_],
                     axis=0)
        indices = np.argsort(importances)

        n = train_X.shape[1]
        n = 20
        indices = indices[-n:]
        # Plot the feature importances of the forest
        plt.figure()
        plt.title("Feature importances")
        plt.barh(range(n), importances[indices],
                 color="r", xerr=std[indices], align="center")
        # If you want to define your own labels,
        # change indices to a list of labels on the following line.
        plt.yticks(range(n), indices)
        plt.ylim([-1, n])
        plt.show()

        
