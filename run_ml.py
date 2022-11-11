# DeepGOld
# Run ML algorithms on datasets generated from network outputs
# copyright 2022 moshe sipper  
# www.moshesipper.com 

description='coopnet' # for args parser

import argparse
import numpy as np
from time import process_time
import optuna
# optuna.logging.set_verbosity(optuna.logging.ERROR) # optuna.logging.WARNING
from sklearn.metrics import accuracy_score # balanced_accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import RidgeClassifier, SGDClassifier, LogisticRegression, PassiveAggressiveClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier


# hyperparameters categorical sets and numerical ranges used by optuna
hyper_params = { 
SGDClassifier: [ ['categorical', 'penalty', ['l2', 'l1', 'elasticnet'] ],
                 ['float', 'alpha', 1e-5, 1, 'log'] ],
PassiveAggressiveClassifier: [ ['float', 'C', 1e-2, 10, 'log'],
                               ['categorical', 'fit_intercept', [True, False] ],
                               ['categorical', 'shuffle', [True, False] ] ],
RidgeClassifier: [ ['categorical', 'solver', ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga'] ],
                   ['float', 'alpha', 1e-3, 10, 'log'] ],
LogisticRegression: [ [ 'categorical', 'penalty', ['l1', 'l2'] ],
                      [ 'categorical', 'solver', ['liblinear', 'saga'] ] ],
KNeighborsClassifier: [ ['categorical', 'weights', ['uniform', 'distance'] ],
                        ['categorical', 'algorithm', ['auto', 'ball_tree', 'kd_tree', 'brute'] ],
                        ['int', 'n_neighbors', 2, 20, 'nolog'] ],
RandomForestClassifier: [ ['int', 'n_estimators', 10, 1000, 'log'],
                          ['float', 'min_weight_fraction_leaf', 0., 0.5, 'nolog'],
                          ['categorical', 'max_features', ['auto','sqrt','log2'] ] ],
MLPClassifier: [ ['categorical', 'activation', ['identity', 'logistic', 'tanh', 'relu'] ],
                 [ 'categorical', 'solver', ['lbfgs', 'sgd', 'adam'] ],
                 [ 'categorical', 'hidden_layer_sizes', [(64,64), (64,64,64), (64,64,64,64), (64,64,64,64)] ] ],
XGBClassifier: [ ['int', 'n_estimators', 10, 1000, 'log'],
                 ['float', 'learning_rate', 0.01, 0.2, 'nolog'],
                 ['float', 'gamma', 0., 0.4, 'nolog'] ],
LGBMClassifier: [ ['int', 'n_estimators', 10, 1000, 'log'],
                  ['float', 'learning_rate', 0.01, 0.2, 'nolog'],
                  ['float', 'bagging_fraction', 0.5, 0.95, 'nolog'] ],
CatBoostClassifier: [ ['int', 'iterations', 2, 10, 'nolog'],
                      ['int', 'depth', 2, 10, 'nolog'],
                      ['float', 'learning_rate', 1e-2, 10, 'log'],
                     ]
}

ml_algs = dict(zip([k.__name__ for k in hyper_params.keys()], list(hyper_params.keys()))) 
# ml_algs: key is algorithm as string, val is algortihm as function name


class Objective(object): # used by optuna
    def __init__(self, alg, X, y):
        self.alg = alg
        self.X = X
        self.y = y

    def create_model(self, trial):
        kwargs = {'hidden_layer_sizes': (64,64,64,64,64)} if self.alg==MLPClassifier else {}
        for param in hyper_params[self.alg]:
            param_name = f'{self.alg.__name__}_{param[1]}'
            if param[0] == 'categorical':
                p = trial.suggest_categorical(param_name, param[2])
            elif param[0] == 'int':
                p = trial.suggest_int(param_name, param[2], param[3], log=param[4]=='log')
            elif param[0] == 'float':
                p = trial.suggest_float(param_name, param[2], param[3], log=param[4]=='log')
            else:
                exit(f'create_model, unknown hyperparameter type: {param[0]}')
            kwargs.update({param[1]: p})
        model = self.alg(**kwargs)
        return model

    def __call__(self, trial):
        model = self.create_model(trial)
        model.fit(self.X, self.y)
        trial.set_user_attr(key='best_model', value=model)
        return accuracy_score(self.y, model.predict(self.X))
    
        '''
        # Perhaps a better score but much slower (for our purposes the above score 
        # proved good enough to drive the hyperparam search):
        model = self.create_model(trial)
        X, y = self.X, self.y 
        
        scores = []
        kf = KFold(n_splits=5, shuffle=True)
        for train_index, test_index in kf.split(X):
            model.fit(X[train_index], y[train_index])            
            scores.append(accuracy_score(y[test_index], model.predict(X[test_index])))
        
        final_score = mean(scores)

        model.fit(X, y)
        trial.set_user_attr(key='best_model', value=model)
      
        return final_score
        '''
# end class Objective


def optuna_callback(study, trial):
    if study.best_trial.number == trial.number:
        study.set_user_attr(key='best_model', value=trial.user_attrs['best_model'])


def get_args():
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--alg', type=str, action='store', default='RidgeClassifier',
                        help=f'ML algorithm to use, one of {ml_algs} (default: RidgeClassifier)')
    parser.add_argument('--dataset', type=str, action='store', default='test',
                        help='dataset: cifar100, imagenet, or test (default: test)')
    parser.add_argument('--optuna', action='store_true', default=False,
                        help='run optuna (default: False)')
    parser.add_argument('--trials', type=int, default=25, 
                        help='number of optuna trials (default: 25)')
    parser.add_argument('--dir', type=str, action='store', default='Results',
                        help='dir where train & test csv files are stored, where to store results (default: Results)')    
    return parser.parse_args()


def load_from_csv(f):
    data = np.loadtxt(f, delimiter=',')
    X, y = data[:, :-1], data[:, -1].astype(int)
    return X, y


def main():     
    args = get_args()
    optn = '-optuna' if args.optuna else ''
    with open(f'{args.dir}/{args.alg}{optn}.ml', 'w') as f:
        for arg in vars(args):
            print (arg, getattr(args, arg), flush=True) # job output file
            print (arg, getattr(args, arg), file=f) # results file
    assert args.alg in list(ml_algs.keys()), f'{args.alg} not implemented'
    assert args.dataset in ['fashionmnist', 'cifar10', 'cifar100', 'tinyimagenet', 'imagenet', 'clf'], f'unknown dataset {args.dataset}' 
    # clf is for debug purposes only, created through sklearn funcs:
    # X, y = make_classification(n_samples=1024, n_features=20, n_informative=10, n_classes=5, class_sep=0.98)
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    
    for n in [3,7,11]:
        start_time = process_time()
        train_file = f'{args.dir}/{args.dataset}-train-{n}.csv'
        test_file = f'{args.dir}/{args.dataset}-test-{n}.csv'
        with open(f'{args.dir}/{args.alg}{optn}.ml', 'a') as f:
            print(train_file, test_file, file=f)    
    
        X_train, y_train = load_from_csv(train_file)
        X_test, y_test = load_from_csv(test_file)
        with open(f'{args.dir}/{args.alg}{optn}.ml', 'a') as f:
            print(f'X_train.shape {X_train.shape}, y_train.shape {y_train.shape}', file=f)
            print(f'X_test.shape {X_test.shape}, y_test.shape {y_test.shape}', file=f)

        sc = StandardScaler() 
        X_train = sc.fit_transform(X_train) 
        X_test = sc.transform(X_test) 
        
        if args.optuna:
            objective = Objective(ml_algs[args.alg], X_train, y_train)
            study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials=args.trials, callbacks=[optuna_callback])
            model = study.user_attrs['best_model']
            print(study.best_trial.params, file=f)        
        
        else:
            kwargs = {'hidden_layer_sizes': (64,64,64,64,64)} if args.alg=='MLPClassifier' else {}
            model = ml_algs[args.alg](**kwargs)
            model.fit(X_train, y_train)     
        
        test_acc = accuracy_score(y_test, model.predict(X_test))
        with open(f'{args.dir}/{args.alg}{optn}.ml', 'a') as f:
            print(f'{args.dataset}, {args.alg}{optn}, {n}, test accuracy score: {100.*test_acc:.2f}%', file=f)
            print(f'{args.alg} {n} time {process_time() - start_time}', file=f)
        

if __name__ == '__main__':
    main()
