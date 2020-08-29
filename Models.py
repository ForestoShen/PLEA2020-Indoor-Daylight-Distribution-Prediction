import sklearn
import numpy as np
import pandas as pd
import xgboost as xgb

import time
import joblib

# data process and helper
from sklearn.model_selection import cross_val_score, train_test_split,GridSearchCV, learning_curve
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error
from sklearn import preprocessing

# model type
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor

# bayesian opt
import skopt
from skopt.space import Real, Categorical, Integer
from skopt.utils import use_named_args
from skopt.plots import plot_convergence,plot_objective,plot_evaluations
from skopt.callbacks import TimerCallback,DeadlineStopper,VerboseCallback,DeltaXStopper

# ignore all future warnings
import warnings
warnings.filterwarnings("ignore")

# helper function
def report_callback(res):
    print("best params so far: %s" % res.x)
    print("current param: ",res.x_iters[-1])

def scoring(y_true,y_pred,multioutput="uniform_average"):
    
    if multioutput=="uniform_average":
        r_sq = r2_score(y_true, y_pred,multioutput="variance_weighted")
    else:
        r_sq = r2_score(y_true, y_pred,multioutput=multioutput)
    mae = mean_absolute_error(y_true, y_pred, multioutput=multioutput)
    mse = mean_squared_error(y_true, y_pred, multioutput=multioutput)
    mape = mae/y_true.mean(0)*100
    if multioutput=="uniform_average": mape=mape.mean()
    result = {"r2":r_sq,
            "MAE":mae,
            "MAPE":mape,
            "MSE":mse
            }
    return result

def load_data():
    data = np.loadtxt(r"dataset\yearly_UDI_4level_pp_final.csv",skiprows=1,delimiter=",")
    return "plea_daylight_dataset",data[:,:7],data[:,7:]

# define model here
def test_baseline():
    t = time.clock()
    baseline_cv = cross_val_score(LinearRegression(),X_train, y_train,cv=5)
    print(" Linear Regression finished in %s s"%(time.clock()-t))
    print(" Linear Regression CV r2 score : %s"%(baseline_cv.mean()))
    lr = LinearRegression().fit(X_train, y_train)
    score = scoring(y_test,lr.predict(X_test),multioutput="uniform_average")
    print("Linear Regression test ", score)
    #print("Linear Regression valid", lr.score(X_valid,Y_valid))
    metric = "_".join([str(v) for v in score.values()])
    joblib.dump(lr,"result/{}_{}_best_{}.model".format(name,"Linear",metric))

class MLP(MLPRegressor):
    def __init__(self,layer_number=2,neuron_number=128,alpha=0.0001,**kwargs):
        self.layer_number=layer_number
        self.neuron_number=neuron_number
        self.alpha=alpha
        self.architecture = self.layer_number*[self.neuron_number]
        super().__init__(hidden_layer_sizes=self.architecture,
                        alpha=self.alpha,
                        solver="lbfgs",
                        max_iter=2000)
    def set_params(self,**kwargs):
        self.layer_number=kwargs['layer_number']
        self.neuron_number=kwargs['neuron_number']
        self.alpha=kwargs['alpha']
        self.architecture =  self.layer_number * [self.neuron_number]
        self.hidden_layer_sizes = self.architecture
        #super().set_params(**kwargs)

def get_model(modelname="MLP",maxiter=30):
    if modelname=="MLP":
        opt = skopt.BayesSearchCV( 
        MLP(),
        {'layer_number': (1,4),
        "neuron_number": (8,512),
        "alpha": (1e-6,0.1,"log-uniform")
        },verbose=1,
        n_iter = maxiter,n_jobs=5,cv=5,return_train_score=True,scoring="neg_mean_squared_error")
    
    if modelname=="RF":
        opt = skopt.BayesSearchCV(
        RandomForestRegressor(),
        {'max_depth': (2,20),
        "n_estimators": (10,2000),
        "min_samples_split":(2,5),
        "min_samples_leaf":(1,5)

        },n_iter = maxiter,n_jobs=5,cv=5,return_train_score=True,scoring="neg_mean_squared_error")
    if modelname=="GBT":
        model = xgb.XGBRegressor(objective='reg:squarederror',
                learning_rate=0.1,
                #nround=5000,
                max_depth=4,
                min_child_weight=2,
                n_estimators=1000,
                subsample=0.6,
                booster='gbtree',
                reg_lambda=1,
                gamma=10,
                )
        multiout = MultiOutputRegressor(model,n_jobs=1)
        opt = skopt.BayesSearchCV(multiout,
                {'estimator__learning_rate':(0.001, 0.05,"uniform"), 
                'estimator__gamma':(0.1, 10,"uniform"), 
                'estimator__n_estimators':(10,1000),
                'estimator__min_child_weight':(1,5),
                'estimator__subsample':(0.5,1.0,"uniform"),
                'estimator__max_depth':(2,8)
                },n_iter=maxiter,cv=5,verbose=0,n_jobs=5,return_train_score=True,scoring="neg_mean_squared_error")
    if modelname=="SVR":
        multiout = MultiOutputRegressor(SVR(),n_jobs=1)
        opt = skopt.BayesSearchCV(
        multiout,
        {'estimator__C': (0.1,1000,"uniform"),
        #"estimator__kernel":["poly"],
        "estimator__epsilon":(0,1,"uniform"),
        #"estimator__coef0": (0,1,"uniform"),
        "estimator__gamma": (0.01,10,"uniform"),
        #"estimator__degree":[2],
        
        },n_iter = maxiter,n_jobs=5,cv=5,return_train_score=True,verbose=1,scoring="neg_mean_squared_error")
    
    return opt

if __name__ =="__main__":
    #### load experiment data
    name,X,Y = load_data()
    # prepare data
    X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size=0.15,random_state=0)
    scaler = preprocessing.MinMaxScaler().fit(X_train)
    X_train = scaler.transform(X_train)[:400]
    X_test = scaler.transform(X_test)[:100]
    y_train = y_train[:400]
    y_test = y_test[:100]
    
    joblib.dump(scaler, "dataset/%s.scalar"%name) 
    print("load data success!")

    ### test baseline model
    test_baseline()

    allmodels = ["MLP","GBT","RF","SVR"]
    for mname in allmodels:
        ### bayesian optimization selected model
        opt = get_model(mname,30)
        print("training start")
        callbacks = [DeadlineStopper(60*60*3),report_callback,VerboseCallback(50)]
        res = opt.fit(X_train, y_train,callback=callbacks)
        print("best params : %s" % opt.best_params_)
        print("best val. score: %s" % opt.best_score_)
        print("test r2: %s" % r2_score(y_test,opt.predict(X_test)))
        result = pd.DataFrame(opt.cv_results_)

        result.to_csv("result/{name}+{mtype}.csv".format(name=name,mtype=mname))
        best_model = opt.best_estimator_
        
        y_pred = best_model.predict(X_test)
        score = scoring(y_test,y_pred)
        print("test score: ",score)
        metric = "_".join([str(v) for v in score.values()])
        joblib.dump(best_model,"result/{}_{}_best_{}.model".format(name,mname,metric))
