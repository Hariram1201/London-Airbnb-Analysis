import numpy as np # Library for numerical operations and arrays
import matplotlib.pyplot as plt #Library for plotting and visualisation

from sklearn.model_selection import train_test_split # Library to split data into train/test sets
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score #Library to import regression evaluation metrics
from sklearn.preprocessing import StandardScaler #Library for feature scaling for ML models

from sklearn.linear_model import LinearRegression #Imports linear regression model
from sklearn.linear_model import Ridge, Lasso #Imports regularised linear regression models
from sklearn.tree import DecisionTreeRegressor #Imports decision tree regression model
from sklearn.ensemble import RandomForestRegressor #Imports ensemble of decision trees for regression
from sklearn.ensemble import GradientBoostingRegressor #Imports gradient boosting regression model
from xgboost import XGBRegressor #Imports Extreme Gradient Boosting regression model

from sklearn.model_selection import GridSearchCV #Imports library for hyperparameter tuning with cross-validation

def predModSetup(df, features, target):

    """
    Prepare the dataset for modeling by selecting features and target,
    removing missing values, splitting into train/test sets, and scaling features.

    Parameters:
    -----------
    df : pandas.DataFrame
        The input DataFrame containing the dataset.
    features : list of str
        List of feature column names to be used for modeling.
    target : str
        The name of the target variable column.

    Returns:
    --------
    xTrain : numpy.ndarray
        Scaled training feature array.
    xTest : numpy.ndarray
        Scaled testing feature array.
    yTrain : pandas.Series
        Training target values.
    yTest : pandas.Series
        Testing target values.
    """

    #Drop rows with missing values in the selected features and target columns
    dfModel = df[features + [target]].dropna()

    #Separate the features (X) and target (y)
    x = dfModel[features]
    y = dfModel[target]

    #Split the data into training and testing sets (80% train, 20% test)
    xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.2, random_state=42)

    #InitialiSe the StandardScaler for feature scaling
    scaler = StandardScaler()

    #Fit the scaler on the training data and transform the training features
    xTrain = scaler.fit_transform(xTrain)
    #Transform the test features using the same scaler
    xTest = scaler.transform(xTest)

    return xTrain, xTest, yTrain, yTest

def linearReg(xTrain, yTrain, xTest, yTest):

    """
    Train a Linear Regression model and evaluate its performance on the test set.

    Parameters:
    -----------
    xTrain : numpy.ndarray
        Scaled training feature data.
    yTrain : pandas.Series or numpy.ndarray
        Training target values.
    xTest : numpy.ndarray
        Scaled testing feature data.
    yTest : pandas.Series or numpy.ndarray
        Testing target values.

    Returns:
    --------
    lr : LinearRegression object
        The trained linear regression model.
    rmse : float
        Root Mean Squared Error of the predictions.
    mae : float
        Mean Absolute Error of the predictions.
    r2 : float
        R-squared (coefficient of determination) of the predictions.
    """

    #InitialiSe and fit the Linear Regression model
    lr = LinearRegression()
    lr.fit(xTrain, yTrain)

    #Make predictions on the test data
    yPred = lr.predict(xTest)

    #Evaluate the model using common regression metrics
    rmse = np.sqrt(mean_squared_error(yTest, yPred))
    mae = mean_absolute_error(yTest, yPred)
    r2 = r2_score(yTest, yPred)

    return lr, rmse, mae, r2

def ridgeReg(xTrain, yTrain, xTest, yTest):

    """
    Train a Ridge Regression model and evaluate its performance on the test set.

    Parameters:
    -----------
    xTrain : numpy.ndarray
        Scaled training feature data.
    yTrain : pandas.Series or numpy.ndarray
        Training target values.
    xTest : numpy.ndarray
        Scaled testing feature data.
    yTest : pandas.Series or numpy.ndarray
        Testing target values.

    Returns:
    --------
    ridge : Ridge object
        The trained Ridge regression model.
    rmse : float
        Root Mean Squared Error of the predictions.
    mae : float
        Mean Absolute Error of the predictions.
    r2 : float
        R-squared (coefficient of determination) of the predictions.
    """

    # Initialise and fit the Ridge Regression model
    ridge = Ridge(alpha=1.0)
    ridge.fit(xTrain, yTrain)

    #Make predictions on the test data
    yPred = ridge.predict(xTest)

    #Evaluate the model using common regression metrics
    rmse = np.sqrt(mean_squared_error(yTest, yPred))
    mae = mean_absolute_error(yTest, yPred)
    r2 = r2_score(yTest, yPred)

    return ridge, rmse, mae, r2

def lassoReg(xTrain, yTrain, xTest, yTest):

    """
    Train a Lasso Regression model and evaluate its performance on the test set.

    Parameters:
    -----------
    xTrain : numpy.ndarray
        Scaled training feature data.
    yTrain : pandas.Series or numpy.ndarray
        Training target values.
    xTest : numpy.ndarray
        Scaled testing feature data.
    yTest : pandas.Series or numpy.ndarray
        Testing target values.

    Returns:
    --------
    lasso : Lasso object
        The trained Lasso regression model.
    rmse : float
        Root Mean Squared Error of the predictions.
    mae : float
        Mean Absolute Error of the predictions.
    r2 : float
        R-squared (coefficient of determination) of the predictions.
    """

    #Initialise and train the Lasso Regression model
    lasso = Lasso(alpha=0.1)
    lasso.fit(xTrain, yTrain)

    #Make predictions on the test data
    yPred = lasso.predict(xTest)

    #Evaluate the model using common regression metrics
    rmse = np.sqrt(mean_squared_error(yTest, yPred))
    mae = mean_absolute_error(yTest, yPred)
    r2 = r2_score(yTest, yPred)

    return lasso, rmse, mae, r2

def decTree(xTrain, yTrain, xTest, yTest):

    """
    Train a Decision Tree Regressor with hyperparameter tuning using GridSearchCV.

    Parameters:
    -----------
    xTrain : numpy.ndarray
        Scaled training feature data.
    yTrain : pandas.Series or numpy.ndarray
        Training target values.
    xTest : numpy.ndarray
        Scaled testing feature data.
    yTest : pandas.Series or numpy.ndarray
        Testing target values.

    Returns:
    --------
    bestTree : DecisionTreeRegressor
        The best decision tree model after hyperparameter tuning.
    rmse : float
        Root Mean Squared Error of the predictions.
    mae : float
        Mean Absolute Error of the predictions.
    r2 : float
        R-squared (coefficient of determination) of the predictions.
    """

    #Initialise a basic Decision Tree Regressor
    tree = DecisionTreeRegressor(random_state=42)

    #Define grid of hyperparameters to search
    param_grid = {
        'max_depth': [3, 5, 7, 10, 13, 15, 18, 20],
        'min_samples_leaf': [1, 5, 10, 20]
    }

    #Perform grid search with 5-fold cross-validation
    grid = GridSearchCV(tree, param_grid, cv=5, scoring='neg_mean_squared_error')
    grid.fit(xTrain, yTrain)

    #Retrieve the best model from the search
    bestTree = grid.best_estimator_

    #Make predictions on the test set
    yPred = bestTree.predict(xTest)

    #Evaluate the model using common regression metrics
    rmse = np.sqrt(mean_squared_error(yTest, yPred))
    mae = mean_absolute_error(yTest, yPred)
    r2 = r2_score(yTest, yPred)

    return bestTree, rmse, mae, r2

def randForest(xTrain, yTrain, xTest, yTest):

    """
    Train a Random Forest Regressor with hyperparameter tuning using GridSearchCV.

    Parameters:
    -----------
    xTrain : numpy.ndarray
        Scaled training feature data.
    yTrain : pandas.Series or numpy.ndarray
        Training target values.
    xTest : numpy.ndarray
        Scaled testing feature data.
    yTest : pandas.Series or numpy.ndarray
        Testing target values.

    Returns:
    --------
    bestForest : RandomForestRegressor
        The best random forest model after hyperparameter tuning.
    rmse : float
        Root Mean Squared Error of the predictions.
    mae : float
        Mean Absolute Error of the predictions.
    r2 : float
        R-squared (coefficient of determination) of the predictions.
    """

    #Initialise a basic Random Forest Regressor
    forest = RandomForestRegressor(random_state=42)

    #Define grid of hyperparameters to search
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 10, 20],
        'min_samples_leaf': [1, 5, 10]
    }

    #Perform grid search with 3-fold cross-validation
    grid = GridSearchCV(forest, param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
    grid.fit(xTrain, yTrain)

    #Retrieve the best model from the search
    bestForest = grid.best_estimator_

    #Make predictions on the test set
    yPred = bestForest.predict(xTest)

    #Evaluate the model using common regression metrics
    rmse = np.sqrt(mean_squared_error(yTest, yPred))
    mae = mean_absolute_error(yTest, yPred)
    r2 = r2_score(yTest, yPred)

    return bestForest, rmse, mae, r2

def gradBoost(xTrain, yTrain, xTest, yTest):

    """
    Train a Gradient Boosting Regressor with hyperparameter tuning using GridSearchCV.

    Parameters:
    -----------
    xTrain : numpy.ndarray
        Scaled training feature data.
    yTrain : pandas.Series or numpy.ndarray
        Training target values.
    xTest : numpy.ndarray
        Scaled testing feature data.
    yTest : pandas.Series or numpy.ndarray
        Testing target values.

    Returns:
    --------
    bestGb : GradientBoostingRegressor
        The best gradient boosting model after hyperparameter tuning.
    rmse : float
        Root Mean Squared Error of the predictions.
    mae : float
        Mean Absolute Error of the predictions.
    r2 : float
        R-squared (coefficient of determination) of the predictions.
    """

    # Initialise a basic Gradient Boosting Regressor
    gb = GradientBoostingRegressor(random_state=42)

    #Define grid of hyperparameters to search
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2],
        'min_samples_leaf': [1, 5, 10]
    }

    #Perform grid search with 3-fold cross-validation
    grid = GridSearchCV(gb, param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
    grid.fit(xTrain, yTrain)

    #Retrieve the best model from the search
    bestGb = grid.best_estimator_

    #Make predictions on the test set
    yPred = bestGb.predict(xTest)

    #Evaluate the model using common regression metrics
    rmse = np.sqrt(mean_squared_error(yTest, yPred))
    mae = mean_absolute_error(yTest, yPred)
    r2 = r2_score(yTest, yPred)

    return bestGb, rmse, mae, r2

def xgBoost(xTrain, yTrain, xTest, yTest):

    """
    Train an XGBoost Regressor with hyperparameter tuning using GridSearchCV.

    Parameters:
    -----------
    xTrain : numpy.ndarray
        Scaled training feature data.
    yTrain : pandas.Series or numpy.ndarray
        Training target values.
    xTest : numpy.ndarray
        Scaled testing feature data.
    yTest : pandas.Series or numpy.ndarray
        Testing target values.

    Returns:
    --------
    bestXgb : XGBRegressor
        The best XGBoost model after hyperparameter tuning.
    rmse : float
        Root Mean Squared Error of the predictions.
    mae : float
        Mean Absolute Error of the predictions.
    r2 : float
        R-squared (coefficient of determination) of the predictions.
    """

    #Initialise an XGBoost Regressor
    xgb = XGBRegressor(random_state=42, use_label_encoder=False, eval_metric='rmse')

    #Define grid of hyperparameters to search
    param_grid = {
    'n_estimators': [100, 200, 300, 500],             
    'max_depth': [3, 5, 7, 9],                         
    'learning_rate': [0.01, 0.05, 0.1, 0.2],          
    'subsample': [0.6, 0.8, 1.0],                      
    'colsample_bytree': [0.6, 0.8, 1.0],                
    'min_child_weight': [1, 3, 5],                      
    'gamma': [0, 0.1, 0.2, 0.3],                        
}
    
    #Perform grid search with 3-fold cross-validation
    grid = GridSearchCV(xgb, param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
    grid.fit(xTrain, yTrain)

    #Retrieve the best model from the search
    bestXgb = grid.best_estimator_

    #Make predictions on the test set
    yPred = bestXgb.predict(xTest)

    #Evaluate the model using common regression metrics
    rmse = np.sqrt(mean_squared_error(yTest, yPred))
    mae = mean_absolute_error(yTest, yPred)
    r2 = r2_score(yTest, yPred)

    return bestXgb, rmse, mae, r2

def plotFeatureImportance(model, feature_names, model_name):
    """
    Plots the feature importance of a fitted model.
    
    Parameters:
    - model: a trained model object (e.g., RandomForestRegressor)
    - feature_names: list of feature names used in the model
    - model_name: string, name of the model for plot title
    """
    
    # Check if the model has 'feature_importances_' attribute (typical for tree models)
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    # Alternatively, if linear model, get absolute value of coefficients
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_)
    else:
        print(f"Feature importance is not available for {model_name}")
        return
    
    # Sort features by importance in descending order
    sorted_indices = np.argsort(importances)[::-1]
    
    # Reorder feature names and importance scores
    sorted_features = [feature_names[i] for i in sorted_indices]
    sorted_importances = importances[sorted_indices]
    
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(sorted_importances)), sorted_importances, color='skyblue')
    plt.xticks(range(len(sorted_features)), sorted_features, rotation=45, ha='right')
    plt.title(f'Feature Importance - {model_name}')
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.tight_layout()
    plt.show()