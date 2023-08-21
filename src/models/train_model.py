import joblib
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV


class Train:
    def train_model(self, X_train, y_train, model_name):
        """
        Train a model using XGBoost regressor and hyperparameter tuning.

        Args:
            X_train (pd.DataFrame): Training features.
            y_train (pd.Series): Training target.
            model_name (str): Name for the saved model file.

        Returns:
            best_model: The best trained model.
        """
        # Create an XGBoost regressor with squared error objective and RMSE metric
        xgb = XGBRegressor(objective='reg:squarederror', eval_metric='rmse')

        # Define the hyperparameter grid for GridSearchCV
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 4, 5],
            'learning_rate': [0.1, 0.01, 0.001]
        }

        # Perform hyperparameter tuning using GridSearchCV with RMSE scoring
        grid_search = GridSearchCV(estimator=xgb, param_grid=param_grid,
                                   cv=3, verbose=1, n_jobs=-1,  scoring='neg_root_mean_squared_error')

        grid_search.fit(X_train, y_train)

        # Get the best trained model from GridSearchCV
        best_model = grid_search.best_estimator_

        # Save the best model to a file using joblib
        model_path = f'../models/{model_name}.pkl'
        joblib.dump(best_model, model_path)

        # Get feature importances
        feature_importances = best_model.feature_importances_
        feature_names = X_train.columns

        # Sort features by importance
        feature_importance_df = pd.DataFrame(
            {'Feature': feature_names, 'Importance': feature_importances})
        feature_importance_df = feature_importance_df.sort_values(
            by='Importance', ascending=False)

        # Get top 5 training scores with corresponding parameters
        cv_results = pd.DataFrame(grid_search.cv_results_)
        top_scores = cv_results.nsmallest(5, 'rank_test_score')[
            ['params', 'mean_test_score']]

        return best_model, feature_importance_df, top_scores
