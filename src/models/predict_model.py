import joblib


class Predict:
    def __init__(self, model_path):
        """
        Initialize the Predict class.

        Args:
            model_path (str): Path to the trained model file.
        """
        self.model = self.load_model(model_path)

    def load_model(self, model_path):
        """
        Load a trained machine learning model.

        Args:
            model_path (str): Path to the trained model file.

        Returns:
            Trained model.
        """
        loaded_model = joblib.load(model_path)
        return loaded_model

    def predict(self, X_test):
        """
        Predict labels using the loaded model.

        Args:
            X_test (pd.DataFrame): Testing features.

        Returns:
            y_pred (array-like): Predicted labels.
        """
        y_pred = self.model.predict(X_test)
        return y_pred
