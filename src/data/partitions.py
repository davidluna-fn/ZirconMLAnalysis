from sklearn.model_selection import train_test_split


def split_data(X, y):
    """
    Split the data into balanced training and testing sets while shuffling.

    Args:
        X (pd.DataFrame): Features data.
        y (pd.Series): Target data.

    Returns:
        X_train (pd.DataFrame): Training features.
        X_test (pd.DataFrame): Testing features.
        y_train (pd.Series): Training target.
        y_test (pd.Series): Testing target.
    """
    # Split the data into training and testing sets with shuffling
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test
