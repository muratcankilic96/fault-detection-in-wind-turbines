from imblearn.over_sampling import SMOTE

def oversampling(X, y):
  X_ = X.reshape(X.shape[0], X.shape[1] * X.shape[2])
  sm = SMOTE(random_state=42);
  X_, y_ = sm.fit_resample(X_, y)
  X_ = X_.reshape(X_.shape[0], X.shape[1], X.shape[2], 1)
  return X_, y_