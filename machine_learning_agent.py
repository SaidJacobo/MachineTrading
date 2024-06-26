import pandas as pd
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.metrics import fbeta_score, make_scorer, auc, roc_curve
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler

class MachineLearningAgent():
  """Agente de Aprendizaje Automático para entrenar y predecir."""

  def __init__(self, tickers, model, param_grid):
    """Inicializa el Agente de Aprendizaje Automático.

    Args:
        tickers (list): Lista de tickers financieros.
        model: Modelo de machine learning.
        param_grid (dict): Parámetros del modelo para búsqueda de hiperparámetros.
    """
    self.model = model
    self.param_grid = param_grid
    self.pipeline = None
    self.tunning = True

    self.stock_predictions = {}
    self.stock_true_values = {}
    for ticker in tickers:
      self.stock_predictions[ticker] = {}
      self.stock_true_values[ticker] = {}
    
    self.train_results = {}

  def predict(self, x):
    """Realiza predicciones.

    Args:
        x (DataFrame): Datos de entrada para realizar las predicciones.

    Returns:
        array: Predicciones del modelo.
    """
    pred = self.pipeline.predict(x)
    return pred

  def predict_proba(self, x):
    """Realiza predicciones de probabilidad.

    Args:
        x (DataFrame): Datos de entrada para realizar las predicciones.

    Returns:
        array: Predicciones de probabilidad del modelo.
    """
    proba = self.pipeline.predict_proba(x)

    try:
      pred = proba[:,1] # si el array tiene dos valores agarro el segundo
    except:
      pred = proba[:,0] # si tiene uno solo es porque esta super seguro de una de las dos clases, y agarro ese valor
      
    return pred

  def train(self, x_train, x_test, y_train, y_test, date_train, verbose=False):
    """Entrena el modelo de machine learning.

    Args:
        x_train (DataFrame): Datos de entrenamiento.
        x_test (DataFrame): Datos de prueba.
        y_train (array): Etiquetas de entrenamiento.
        y_test (array): Etiquetas de prueba.
        verbose (bool, optional): Indica si mostrar información detallada durante el entrenamiento. Por defecto False.
    """
    if self.tunning:
      print('Starting tunning')
      columns_to_scale = x_train.columns
      # Definir el transformador para las columnas que se deben escalar
      scaler = ColumnTransformer(
          transformers=[
              ('scaler', RobustScaler(), columns_to_scale)
          ],
          remainder='passthrough'  # Las demás columnas se mantienen sin cambios
      )

      pipe = Pipeline([
          ('scaler', scaler),
          ('model', self.model)
      ])

      n_splits = 3
      stratified_kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

      search = GridSearchCV(
          pipe,
          self.param_grid,
          n_jobs=-1,
          cv=stratified_kfold,
          scoring=make_scorer(fbeta_score, beta=0.2)
      )

      # Entrenar
      search.fit(x_train, y_train)

      if verbose:
        print("Best parameter (CV score=%0.3f):" % search.best_score_)

        print(f'Best params: {search.best_params_}')

      # Obtengo el best estimator
      self.pipeline = search.best_estimator_

      self.tunning = False

    else:
      print('Starting train')
      print(f'y_train value_counts: {y_train.value_counts()}')
      
      try:
        self.pipeline.fit(x_train, y_train)

      except:
        print('Entrenamiento cancelado')
      
    
    x_train['preds'] = self.pipeline.predict(x_train)
    x_train['target'] = y_train

    fpr, tpr, _ = roc_curve(x_train['preds'], x_train['target'])
    auc_score = auc(fpr, tpr)
    
    self.train_results[date_train] = auc_score

    if verbose:
      print('train auc: ', auc_score)


  def save_predictions(self, date, ticker, y_true, y_pred):
    """Guarda las predicciones del modelo.

    Args:
        date (datetime): Fecha de las predicciones.
        ticker (str): Ticker financiero.
        y_true (array): Valores verdaderos.
        y_pred (array): Valores predichos.
    """
    self.stock_predictions[ticker][date] = y_pred
    self.stock_true_values[ticker][date] = y_true

  def get_results(self):
    """Obtiene los resultados del modelo.

    Returns:
        DataFrame: Predicciones del modelo.
        DataFrame: Valores verdaderos.
    """
    stock_predictions_df = pd.DataFrame(self.stock_predictions)
    stock_true_values_df = pd.DataFrame(self.stock_true_values)
    stock_train_results_df = pd.DataFrame(
      {
        'fecha': self.train_results.keys(), 
        'auc': self.train_results.values()
      }
    )

    stock_predictions_df = stock_predictions_df.reset_index().rename(columns={'index':'fecha'})
    stock_true_values_df = stock_true_values_df.reset_index().rename(columns={'index':'fecha'})

    return stock_predictions_df, stock_true_values_df, stock_train_results_df