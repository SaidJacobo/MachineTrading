import pandas as pd
import yaml
import os
import itertools
from importlib import import_module
from machine_learning_agent import MachineLearningAgent
from trading_agent import TradingAgent
from back_tester import BackTester

def load_function(dotpath: str):
    """Carga una función desde un módulo."""
    module_, func = dotpath.rsplit(".", maxsplit=1)
    m = import_module(module_)
    return getattr(m, func)

def get_parameter_combinations(models, train_window, train_period, trading_strategies):
    parameter_combinations = []
    if None in models:
        strategies = [x for x in trading_strategies if x != 'strategies.ml_strategy']
        parameter_combinations += list(itertools.product(
            [None], [0], [0], strategies
        ))

        models.remove(None)
    
    parameter_combinations += list(itertools.product(
        models, train_window, train_period, trading_strategies
    ))

    return parameter_combinations

if __name__ == '__main__':

    # Carga de configuraciones desde archivos YAML
    with open('configs/project_config.yml', 'r') as file:
        config = yaml.safe_load(file)
    
    with open('configs/parameters.yml', 'r') as file:
        parameters = yaml.safe_load(file)
    
    with open('configs/model_config.yml', 'r') as file:
        model_configs = yaml.safe_load(file)

    # Obtención de parámetros del proyecto
    period = config['period']
    mode = config['mode']
    limit_date_train = config['limit_date_train']
    tickers = config["tickers"] 
    days_back = config['days_back_target']
    
    # Obtención de parámetros de entrenamiento
    models = parameters['models']
    train_window = parameters['train_window']
    train_period = parameters['train_period']
    trading_strategies = parameters['trading_strategy']

    # Combinaciones de parámetros
    parameter_combinations = get_parameter_combinations(models, train_window, train_period, trading_strategies)

    for combination in parameter_combinations:
        model_name, train_window, train_period, trading_strategies = combination
        
        # Definición de la ruta de resultados
        results_path = f'mode_{mode}-model_{model_name}-trainwindow_{train_window}-trainperiod_{train_period}-tradingstrategy_{trading_strategies}'
        path = os.path.join('data', results_path)
        
        if os.path.exists(path):
            print(f'El entrenamiento con la configuracion: {results_path} ya fue realizado. Se procederá al siguiente.')
            continue

        # Carga del agente de estrategia de trading
        strategy = load_function(trading_strategies)
        trading_agent = TradingAgent(
            start_money=config['start_money'], 
            trading_strategy=strategy,
            threshold_up=config['threshold_up'],
            threshold_down=config['threshold_down'],
            allowed_days_in_position=config['days_back_target']
        )

        # Configuración del modelo de machine learning
        model = None
        mla = None
        param_grid = None

        if model_name is not None:
            param_grid = model_configs[model_name]['param_grid']
            model = load_function(model_configs[model_name]['model'])(random_state=42)
            mla = MachineLearningAgent(tickers, model, param_grid)

        # Inicio del backtesting
        back_tester = BackTester(
            tickers=tickers, 
            ml_agent=mla, 
            trading_agent=trading_agent
        )

        if not os.path.exists('./data/dataset.csv'):
            back_tester.create_dataset(
                data_path='./data', 
                days_back=days_back, 
                period=period,
                # limit_date_train=limit_date_train
            )

        # data_path = './data/train.csv' if mode == 'train' else './data/test.csv'
        data_path = './data/dataset.csv'

        back_tester.start(
            data_path=data_path,
            train_window=train_window, 
            train_period=train_period,
            mode=mode,
            limit_date_train=limit_date_train,
            results_path=results_path
        )
