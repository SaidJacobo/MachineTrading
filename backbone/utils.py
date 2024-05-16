from importlib import import_module
import itertools
from backbone.order import Order
from datetime import datetime
from collections import namedtuple

def load_function(dotpath: str):
    """Carga una función desde un módulo."""
    module_, func = dotpath.rsplit(".", maxsplit=1)
    m = import_module(module_)
    return getattr(m, func)

def get_parameter_combinations(
        models,
        train_window, 
        train_period, 
        trading_strategies, 
        periods_forward_target, 
        stop_loses_in_pips, 
        take_profits_in_pips,
        use_days_in_position
    ):
    parameter_combinations = []
    if None in models:
        strategies = [x for x in trading_strategies if x != 'strategies.ml_strategy']
        parameter_combinations += list(itertools.product(
            [None], [0], [0], strategies
        ))

        models.remove(None)
    
    parameter_combinations += list(itertools.product(
        models, 
        train_window, 
        train_period, 
        trading_strategies, 
        periods_forward_target, 
        stop_loses_in_pips, 
        take_profits_in_pips,
        use_days_in_position
    ))

    return parameter_combinations

def from_mt_order_to_order(mt_order) -> Order:
    order = Order(
        id=mt_order.ticket, 
        order_type='buy' if mt_order.type == 0 else 'sell', 
        ticker=mt_order.symbol, 
        open_time=datetime.fromtimestamp(mt_order.time),
        open_price=mt_order.price_open,
        units=mt_order.volume, #yo tengo unidades y son lotes
        stop_loss=mt_order.sl, 
        take_profit=mt_order.tp
    )

    return order

def from_order_to_mt_order(order:Order) -> dict:
    MtOrder = namedtuple('MtOrder', [
        'ticket','type', 'symbol', 'time','price_open','volume', 'sl', 'tp' 
    ])

    mt_order = MtOrder(
        order.id,
        0 if order.operation_type == 'buy' else 1, 
        order.ticker, 
        order.open_time,
        order.open_price,
        order.units, 
        order.stop_loss, 
        order.take_profit 
    )

    return mt_order