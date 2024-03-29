import uuid

class Order():
    def __init__(self, order_type, ticker, open_date, open_price):
        self.id = uuid.uuid1()
        self.ticker = ticker
        self.type = order_type
        self.open_date = open_date
        self.close_date = None
        self.open_price = open_price
        self.close_price = None
        self.profit = None

    def close(self, close_price, close_date):
        self.close_price = close_price
        self.close_date = close_date
        self.profit = self.get_profit()
    
    def get_profit(self):
        return self.open_price - self.close_price if self.type == 'sell' else self.close_price - self.open_price