from test_main import Bot
import pandas as pd
import numpy as np
from strategy1 import *

stock_name=input("Enter the name of the stock")
start_date=input("Enter the start date")
end_date=input("Enter the end date")

a=Bot(stock_name,start_date,end_date)
print("------B&H EQUITY CURVE------")
a.bh_plot()

print("--------CREATING MODEL-------")
test_predict,data=a.model()
combined_df=combine(test_predict=test_predict,data=data)
profits_real=strat1(combined_df)
print("-------EQUITY CURVE OF MODEL 1-------")
strat1_plot(profit=profits_real)

