import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

change=3

def combine(test_predict,data):
    data1=np.ravel(test_predict)
    df_pred=pd.DataFrame(data1, columns=['Close_pred'])
    df_actual=data.iloc[len(data)-len(df_pred):]
    df_actual=df_actual.copy()
    df_actual.rename(columns={'Close': 'Close_actual'},inplace=True)
    df_pred.reset_index(drop=True,inplace=True)
    df_actual.reset_index(drop=True,inplace=True)
    combined_df=df_pred.join(df_actual)
    return combined_df


def strat1(combined_df):
    start=0
    pos=0
    c=0
    amount=10000
    pnl=0
    pnl_actual=0
    total_loss=0
    profits = []
    profits_real = []
    trades={"long":0,"short":0}
    for i in range(len(combined_df)):
        if  not pos:
            initial_amount=combined_df['Close_pred'][start]
            qty=10000//initial_amount
            pos=1

            initial_amount_actual=combined_df['Close_actual'][start]
            qty_actual=10000//initial_amount_actual

        s=((combined_df['Close_pred'][i]-initial_amount)/combined_df['Close_pred'][i])*100

        if abs(s)>=change and s<0:
            trades['short']+=1
            pos=0
            start=i


            pnl+=(initial_amount-combined_df['Close_pred'][i])*qty

            pnl_actual+=(initial_amount_actual-combined_df['Close_actual'][i])*qty_actual

            profits_real.append(pnl_actual)


            
        
        if abs(s)>=change and s>0:
            trades['long']+=1
            pos=0
            start=i


            pnl+=(combined_df['Close_pred'][i]-initial_amount)*qty

            pnl_actual+=(combined_df['Close_actual'][i]-initial_amount_actual)*qty_actual

            profits_real.append(pnl_actual)


            
    print(trades)
    #print("profit={}".format(pnl))
    #print("returns={:.2f}%".format((pnl/amount)*100))
    print("profit_real={}".format(pnl_actual))
    
    return profits_real

def strat1_plot(profit):
    plt.figure(figsize=(10, 6))
    plt.plot(profit, label='Equity Curve', color='blue')
    plt.title('Equity Curve')
    plt.xlabel('Trades')
    plt.ylabel('Profit/Loss')
    plt.legend()
    plt.grid(True)
    plt.show()




