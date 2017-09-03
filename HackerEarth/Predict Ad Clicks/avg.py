import pandas as pd

t1 = pd.read_csv("lgb_pyst.csv")
t2 = pd.read_csv("lgb_pyst_Keras_4_0.967189916545.csv")
t2['click'] =  t2['click']*0.8 +t1['click']*0.2
t2.to_csv('avg_lgb_pyst_Keras_4_2_8.csv', index=False)