import pandas as pd

t1 = pd.read_csv("lgb_pyst.csv")
t2 = pd.read_csv("lgb_pyst_Keras_6_0.944115611168.csv")
t3 = pd.read_csv("lgb_pyst_KerasLSTM_1_0.988637272183.csv")
t2['click'] =  t1['click']*0.4 +t2['click']*0.3+t3['click']*0.3
t2.to_csv('avg_l_K_6_KL1_4_3_3.csv', index=False)