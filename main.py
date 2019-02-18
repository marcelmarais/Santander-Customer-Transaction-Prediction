import pandas as pd
import numpy as np
import time 



import KaggleTools.tools as kgt
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


train_data = pd.read_pickle('Data/train.pkl')

x = train_data
target = train_data['target']
clean_train_data = train_data.drop(columns = ['target','ID_code'])

X_train, X_test, y_train, y_test = train_test_split(clean_train_data, target, test_size=0.2, random_state=42)

#train_data = lgb.Dataset(X_train, label=y_train)
#test_data = lgb.Dataset(X_test, label= y_test )

model = lgb.LGBMClassifier(max_depth=5,
                               n_estimators=999999,
                               learning_rate=0.02,
                               colsample_bytree=0.3,
                               num_leaves=2,
                               metric='auc',
                               objective='binary', 
                               n_jobs=-1 
                               )


start = time.time()
model.fit(X_train, y_train, 
              eval_set=[(X_test, y_test)],
              verbose=0, 
              early_stopping_rounds=100)

cv_val = model.predict_proba(X_test)[:,1]
auc_lgb  = round(roc_auc_score(y_test, cv_val),9)

#kgt.upload_to_kaggle('santander-customer-transaction-prediction')

print(auc_lgb)

model.booster_.save_model('model.txt')

end = time.time()
print("Time taken = ",(end - start)/60, "minutes")
