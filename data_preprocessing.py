#!/usr/bin/env python
# coding: utf-8

# **Есть исторические данные(события взаимодействия пользователей с баннерами на сайте).
# Формат данных :
# date - дата когда было произведено действие.
# banner_id - идентификатор баннера
# adv_bid - ставка за 1 показ(в рублях)
# placement - идентификатор страницы на которой произошло взаимодействие
# user_id - id пользователя
# action_type - тип события
# Нужно сделать скрипт(подойдет любой вариант как и jupyter notebook, так и просто скрипт на python) который бы на вход принимал 2 параметра - ставку за 1 показ и  идентификатор страницы. А в результате выводил предсказание о том сколько показов получит этот баннер за  следующий день.**

# In[266]:


import pandas as pd
import numpy as np


# In[284]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler


# In[299]:


start_data = pd.read_csv("test_data_3.csv")


# In[300]:


start_data.groupby(['action_type']).count()


# В первую очередь понятно что совсем не нужны данные о пользователях.
# Конечно, также, видно, что в данных есть повторы, но я не удаляю их потому что предположу, что один и тот же человек может несколько раз увидеть/нажать/добавить в козину один и тот же баннер

# In[301]:


start_data.drop("user_id", axis=1,inplace=True)


# In[302]:


start_data.head()


# Посмотрим о чем вообще датасет

# In[297]:


print(len(start_data.banner_id.unique()),
len(start_data.adv_bid.unique()), len(start_data.date.unique()))


# In[305]:


data = start_data.groupby(["banner_id", "date"])


# In[ ]:


array = []
for banner_id, banner_df in data:
    print(banner_id)
    #print(banner_df)
    print("Количество показов:", len(banner_df), banner_df["adv_bid"].min(), banner_df["placement"].min())
    new_row = [banner_df["adv_bid"].min(), banner_df["placement"].min(), len(banner_df)]
    array.append(new_row)
    print(type(new_row))
processed_set = pd.DataFrame(array)


# Убираю слишком маленькие значения из выборки
# 

# In[248]:


processed_set = processed_set[processed_set[2] > 66]


# Заполняю тренировочный датасет данными (для одних и тех же ставок на одной и той же странице добавляю среднее)

# In[259]:


arr = []
for set_id, set_df in processed_set.groupby([0,1]):
    print(set_df)
    print(set_df[2].mean())
    new_row = [set_df.iloc[0,0], set_df[1].iloc[0], set_df[2].mean()]
    arr.append(new_row)
processed_set_ = pd.DataFrame(arr)


# In[304]:


processed_set_


# In[261]:


X = processed_set_.drop(2, axis=1)
Y = processed_set_[2]


# In[262]:


x_train, x_validation, y_train, y_validation = train_test_split(X, Y, random_state=13)


# In[263]:


model = LinearRegression(normalize=True).fit(x_train, y_train)


# In[264]:


model.score(x_validation, y_validation)


# In[277]:


predictions = model.predict(x_train)


# In[283]:


mae = mean_absolute_error(y_train, predictions)
print(y_train)
print(predictions)
print(mae)


# Попробую постороить регерессию еще одним спобом

# In[285]:


scaler = StandardScaler()
scaler.fit(processed_set_.drop(2, axis=1)) 
scaled_df = scaler.transform(processed_set_.drop(2, axis=1))


# In[293]:


linear_regression_model = SGDRegressor(tol=.01, eta0=.1) 
linear_regression_model.fit(scaled_df, Y)
predictions = linear_regression_model.predict(scaled_df)
mae = mean_absolute_error(Y, predictions) 
print("RMAE: {}".format(np.sqrt(mae)))


# In[294]:


print(y_train)
print(predictions)


# Мне не нравятся полученные результаты, лучше использовать первую модель
