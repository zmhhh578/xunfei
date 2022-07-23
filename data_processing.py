import pandas as pd
import numpy as np
import datetime
sale_train_need=pd.read_csv('/Users/zhongming/Desktop/sale1/商品需求训练集.csv')
sale_test_need=pd.read_csv('/Users/zhongming/Desktop/sale1/商品需求测试集.csv')
sale_train_month=pd.read_csv('/Users/zhongming/Desktop/sale1/商品月订单训练集.csv')
sale_test_month=pd.read_csv('/Users/zhongming/Desktop/sale1/商品月订单测试集.csv')
###将测试与训练合并
data_sale=pd.concat([sale_train_need,sale_test_need])
t=sale_train_month[['product_id','type']]
m=t.drop_duplicates()
train_df=pd.merge(data_sale,m,on='product_id')
train_df_columns=['product_id','type','date','label','is_sale_day']
df=train_df[train_df_columns]
df['date']  = df['date'].map(lambda x: x[:-3]).values

df=df.pivot_table(index=['product_id','type'],columns='date',aggfunc='sum')

date_range=['2018-02', '2018-03', '2018-04', '2018-05', '2018-06', '2018-07',
       '2018-08', '2018-09', '2018-10', '2018-11', '2018-12', '2019-01',
       '2019-02', '2019-03', '2019-04', '2019-05', '2019-06', '2019-07',
       '2019-08', '2019-09', '2019-10', '2019-11', '2019-12', '2020-01',
       '2020-02', '2020-03', '2020-04', '2020-05', '2020-06', '2020-07',
       '2020-08', '2020-09', '2020-10', '2020-11', '2020-12', '2021-01',
       '2021-02', '2021-03']
data_range_d={}
for i in range(len(date_range)):
       data_range_d[date_range[i]]=i+1
dd=df['label'].reset_index()
dd.columns=['product_id']+['type']+date_range
dat=df['is_sale_day'].reset_index()
dat.columns=['product_id']+['type']+date_range

grid_df=pd.melt(dd,id_vars=['product_id','type'],var_name='date',value_name='label')
grid_df['m']=grid_df['date'].map(data_range_d)
for col in ['product_id','type']:
    grid_df[col]=grid_df[col].astype('category')
grid_df=grid_df[['product_id','type','date','m','label']]
grid_df.to_pickle('grd_part_1.pkl')
dat=pd.melt(dat,id_vars=['product_id','type'],var_name='date',value_name='is_sale_day')
dat.to_pickle('grid_part_3.pkl')
data_mont=pd.concat([sale_train_month,sale_test_month])
data_mont['date'] = data_mont['year'].astype(str) + '-' + data_mont['month'].map(lambda x:'0'+str(x) if x<=9 else str(x))
data_mont.drop(['year','month'],axis=1,inplace=True)
df1=data_mont.pivot_table(index=['product_id','type'],columns=['date'])
dd_stock=df1['start_stock'].reset_index()
dd_stock.columns=['product_id']+['type']+date_range
stock_df=pd.melt(dd_stock,id_vars=['product_id','type'],var_name='date',value_name='start_stock')

dd_end_stock=df1['end_stock'].reset_index()
dd_end_stock.columns=['product_id']+['type']+date_range
stock_end_df=pd.melt(dd_end_stock,id_vars=['product_id','type'],var_name='date',value_name='end_stock')

dd_order=df1['order'].reset_index()
dd_order.columns=['product_id']+['type']+date_range
order_df=pd.melt(dd_order,id_vars=['product_id','type'],var_name='date',value_name='order')
order_df['year']=order_df['date'].map(lambda x:str(x)[:4])

order_df['type_order_sum'] = order_df.groupby(['type', 'date'])['order'].transform('sum')
order_df['order_ratio'] = order_df['order'].values / order_df['type_order_sum'].values

# stock_diff在该类型中的比例


order_df['order_max'] = order_df.groupby(['product_id', 'type'])['order'].transform('max')
order_df['order_min'] = order_df.groupby(['product_id', 'type'])['order'].transform('min')
order_df['order_std'] = order_df.groupby(['product_id', 'type'])['order'].transform('std')
order_df['order_mean'] = order_df.groupby(['product_id', 'type'])['order'].transform('mean')


order_df['order_momentum_m'] = order_df['order'] / order_df.groupby(['product_id', 'type', 'date'])['order'].transform(
    'mean')
order_df['order_momentum_y'] = order_df['order'] / order_df.groupby(['product_id', 'type', 'year'])['order'].transform(
    'mean')

orgin_columns = list(grid_df)

grid_df_order = grid_df.merge(order_df, on=['product_id', 'type', 'date'], how='left')
keep_columns = [col for col in list(grid_df_order) if col not in orgin_columns]
grid_df_order = grid_df_order[['product_id', 'type', 'date', 'm'] + keep_columns]
grid_df_order.to_pickle('grid_part_2.pkl')


stock_df['type_start_sum']=stock_df.groupby(['type','date'])['start_stock'].transform('sum')
stock_df['start_ratio'] = stock_df['start_stock'].values/ stock_df['type_start_sum'].values

stock_df['stock_max'] = stock_df.groupby(['product_id','type'])['start_stock'].transform('max')
stock_df['stock_min'] = stock_df.groupby(['product_id','type'])['start_stock'].transform('min')
stock_df['stock_std'] = stock_df.groupby(['product_id','type'])['start_stock'].transform('std')
stock_df['stock_mean'] = stock_df.groupby(['product_id','type'])['start_stock'].transform('mean')
stock_df['stock_momentum_m'] = stock_df['start_stock']/stock_df.groupby(['product_id','type','date'])['start_stock'].transform('mean')



orgin_columns=list(grid_df)


grid_df_stock=grid_df.merge(stock_df,on=['product_id','type','date'],how='left')
keep_columns=[col for col in list(grid_df_stock) if col not in orgin_columns]
grid_df_stock=grid_df_stock[['product_id','type','date','m']+keep_columns]
grid_df_stock.to_pickle('grid_part_4.pkl')



stock_end_df['type_end_sum']=stock_end_df.groupby(['type','date'])['end_stock'].transform('sum')
stock_end_df['end_ratio'] = stock_end_df['end_stock'].values/ stock_end_df['type_end_sum'].values
stock_end_df['end_stock_max'] = stock_end_df.groupby(['product_id','type'])['end_stock'].transform('max')
stock_end_df['end_stock_min'] = stock_end_df.groupby(['product_id','type'])['end_stock'].transform('min')
stock_end_df['end+stock_std'] = stock_end_df.groupby(['product_id','type'])['end_stock'].transform('std')
stock_end_df['end_stock_mean'] = stock_end_df.groupby(['product_id','type'])['end_stock'].transform('mean')
stock_end_df['end_stock_momentum_m'] = stock_end_df['end_stock']/stock_end_df.groupby(['product_id','type','date'])['end_stock'].transform('mean')



orgin_columns=list(grid_df)

grid_df_end_stock=grid_df.merge(stock_end_df,on=['product_id','type','date'],how='left')
keep_columns=[col for col in list(grid_df_end_stock) if col not in orgin_columns]
grid_df_end_stock=grid_df_end_stock[['product_id','type','date','m']+keep_columns]
grid_df_end_stock.to_pickle('grid_part_5.pkl')


shit_moth=3
Lag_month=[col for col in range(shit_moth,shit_moth+5)]

grid_df = grid_df.assign(**{
        '{}_lag_{}'.format(col, l): grid_df.groupby(['product_id'])[col].transform(lambda x: x.shift(l))
        for l in Lag_month
        for col in ['label']
    })

for i in [3,6,9,12]:
    print('Rolling period:', i)
    grid_df['rolling_mean_'+str(i)] = grid_df.groupby(['product_id'])['label'].transform(lambda x: x.shift(3).rolling(i).mean()).astype(np.float16)
    grid_df['rolling_std_'+str(i)]  = grid_df.groupby(['product_id'])['label'].transform(lambda x: x.shift(3).rolling(i).std()).astype(np.float16)

# Rollings
# with sliding shift
for d_shift in [1,3,6]:
    print('Shifting period:', d_shift)
    for d_window in [3,6,9,12]:
        col_name = 'rolling_mean_tmp_'+str(d_shift)+'_'+str(d_window)
        grid_df[col_name] = grid_df.groupby(['product_id'])['label'].transform(lambda x: x.shift(d_shift).rolling(d_window).mean()).astype(np.float16)
    grid_df.to_pickle('lags_df_' + str(3) + '.pkl')

grid_df['label'][grid_df['m'] > 35 - 3] = np.nan
base_col = list(grid_df)
icols = ['type']
for i in icols:
    grid_df['enc' + i + 'mean'] = grid_df.groupby(col)['label'].transform('mean')
    grid_df['enc' + i + 'std'] = grid_df.groupby(col)['label'].transform('std')
keep_cols = [col for col in list(grid_df) if col not in base_col]
print(keep_cols)
mean_encoding_df = grid_df[['product_id', 'type', 'date', 'm'] + keep_cols]
mean_encoding_df.to_pickle('mean_encoding_df.pkl')


