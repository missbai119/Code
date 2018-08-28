import pandas as pd
from pandas import Series,DataFrame
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import calendar
import datetime

train_df = pd.read_csv(r"E:\practice\sharing-bike\train.csv")

test_df = pd.read_csv(r"E:\practice\sharing-bike\test.csv")
#test_df.info()

test_df['casual'] = 0
test_df['registered'] = 0
test_df['count'] = 0
test_df['traintest']='test'
train_df['traintest']='train'
all_df=pd.concat((train_df,test_df))

#日期的处理
all_df['date']=all_df.datetime.apply(lambda x : x.split()[0])
all_df['monthnum']=all_df.datetime.apply(lambda x : int(x.split()[0].split('-')[1]))
all_df['daynum']=all_df.datetime.apply(lambda x : int(x.split()[0].split('-')[2]))

all_df['hour']=all_df.datetime.apply(lambda x : int(x.split()[1].split(":")[0]))
all_df['weekday']=all_df.datetime.apply(lambda dateString : calendar.day_name[datetime.datetime.strptime(dateString.split()[0],"%Y-%m-%d").weekday()])
#按月看自行车用量
all_df.monthnum.value_counts().sort_index().plot(kind='line')
#plt.show()
#按小时进行切分
all_df.groupby('hour').sum()['count'].sort_index().plot(kind='line')
#plt.show()


#划分时段
def hour_section(hour):
    if hour>=0 and hour<=7:
        return 0
    elif hour>=8 and hour<=10:
        return 1
    elif hour>=11 and hour<=15:
        return 2
    elif hour>=16 and hour<=20:
        return 3
    else:
        return 4

all_df['hour_section']=all_df.hour.apply(hour_section)


#按月份查看使用量（训练数据）

fig=plt.figure()
'''
#all_df.loc[all_df.traintest=='train'].groupby(['monthnum'])['count'].mean().plot(kind='bar')
#plt.show()

#按周几和小时查看骑行情况
#hourAggregated = pd.DataFrame(all_df.loc[all_df.traintest=='train'].groupby(['hour','weekday'],sort=True)['count'].mean()).reset_index()
#sn.pointplot(x=hourAggregated['hour'],y=hourAggregated['count'],hue=hourAggregated['weekday'],data=hourAggregated,join=True)
#plt.show()

#按季节查看
#all_df.loc[all_df.traintest=='train'].groupby(['season'])['count'].mean().plot(kind='bar')
#plt.show()

#每个季节每天的使用情况
hourAggregated = pd.DataFrame(all_df.loc[all_df.traintest=='train'].groupby(['hour','season'],sort=True)['count'].mean()).reset_index()
sn.pointplot(x=hourAggregated['hour'],y=hourAggregated['count'],hue=hourAggregated['season'],data=hourAggregated,join=True)
plt.show()

#用量与工作日的关系

hourAggregated = pd.DataFrame(all_df.loc[all_df.traintest=='train'].groupby(['hour','workingday'],sort=True)['count'].mean()).reset_index()
sn.pointplot(x=hourAggregated['hour'],y=hourAggregated['count'],hue=hourAggregated['workingday'],data=hourAggregated,join=True)
plt.show()    

#天气与骑行情况
all_df.groupby('weather')['count'].mean().plot(kind='bar')
plt.show()


all_df['temp_int']=all_df.temp.apply(lambda x :int(x))
all_df.groupby('temp_int')['count'].mean().plot(kind='bar')
plt.show()

#各特征的相关系数
corrMatt=all_df.loc[all_df.traintest=='train',['temp','atemp','casual','registered','humidity','windspeed','count']].corr()
mask=np.array(corrMatt)
mask[np.tril_indices_from(mask)]=False
fig,ax=plt.subplots()
fig.set_size_inches(10,5)
sn.heatmap(corrMatt,mask=mask,vmax=.8,square=True,annot=True)
plt.show()

#注册用户用量
all_df.groupby('hour').sum()['registered'].sort_index().plot(kind='line')
plt.show()

#非注册用户用量
all_df.groupby('hour').sum()['casual'].sort_index().plot(kind='line')
plt.show()

fig,axes=plt.subplots(nrows=2,ncols=2)
fig.set_size_inches(12,10)
sn.boxplot(data=all_df.loc[all_df.traintest=='train'],y='count',ax=axes[0][0])
sn.boxplot(data=all_df.loc[all_df.traintest=='train'],y='count',x='season',ax=axes[0][1])
sn.boxplot(data=all_df.loc[all_df.traintest=='train'],y='count',x='hour',ax=axes[1][0])
sn.boxplot(data=all_df.loc[all_df.traintest=='train'],y='count',x='workingday',ax=axes[1][1])

axes[0][0].set(ylabel='Count',title='Box Plot On Count')
axes[0][1].set(xlabel='Season',ylabel='Count',title='Box Plot On Count Across Season')
axes[1][0].set(xlabel='Hour Of The Day',ylabel='Count',title='Box Plot On Count Across Hour Of The Day')
axes[1][1].set(xlabel='Working Day',ylabel='Count',title='Box Plot On Count Across Working Day')
plt.show()

outliers=np.abs(all_df.loc[all_df.traintest=='train',['count']]-all_df.loc[all_df.traintest=='train',['count']].mean()) > (3*all_df.loc[all_df.traintest=='train',['count']].std())
goodpoints=np.abs(all_df.loc[all_df.traintest=='train',['count']]-all_df.loc[all_df.traintest=='train',['count']].mean()) <= (3*all_df.loc[all_df.traintest=='train',['count']].std())
all_df=pd.concat((all_df.loc[all_df.traintest=='train'][goodpoints['count'].values],all_df.loc[all_df.traintest=='test']))
#all_df.traintest.value_counts()
#目标的正态化
all_df.loc[all_df.traintest=='train',['count']].plot(kind='kde')
plt.show()

import math
all_df.loc[all_df.traintest=='train',['count']]['count'].apply(lambda x:math.log(1+x)).plot(kind='kde')
plt.show()
'''

#日期计算
def date_diff(date):
    first_new_year=str(date[0:4])+'-01-01 00:00:00'
    next_new_year=str(int(date[0:4])+1)+'-01-01 00:00:00'

    date=datetime.datetime.strptime(date,'%Y-%m-%d %H:%M:%S')
    first_new_year=datetime.datetime.strptime(first_new_year,'%Y-%m-%d %H:%M:%S')
    next_new_year=datetime.datetime.strptime(next_new_year,'%Y-%m-%d %H:%M:%S')
    if (abs((date-first_new_year).days))>(abs((date-next_new_year).days)):
        return (abs((date-next_new_year).days))
    else:return (abs((date-first_new_year).days))

all_df['date_newyear_num']=all_df.datetime.apply(date_diff)

all_df=pd.get_dummies(all_df,columns=['season'])
all_df=pd.get_dummies(all_df,columns=['weather'])
all_df=pd.get_dummies(all_df,columns=['monthnum'])
all_df=pd.get_dummies(all_df,columns=['hour'])
all_df=pd.get_dummies(all_df,columns=['weekday'])

import sklearn.preprocessing as preprocessing
scaler=preprocessing.StandardScaler()
temp_scale_param=scaler.fit(all_df[['temp']])
all_df['temp_scaled']=scaler.fit_transform(all_df[['temp']],temp_scale_param)

scaler=preprocessing.StandardScaler()
atemp_scale_param=scaler.fit(all_df[['atemp']])
all_df['atemp_scaled']=scaler.fit_transform(all_df[['atemp']],atemp_scale_param)

scaler=preprocessing.StandardScaler()
humidity_scale_param=scaler.fit(all_df[['humidity']])
all_df['humidity_scaled']=scaler.fit_transform(all_df[['humidity']],humidity_scale_param)

scaler=preprocessing.StandardScaler()
windspeed_scale_param=scaler.fit(all_df[['windspeed']])
all_df['windspeed_scaled']=scaler.fit_transform(all_df[['windspeed']],windspeed_scale_param)

scaler=preprocessing.StandardScaler()
date_newyear_num_scale_param=scaler.fit(all_df[['date_newyear_num']])
all_df['date_newyear_num_scaled']=scaler.fit_transform(all_df[['date_newyear_num']],date_newyear_num_scale_param)

all_df.to_csv('feature_engine.csv')

feature_columns=['holiday', 'workingday', 
       'season_1', 'season_2', 'season_3', 'season_4', 'weather_1',
       'weather_2', 'weather_3', 'weather_4', 'temp_scaled',
       'atemp_scaled', 'humidity_scaled', 'windspeed_scaled',
       'date_newyear_num', 'date_newyear_num_scaled', 'month_April',
       'month_August', 'month_December', 'month_February', 'month_January',
       'month_July', 'month_June', 'month_March', 'month_May',
       'month_November', 'month_October', 'month_September', 'hour_0',
       'hour_1', 'hour_2', 'hour_3', 'hour_4', 'hour_5', 'hour_6',
       'hour_7', 'hour_8', 'hour_9', 'hour_10', 'hour_11', 'hour_12',
       'hour_13', 'hour_14', 'hour_15', 'hour_16', 'hour_17', 'hour_18',
       'hour_19', 'hour_20', 'hour_21', 'hour_22', 'hour_23',
       'weekday_Friday', 'weekday_Monday', 'weekday_Saturday',
       'weekday_Sunday', 'weekday_Thursday', 'weekday_Tuesday',
       'weekday_Wednesday', 'hour_workingday_0_0', 'hour_workingday_0_1',
       'hour_workingday_10_0', 'hour_workingday_10_1',
       'hour_workingday_11_0', 'hour_workingday_11_1',
       'hour_workingday_12_0', 'hour_workingday_12_1',
       'hour_workingday_13_0', 'hour_workingday_13_1',
       'hour_workingday_14_0', 'hour_workingday_14_1',
       'hour_workingday_15_0', 'hour_workingday_15_1',
       'hour_workingday_16_0', 'hour_workingday_16_1',
       'hour_workingday_17_0', 'hour_workingday_17_1',
       'hour_workingday_18_0', 'hour_workingday_18_1',
       'hour_workingday_19_0', 'hour_workingday_19_1',
       'hour_workingday_1_0', 'hour_workingday_1_1',
       'hour_workingday_20_0', 'hour_workingday_20_1',
       'hour_workingday_21_0', 'hour_workingday_21_1',
       'hour_workingday_22_0', 'hour_workingday_22_1',
       'hour_workingday_23_0', 'hour_workingday_23_1',
       'hour_workingday_2_0', 'hour_workingday_2_1', 'hour_workingday_3_0',
       'hour_workingday_3_1', 'hour_workingday_4_0', 'hour_workingday_4_1',
       'hour_workingday_5_0', 'hour_workingday_5_1', 'hour_workingday_6_0',
       'hour_workingday_6_1', 'hour_workingday_7_0', 'hour_workingday_7_1',
       'hour_workingday_8_0', 'hour_workingday_8_1', 'hour_workingday_9_0',
       'hour_workingday_9_1', 'hour_week_section_0', 'hour_week_section_1',
       'hour_week_section_2', 'hour_week_section_3', 'hour_week_section_4',
       'hour_week_section_5', 'hour_week_section_6', 'hour_week_section_7']



X=all_df.loc[all_df.traintest=='train',feature_columns].values
y_casual=all_df.loc[all_df.traintest=='train'].casual.apply(lambda x: np.log1p(x)).values
y_regstered=all_df.loc[all_df.traintest=='train'].registered.apply(lambda x: np.log1p(x)).values
y_all=all_df.loc[all_df.traintest=='train','count'].apply(lambda x: np.log1p(x)).values
X_test=all_df.loc[all_df.traintest=='test',feature_columns].values
X_date=all_df.loc[all_df.traintest=='test','datetime'].values

all_df.loc[all_df.traintest=='train',feature_columns].to_csv("X.csv")
all_df.loc[all_df.traintest=='train'].casual.apply(lambda x: np.log1p(x)).to_csv("y_casual.csv")
all_df.loc[all_df.traintest=='train'].registered.apply(lambda x: np.log1p(x)).to_csv("y_regstered.csv")
all_df.loc[all_df.traintest=='train','count'].apply(lambda x: np.log1p(x)).to_csv("y_all.csv")
all_df.loc[all_df.traintest=='test',feature_columns].to_csv("X_test.csv")
all_df.loc[all_df.traintest=='test','datetime'].to_csv("X_date.csv")



from sklearn.learning_curve import learning_curve
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=10, n_jobs=1,
                        train_sizes=np.linspace(.05, 1., 20), verbose=0, plot=True):
    """
    画出data在某模型上的learning curve.
    参数解释
    ----------
    estimator : 你用的分类器。
    title : 表格的标题。
    X : 输入的feature，numpy类型
    y : 输入的target vector
    ylim : tuple格式的(ymin, ymax), 设定图像中纵坐标的最低点和最高点
    cv : 做cross-validation的时候，数据分成的份数，其中一份作为cv集，其余n-1份作为training(默认为3份)
    n_jobs : 并行的的任务数(默认1)
    """

    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, verbose=verbose)

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    if plot:
        plt.figure()
        plt.title(title)
        if ylim is not None:
            plt.ylim(*ylim)
        plt.xlabel(u"训练样本数")
        plt.ylabel(u"得分")
        plt.gca().invert_yaxis()
        plt.grid()

        plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std,
                         alpha=0.1, color="b")
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std,
                         alpha=0.1, color="r")
        plt.plot(train_sizes, train_scores_mean, 'o-', color="b", label=u"训练集上得分")
        plt.plot(train_sizes, test_scores_mean, 'o-', color="r", label=u"交叉验证集上得分")

        plt.legend(loc="best")

        plt.draw()
        plt.gca().invert_yaxis()
        plt.show()

    midpoint = ((train_scores_mean[-1] + train_scores_std[-1]) + (test_scores_mean[-1] - test_scores_std[-1])) / 2
    diff = (train_scores_mean[-1] + train_scores_std[-1]) - (test_scores_mean[-1] - test_scores_std[-1])
    return midpoint, diff







