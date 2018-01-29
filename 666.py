import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn import preprocessing
# 上面是载入要用的包，没装的自己装一下


data = pd.read_excel('C:\\Users\\zhout\\Desktop\\北京机构数据.xlsx',encoding="gb2312") # 读取文件，自己改路径

data = data.drop(0).dropna() # drop 是删除的意思，删除第一条，和删除缺失值

data.head(5) # 查看数据
data.dtypes #查看数据类型

for i in data.columns[1:]:
	try:
		data[i] = data[i].astype('float')
	except:
		data[i] = pd.Categorical(data[i]).labels.astype('float')

data.head(5) # 查看新生成的数据

data.shape #查看维度

feature = list(data.columns) #选取所有的变量的名称
feature.remove('案件号') #删除案件号
feature.remove('逾期天数') #删除逾期期天数

feature #查看是不是有所有的自变量
label = ['逾期天数'] #定义因变量

x = data[feature] #从data里挑选自变量数据
y = data[label] #data里面挑选因变量数据

x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=1) #划分数据，训练数据：测试集 =3:1

x_train.shape
x_test.shape
y_train.shape
y_test.shape

x_train = preprocessing.scale(x_train) # 为消除量纲影响，按系统默认的，均值=0，方差=1来标准化
x_test = preprocessing.scale(x_test)
y_train = preprocessing.scale(y_train)
y_test = preprocessing.scale(y_test)

lin = LinearRegression() #定义线性模型

lin.fit(x_train,y_train) #开始训练数据

print('截距',lin.intercept_) #打印对应的值
print('特征值：',feature)
print ('对应权重：',lin.coef_)

y_pred = lin.predict(x_test) #喂给训练好的模型测试数据，计算预测值

score = metrics.mean_squared_error(y_true=y_test,y_pred=y_pred) #计算真实值和预测值的均方差

print('按均方差为测试指标，误差为：',score)
