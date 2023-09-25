# 2022-05-26  14:32
import pandas as pd
import numpy as np
import csv
import matplotlib.pyplot as plt
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
'''
path = '../../../网课课件/一、python/第一章 python基础/ml-1-1-env-and-datastruct.git/data/BeijingPM20100101_20151231.csv'

f = open(path, 'r')
reader = csv.DictReader(f)

pm_name_list = ['PM_Dongsi', 'PM_Dongsihuan', 'PM_Nongzhanguan', 'PM_US Post']
pm_avarage = []
iws = []
num = 0
pm_sum = 0
for row in reader:
    for i in row:
        if i in pm_name_list:
            if row[i] == 'NA':
                pm_sum += 0
            else:
                pm_sum += int(row[i])
                num += 1
    if num == 0:
        pm_avarage.append('NA')
    else:
        pm_avarage.append(pm_sum/num)
    for j in row:
        if j == 'Iws':
            iws.append(row[j])
pm_index = []
iws_index = []
for i in range(len(pm_avarage)):
    if pm_avarage[i] == 'NA':
        pm_index.append(i)

for j in range(len(iws)):
    if iws[j] == 'NA':
        iws_index.append(j)
li = list(set(pm_index + iws_index))
li.sort()
for i in li[::-1]:
    pm_avarage.pop(i)
    iws.pop(i)

iws = list(map(float, iws))
iws_mean = sum(iws) / len(iws)
pm_mean = sum(pm_avarage) / len(pm_avarage)

iws_std = (sum(map(lambda x: (x - iws_mean) ** 2, iws)) / len(iws)) ** 0.5
pm_std = (sum(map(lambda x: (x - pm_mean) ** 2, pm_avarage)) / len(pm_avarage)) ** 0.5

pm_z_score = list(map(lambda x: (x - pm_mean) / pm_std, pm_avarage))
iws_z_score = list(map(lambda x: (x - iws_mean) / iws_std, iws))

plt.scatter(pm_z_score, iws_z_score)
plt.show()
'''
'''
university_lst = [["河北省","石家庄","河北师范大学"],
            ["河南省","郑州","郑州大学"],
            ["河北省","石家庄","河北科技大学"],
            ["河北省","石家庄","河北经贸大学"],
            ["河北省","保定","保定学院"],
            ["河北省","邯郸","河北工程大学"],
            ["河南省","郑州","河南工业大学"],
            ["河南省","郑州","河南中医药大学"],
            ["河北省","石家庄","河北医科大学"],
            ["河北省","保定","河北农业大学"],
            ["河北省","邯郸","邯郸学院"]]
df = pd.DataFrame(university_lst, columns=['省', '省会', '大学'])
df.set_index(['省', '省会'], inplace=True)
df.sort_index(level=1, inplace=True)
print(df)
# print(df.loc['河北省', '邯郸'])
'''
'''
filepath = '../../../网课课件/一、python/第八章 pandas进阶/lesson_06/examples/datasets/2016_happiness.csv'
data = pd.read_csv(filepath, usecols=['Country', 'Region', 'Happiness Rank', 'Happiness Score'],
                   engine='python', encoding='utf-8')

def f(x):
    if x <= 4:
        y = 'low'
    elif x <= 6:
        y = 'middle'
    else:
        y = 'high'
    return y

# data2 = data.set_index('Happiness Score').groupby(f).size()
data['score group'] = data['Happiness Score'].apply(f)
data2 = data.groupby('Region').agg({'Happiness Score':[np.mean, np.max], 'Happiness Rank': np.max})
print(data2)
'''
'''
d = {
    'Name': ['Alisa', 'Bobby', 'Cathrine', 'Alisa', 'Bobby', 'Cathrine',
             'Alisa', 'Bobby', 'Cathrine', 'Alisa', 'Bobby', 'Cathrine'],
    'Semester': ['Semester 1', 'Semester 1', 'Semester 1', 'Semester 1', 'Semester 1', 'Semester 1',
                 'Semester 2', 'Semester 2', 'Semester 2', 'Semester 2', 'Semester 2', 'Semester 2'],
    'Subject': ['Mathematics', 'Mathematics', 'Mathematics', 'Science', 'Science', 'Science',
                'Mathematics', 'Mathematics', 'Mathematics', 'Science', 'Science', 'Science'],
    'Score': [62, 47, 55, 74, 31, 77, 85, 63, 42, 67, 89, 81]}

df = pd.DataFrame(d)
df1 = df.pivot_table(values='Score', columns='Subject', index='Semester', margins=True)
print(df)
print(df['Score'])
'''
'''
df1 = pd.DataFrame({"score1":np.random.randint(1,5,size=5),
                   "score2":np.random.randint(5,10,size=5),
                   "score3":np.random.randint(10,15,size=5)},
                  index=[1,2,3,4,5])
df2 = pd.DataFrame({"score1":np.random.randint(1,5,size=3),
                   "score2":np.random.randint(5,10,size=3),
                   "score3":np.random.randint(10,15,size=3)},
                  index=[4,5,6])
df3 = pd.concat([df1,df2],axis=1)
print(df2.iloc[0, 2])
'''
'''
staff_df = pd.DataFrame([{'姓名': '张三', '部门': '研发部'},
                        {'姓名': '李四', '部门': '财务部'},
                        {'姓名': '赵六', '部门': '市场部'}])
student_df = pd.DataFrame([{'姓名': '张三', '专业': '计算机'},
                        {'姓名': '李四', '专业': '会计'},
                        {'姓名': '王五', '专业': '市场营销'}])
staff_df['地址'] = ['天津', '北京', '上海']
student_df['地址'] = ['天津', '上海', '广州']
staff_df.set_index('姓名', inplace=True)
student_df.set_index('姓名', inplace=True)

df = pd.merge(staff_df, student_df, how='outer', left_index=True, right_index=True)
print(df)
'''
'''
header = pd.MultiIndex.from_product([['python','机器学习'],['初级','高阶']])
df = pd.DataFrame(np.random.randint(0,150,size=(4,4)),
               index = ['第一季度','第二季度','第三季度','第四季度'],
               columns=header)
stacked_df=df.stack()
unstacked_df = stacked_df.unstack(level=0)
print(unstacked_df)
'''
arr = [[1.9,2.5],[1.6,7.3]]
arr1 = np.ceil(arr)
print(arr1)

