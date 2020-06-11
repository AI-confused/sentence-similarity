import pandas as pd

#合并数据集
train0 = pd.read_csv('atec_nlp_sim_train.csv',sep='\t')
train1 = pd.read_csv('atec_nlp_sim_train_add.csv',sep='\t')
data = pd.concat([train0, train1], axis=0)
del data['id']
ids = []
for i in range(len(data)):
    ids.append(100000+i)
data['id'] = ids
data[['id','query0','query1','label']].to_csv('all_data.csv',header=True,index=False)
print('all_data.csv done!')
#拆分数据集
data = pd.read_csv('all_data.csv')
test = data.sample(n=int(len(data)*0.1), random_state=1)
test.to_csv('test.csv', index=False,header=True)
print('test.csv done!')
remain_data = data.drop(index=test.index)
dev = remain_data.sample(n=int(len(data)*0.1), random_state=1)
dev.to_csv('dev.csv',index=False,header=True)
print('dev.csv done!')
train = remain_data.drop(index=dev.index)
train.to_csv('train.csv', index=False,header=True)
print('train.csv done!')
assert len(train)+len(dev)+len(test) == len(data)
