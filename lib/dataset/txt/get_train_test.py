import time,random
f=open('set_data.txt').readlines()
f2=open('train.txt','a',encoding='utf8')

for x in f:
    name=r'train/'+x.split( )[0]
    f2.write(name+'\t'+x.split( )[1]+'\n')



