import time,random
start =  time.process_time()
s=''
f3=open('char_test.txt','a',encoding='utf8')
f2=open('char_train.txt','a',encoding='utf8')
f=open('set_data.txt').readlines()
with open('char_std_5990.txt', 'rb') as file:
    char_dict = {num: char.strip().decode() for num, char in enumerate(file.readlines())}


for x in f:
    labels=x.split( )[1:][0]
    labels_num=[]
    for b in labels:
        char_num = list(char_dict.values()).index(b)
        labels_num.append(char_num)
        s=s+b
    labels_num_str=' '.join(str(item) for item in labels_num)
    f2.write(x.split( )[0]+' '+labels_num_str+'\n')



f2=open('char_train.txt','r',encoding='utf8')
raw_list = f2.readlines()
random.shuffle(raw_list)
for i in range(int(len(raw_list)/4)):  # 随机抽取数目 n
    f3.writelines(raw_list[i])

end = time.process_time()
print("cost time is %f" % (end - start))
print(s)