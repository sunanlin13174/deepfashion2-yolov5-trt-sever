import os
labels_path = 'D:/temple_deepfashion2/labels/val2017'
labels_file = os.listdir(labels_path)
b=[]
for ele in labels_file:
    txt_path = os.path.join(labels_path,ele)
    with open(txt_path,'r') as f:
        for i in f:
           b = i.split(' ')[1:]
           print(b)
    with open(txt_path,'w') as g:
           g.write('{} {} {} {} {}'.format(13,b[0],b[1],b[2],b[3]))




