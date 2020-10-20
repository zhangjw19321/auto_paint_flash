# 1、导入模块
import os
import paddlehub as hub

# 2、加载模型
humanseg = hub.Module(name='deeplabv3p_xception65_humanseg')

# 3、获取文件列表
# 图片文件的目录
path = './imgs/'
# 获取目录下的文件
files = os.listdir(path)
# 用来装图片的
imgs = []
# 拼接图片路径
for i in files:
    imgs.append(path + i)
    print(imgs)
#抠图
results = humanseg.segmentation(data={'image':imgs})
