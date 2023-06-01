import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.patches import  ConnectionPatch

def draw_result(path):

    show_train_loss = [[], []]     # 用于显示的数据

    draw_epoch_train = np.loadtxt('%s/train_epoch.txt' % path)  # 读取绘图的周期点

    for i in range(2):
        show_train_loss[i] = np.loadtxt('%s/train_loss%d.txt' % (path, i+1))   # 读取txt文件，不同优化器的损失

    mpl.rc('font',family='Times New Roman', weight='semibold', size=9)  # 设置matplotlib中所有绘图风格的设置
    font1 = {'weight' : 'semibold', 'size' : 11}  #设置文字风格

    fig = plt.figure(figsize = (7,6))    #figsize是图片的大小`

    ax1 = fig.add_subplot(2, 1, 1)       # ax1是子图的名字
    ax1.plot(draw_epoch_train, show_train_loss[0],color = 'red', label = u'cml-1', linewidth =1.0, linestyle = 'dashed')
    ax1.plot(draw_epoch_train, show_train_loss[1],color = 'forestgreen', label = u'cml-2', linewidth =1.0, linestyle = 'dashed')
    ax1.legend(ncol=2)   #显示图例
    ax1.set_title('Training Loss', font1)
    ax1.set_xlabel(u'Epoch', font1)

    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.45) # hspace为子图上下间距
    plt.savefig('%s/Result.pdf' % (path), dpi=600)

if __name__ == '__main__':
    root = os.getcwd()
    my_path = os.path.join(root, "result")
    draw_result(my_path)