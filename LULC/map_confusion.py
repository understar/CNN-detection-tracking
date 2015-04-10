import numpy as np
import matplotlib.pyplot as plt

def norm_it(conf_arr):
    conf_arr = conf_arr.tolist()
    norm_conf = []
    for i in conf_arr:
        a = 0
        tmp_arr = []
        a = sum(i, 0)
        for j in i:
            tmp_arr.append(float(j)/float(a))
        norm_conf.append(tmp_arr)
    return norm_conf

def plot_conf(conf_arr, label_list, save_name='confusion_matrix.png'):
    norm_conf = norm_it(conf_arr)
    fig = plt.figure()
    plt.clf()
    ax = fig.add_subplot(111)
    ax.set_aspect(1)
    res = ax.imshow(np.array(norm_conf), cmap=plt.cm.Reds, 
                    interpolation='nearest')
    
    width = len(conf_arr)
    height = len(conf_arr[0])
    
    for x in xrange(width):
        for y in xrange(height):
            if norm_conf[x][y] != 0:
                ax.annotate("{:3.2f}".format(norm_conf[x][y]), xy=(y, x), 
                            horizontalalignment='center',
                            verticalalignment='center')
            #else:
            #    ax.annotate("0", xy=(y, x), 
            #                horizontalalignment='center',
            #                verticalalignment='center')
                            
    fig.colorbar(res)
    alphabet = label_list
    plt.xticks(range(width), alphabet[:width])
    plt.yticks(range(height), alphabet[:height])
    plt.savefig(save_name, format='png')
    
if __name__ == '__main__':
    # confusion matrix sample
    conf_arr = [[33,2,0,0,0,0,0,0,0,1,3], 
                [3,31,0,0,0,0,0,0,0,0,0], 
                [0,4,41,0,0,0,0,0,0,0,1], 
                [0,1,0,30,0,6,0,0,0,0,1], 
                [0,0,0,0,38,10,0,0,0,0,0], 
                [0,0,0,3,1,39,0,0,0,0,4], 
                [0,2,2,0,4,1,31,0,0,0,2],
                [0,1,0,0,0,0,0,36,0,2,0], 
                [0,0,0,0,0,0,1,5,37,5,1], 
                [3,0,0,0,0,0,0,0,0,39,0], 
                [0,0,0,0,0,0,0,0,0,0,38]]
    conf_arr = np.array(conf_arr)
    plot_conf(conf_arr, range(26))