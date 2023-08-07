'''
1
Long-tail 10 groups
'''
import numpy as np
import matplotlib.pyplot as plt

X_values = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
X1 = np.array([0.0007955,	0.0007856,	0.0009892,	0.00188,	0.002166,	0.002793,	0.004088,	0.005902,	0.0104,	0.04098])
X2 = np.array([0.0006211,	0.0006633,	0.0008189,	0.00173,	0.001778,	0.002484,	0.003673,	0.005282,	0.009737,	0.03884])
X3 = np.array([0.0004855,	0.0005537,	0.0007032,	0.001668,	0.00163,	0.00244,	0.003762,	0.005663,	0.01045,	0.04103])
X4 = np.array([0.000832,	0.0008482,	0.001053,	0.00194,	0.0024,	0.002971,	0.004306,	0.006176,	0.01072,	0.03995])
X5 = np.array([0.00196,	0.001831,	0.001883,	0.002763,	0.002951,	0.003618,	0.004869,	0.006146,	0.009821,	0.03736])
X6 = np.array([0.001907,	0.001831,	0.001915,	0.00299,	0.003031,	0.004006,	0.005376,	0.006966,	0.01101,	0.03631])

def groups(X_values, X1, X2, X3, X4, X5, X6):

    # 设置图形大小和DPI以获得更高的分辨率
    figsize = (8, 6)
    dpi = 500

    # 绘制图形
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    plt.subplots_adjust(left=0.112, bottom=0.1, right=0.95, top=0.95)

    bar_width = 0.11
    bar_positions_X1 = X_values - 2.5*bar_width
    bar_positions_X2 = X_values -1.5*bar_width
    bar_positions_X3 = X_values - 0.5*bar_width
    bar_positions_X4 = X_values + 0.5*bar_width
    bar_positions_X5 = X_values + 1.5*bar_width
    bar_positions_X6 = X_values + 2.5*bar_width
    '''
    red:#DB7B6E
    orange:#FFB265
    yellow:#FFF2CC
    green:#D1ECB9
    blue:#6DC3F2
    purple:#A68BC2
    '''
    ax.bar(bar_positions_X1, X1, width=bar_width, color='#DB7B6E', label='LightGCN', edgecolor='black', linewidth=0.5)
    ax.bar(bar_positions_X2, X2, width=bar_width, color='#FFB265', label='SGL-RW', edgecolor='black', linewidth=0.5)
    ax.bar(bar_positions_X3, X3, width=bar_width, color='#FFF2CC', label='SimGCL', edgecolor='black', linewidth=0.5)
    ax.bar(bar_positions_X4, X4, width=bar_width, color='#D1ECB9', label='PDA', edgecolor='black', linewidth=0.5)
    ax.bar(bar_positions_X5, X5, width=bar_width, color='#6DC3F2', label='BC loss', edgecolor='black', linewidth=0.5)
    ax.bar(bar_positions_X6, X6, width=bar_width, color='#A68BC2', label='Ours', edgecolor='black', linewidth=0.5)


    # 添加标签和标题
    plt.xlabel('Group ID', fontsize=16, fontweight='bold')
    plt.ylabel('$Recall^{(g)}@20$', fontsize=16, fontweight='bold')
    plt.title('Last-FM', fontsize=16)
    plt.xticks(X_values)
    plt.legend(loc='upper left', fontsize=16)

    # 设置坐标轴刻度字体大小和加粗
    plt.tick_params(axis='both', labelsize=12, width=2, length=6, labelcolor='black')

    # 增加横向的网格参考线
    plt.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.7)

    # 显示图形
    plt.savefig('group-last-fm.jpg')

# groups(X_values, X1, X2, X3, X4, X5, X6)









'''
2
hyperparameters  折线图
双纵轴：Overall和Long-Tail
'''
# [0., 0.01, 0.05, 0.08, 0.1, 0.12, 0.15, 0.2, 0.5, 1.]
# [0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.13, 0.14, 0.15]
X_values = [0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.13, 0.14, 0.15]
Y_values_1 = [0.05077,	0.05477,	0.05647,	0.05737,	0.05679,	0.05525,	0.05364,	0.05205,	0.04993,	0.04815]
Y_values_2 = [0.003105,	0.003723,	0.00441,	0.004503,	0.004367,	0.003754,	0.003434,	0.003003,	0.002643,	0.002334]

def double_y(X_values, Y_values_1,Y_values_2):
    # 设置图形大小和DPI以获得更高的分辨率
    figsize = (8, 6)
    dpi = 500

    fig, ax1 = plt.subplots(figsize=figsize, dpi=dpi)
    plt.subplots_adjust(left=0.2, bottom=0.05, right=0.8, top=0.93)

    # title = ('Tune $\lambda\'$ on Amazon-Book')
    title = ('Tune $\\tau$ on Amazon-Book')
    plt.title(title,fontsize=20)
    plt.grid(axis='y',color='grey',linestyle='--',lw=0.5,alpha=0.5)
    plt.tick_params(axis='both',labelsize=16)
    plot1 = ax1.plot(X_values, Y_values_1, color='#DB7B6E', label='$Recall@20$', marker='o', linestyle='-')
    ax1.set_ylabel('Overall  $Recall@20$', fontsize = 18, color='black', labelpad=10)
    # ax1.set_ylim(0,240)
    for tl in ax1.get_yticklabels():
        tl.set_color('#DB7B6E')    
    ax2 = ax1.twinx()
    plot2 = ax2.plot(X_values, Y_values_2, color='#A68BC2', label='$Recall^{(1)}@20$', marker='s', linestyle='--')
    ax2.set_ylabel('Long-tail  $Recall^{(1)}@20$',fontsize=18, color='black', labelpad=10)
    # ax2.set_ylim(0,0.08)
    ax2.tick_params(axis='y',labelsize=16)
    for tl in ax2.get_yticklabels():
        tl.set_color('#A68BC2')                    
    # ax2.set_xlim(1966,2014.15)
    lines = plot1 + plot2           
    ax1.legend(lines,[l.get_label() for l in lines], loc='lower center', fontsize=16)    
    ax1.set_yticks(np.linspace(ax1.get_ybound()[0],ax1.get_ybound()[1],9)) 
    ax2.set_yticks(np.linspace(ax2.get_ybound()[0],ax2.get_ybound()[1],9)) 
    for ax in [ax1,ax2]:
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)          

    # 显示图形
    plt.savefig('tau-amazon-book.jpg')


double_y(X_values, Y_values_1,Y_values_2)



'''
3
hyperparameters  折线图
双纵轴：Overall和Long-Tail
'''
Tau_values = [0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.13, 0.14, 0.15]
Lambda_values = [0., 0.01, 0.05, 0.08, 0.1, 0.12, 0.15, 0.2, 0.5]
Y_values_tau = [0.07347,	0.07822,	0.08378,	0.08915,	0.09433,	0.09885,	0.1029,	0.1063,	0.1092,	0.112]
Y_values_lambda = [0.09352,	0.09441,	0.09514,	0.09467,	0.0946,	0.09409,	0.09328,	0.09271,	0.09375]
BC_tau = [0.0992, 0.0992, 0.0992, 0.0992, 0.0992, 0.0992, 0.0992, 0.0992, 0.0992, 0.0992]
