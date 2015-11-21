from mpl_toolkits.mplot3d.axes3d import Axes3D
import numpy
import numpy as np



def plot_raw_y(data_y,fig_name):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
    color=['red', 'green']
    plt.gca().set_color_cycle(['red'])
    legends=[]
    legends.append("GT data")

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    x =data_y[:,0]
    y =data_y[:,1]
    z =data_y[:,2]
    ax.plot(x, y, z, color[0])

    ax.legend(legends,loc='upper left')
    plt.savefig("predictions/"+fig_name)
    plt.show()


def plot_y(data_y,fig_name):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
    color=['red', 'green']
    plt.gca().set_color_cycle(['red', 'green'])
    legends=[]
    legends.append("GT data")
    legends.append("Est data")

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    i=0
    for d in data_y:
        x =d[:,0]
        y =d[:,1]
        z =d[:,2]
        ax.plot(x, y, z, color[i])
        i +=1


    ax.legend(legends,loc='upper left')
    plt.savefig("predictions/"+fig_name)
    plt.show()



def plot_err(err,fig_name):
    import matplotlib.pyplot as plt
    f, (ax1, ax2, ax3) = plt.subplots(3, sharex=True, sharey=True)
    fr = range(len(err))
    x = err[:, 0]
    ax1.plot(fr, x)
    plt.grid()

    y = err[:, 1]
    ax2.plot(fr, y)
    plt.grid()

    z = err[:, 2]
    ax3.plot(fr, z)
    plt.grid()
    #f.subplots_adjust(hspace=0)
    plt.savefig("predictions/"+fig_name)
    #plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)
    plt.show()

