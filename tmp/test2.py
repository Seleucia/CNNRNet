import matplotlib.pyplot as plt
import numpy

from helper import model_saver

multi=1
step_size=1
rn_id=10;
#dataset="/home/coskun/PycharmProjects/data/rgbd_dataset_freiburg3_cabinet_validation/"
dataset="/home/coskun/PycharmProjects/data/rgbd_dataset_freiburg3_large_cabinet/"
#datasets = dataset_loader.load_tum_data_valid(dataset,step_size,multi)
#datasets =  dataset_loader.load_tum_dataV2(dataset,rn_id,multi)

params = model_saver.load_model("1_2_best_model.pkl")

data_x=datasets[0]
delta_y10=datasets[1]
data_y=datasets[2]
overlaps=datasets[3]
delta_yy=[]
for index in range(len(overlaps)):
    value= sum(delta_y10[overlaps[index]]) / (step_size * len(overlaps[index]))
    delta_yy.append(value)
    #print value

delta_y= (numpy.vstack([[0, 0, 0], data_y]) - numpy.vstack([data_y, [0, 0, 0]]))[1:-1, :]


x =numpy.squeeze(numpy.asarray(data_y[:,0]))
y =numpy.squeeze(numpy.asarray(data_y[:,1]))
z =numpy.squeeze(numpy.asarray(data_y[:,2]))



fig = plt.figure()
ax = plt.gca(projection='3d')

#ax = fig.add_subplot(111, projection='3d')
#ax.scatter(x, y, z, c='r', marker='o')

ax.plot(x, y, z, '-b')

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
ax.legend()
plt.show()


print(numpy.mean(abs(delta_y-delta_yy)))
delta_yy=numpy.cumsum(delta_yy, axis=0)
delta_yy=numpy.insert(delta_yy, 0, data_y[0], 0)
diff= data_y - delta_yy
mn=numpy.mean(numpy.abs(diff))
print("okkk")