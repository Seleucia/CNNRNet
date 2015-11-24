import dataset_loader as dl
import numpy
import theano
import theano.tensor as T
from PIL import Image
import os
import glob
from sklearn.cross_validation import train_test_split

def load_tum_data():
    def shared_dataset(data_x, data_y, borrow=True):
            """ Function that loads the dataset into shared variables

            The reason we store our dataset in shared variables is to allow
            Theano to copy it into the GPU memory (when code is run on GPU).
            Since copying data into the GPU is slow, copying a minibatch everytime
            is needed (the default behaviour if the data is not in a shared
            variable) would lead to a large decrease in performance.
            """
            shared_x = theano.shared(numpy.asarray(data_x,
                                                   dtype=theano.config.floatX),
                                     borrow=borrow)
            shared_y = theano.shared(numpy.asarray(data_y,
                                                   dtype=theano.config.floatX),
                                     borrow=borrow)
            # When storing data on the GPU it has to be stored as floats
            # therefore we will store the labels as ``floatX`` as well
            # (``shared_y`` does exactly that). But during our computations
            # we need them as ints (we use labels as index, and if they are
            # floats it doesn't make sense) therefore instead of returning
            # ``shared_y`` we will have to cast it to int. This little hack
            # lets ous get around this issue
            return shared_x, shared_y

    def read_file_list(filename):
        """
        Reads a trajectory from a text file.

        File format:
        The file format is "stamp d1 d2 d3 ...", where stamp denotes the time stamp (to be matched)
        and "d1 d2 d3.." is arbitary data (e.g., a 3D position and 3D orientation) associated to this timestamp.

        Input:
        filename -- File name

        Output:
        dict -- dictionary of (stamp,data) tuples

        """
        file = open(filename)
        data = file.read()
        lines = data.replace(","," ").replace("\t"," ").split("\n")
        list = [[v.strip() for v in line.split(" ") if v.strip()!=""] for line in lines if len(line)>0 and line[0]!="#"]
        list = [(float(l[0]),l[1:]) for l in list if len(l)>1]
        return dict(list)
    def associate(first_list, second_list,offset,max_difference):
        """
        Associate two dictionaries of (stamp,data). As the time stamps never match exactly, we aim
        to find the closest match for every input tuple.

        Input:
        first_list -- first dictionary of (stamp,data) tuples
        second_list -- second dictionary of (stamp,data) tuples
        offset -- time offset between both dictionaries (e.g., to model the delay between the sensors)
        max_difference -- search radius for candidate generation

        Output:
        matches -- list of matched tuples ((stamp1,data1),(stamp2,data2))

        """
        first_keys = first_list
        second_keys = second_list
        potential_matches = [(abs(a - (b + offset)), a, b)
                             for a in first_keys
                             for b in second_keys
                             if abs(a - (b + offset)) < max_difference]
        potential_matches.sort()
        matches = []
        for diff, a, b in potential_matches:
            if a in first_keys and b in second_keys:
                first_keys.remove(a)
                second_keys.remove(b)
                matches.append((a, b))

        matches.sort()
        matches=numpy.array(matches)
        return matches

    dataset='/home/coskun/PycharmProjects/TheanoExamples/data/rgbd_dataset_freiburg3_large_cabinet'
    ''' Loads the dataset

    :type dataset: string
    :param dataset: the path to the dataset (here TUM)
    '''

    size = 32, 24
    offset=0
    max_difference=0.2


    dir_f=dataset+'/depth/'
    full_path=dataset+'/depth/*.png'
    lst=glob.glob(full_path)
    n_lst = [l.replace(dir_f, '') for l in lst]
    lst = [l.replace('.png', '') for l in n_lst]
    first_list=[float(i) for i in lst]
    filename='/home/coskun/PycharmProjects/TheanoExamples/data/rgbd_dataset_freiburg3_large_cabinet/groundtruth.txt';
    second_list=read_file_list(filename)

    #Find closes trajectry for depth image
    matches=associate(first_list, second_list.keys(),offset,max_difference)

    i=0
    m_size=(len(matches),size[0]*size[1])
    data_x=numpy.empty(m_size,float)
    data_y=numpy.matrix([[float(value) for value in second_list[b][0:1]] for a,b in matches])

    dir_list=[["%s%f%s" %(dir_f,a,".png")] for a,b in matches]

    for dImg in dir_list:
        img = Image.open(dImg[0])
        img.thumbnail(size, Image.ANTIALIAS)
        arr = numpy.array(img) # 640x480x4 array
        v= numpy.transpose(numpy.reshape(arr,(size[0]*size[1],1)))
        data_x[i]=v
        i=i+1;


    X_train, X_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.10, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.20, random_state=42)


    train_set_x, train_set_y = shared_dataset(X_train,y_train)
    test_set_x, test_set_y = shared_dataset(X_test,y_test)
    valid_set_x, valid_set_y = shared_dataset(X_val,y_val)

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    return rval


datasets = load_tum_data()
train_set_x, train_set_y = datasets[0]
valid_set_x, valid_set_y = datasets[1]
test_set_x, test_set_y = datasets[2]

print train_set_x.shape.eval()
print valid_set_x.shape.eval()
print test_set_x.shape.eval()

print train_set_y.shape.eval()
print valid_set_y.shape.eval()
print test_set_y.shape.eval()