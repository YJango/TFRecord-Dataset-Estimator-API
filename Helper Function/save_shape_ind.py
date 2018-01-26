from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
import numpy as np
import tensorflow as tf

np.set_printoptions(precision=3)

def display(alist, show = True):
    print('type:%s\nshape: %s' %(alist[0].dtype,alist[0].shape))
    if show:
        for i in range(3):
            print(alist[i])

# create a list of 3 scalars
scalars = np.array([1,2,3],dtype=int64)
display(scalars)

# create a list of 3 vectors
vectors = np.array([[0.1,0.1,0.1],
                   [0.2,0.2,0.2],
                   [0.3,0.3,0.3]],dtype=float32)
display(vectors)

# create a list of 3 matrices
matrices = np.array([np.array((vectors[0],vectors[0])),
                    np.array((vectors[1],vectors[1])),
                    np.array((vectors[2],vectors[2]))],dtype=float32)
display(matrices)

# shape of image：(806,806,3)
img=mpimg.imread('YJango.jpg')
img=mpimg.imread('YJango.jpg')
img=mpimg.imread('YJango.jpg')
tensors = np.array([img,img,img])
# show image
display(tensors, show = False)
plt.imshow(img)




def feature_writer(features, name, value, isbyte=False):
	'''
	Writes a single feature in features whose type is dictionary
	Args:
		features: a dictionary containing all features of one example
		name: the name of the feature to be written
		value: an array : the value of the feature to be written
		toString: whether to convert to string
	
	Returns:
		features: the dictionary that has been added one feature 
	
	Note:
		the tfrecord type will be as same as the numpy dtype
		if the feature's rank >= 2, the shape (type: int64) will also be added in features
		e.g. the names of shape info of matrix input are: 
			name+'_shape_0'
			name+'_shape_1'
	Raises:
		TypeError: Type is not one of ('int64', 'float32')
	'''
	# get the corresponding type function
	if isbyte:
		feature_typer = lambda value : tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.tostring()]))
	else:
		if value.dtype == np.int64:
			feature_typer = lambda value : tf.train.Feature(int64_list=tf.train.Int64List(value=value))
		elif value.dtype == np.float32:
			feature_typer = lambda value : tf.train.Feature(float_list = tf.train.FloatList(value=value))
		else:
			raise TypeError("Type is not one of 'int64', 'float32'")
	# check whether the input is (1D-array)
	# if the input is a scalar, convert it to list
	if len(value.shape)==0:
		features[name] = feature_typer([value])
	elif len(value.shape)==1:
		features[name] = feature_typer(value)
	# if # if the rank of input array >=2, flatten the input and save shape info
	elif len(value.shape) >1:
		features[name] = feature_typer(value.reshape(-1))
		# write shape info
		rank = 0
		for dim in value.shape[:-1]:
			features['%s_shape_%s' %(name,rank)] = tf.train.Feature(int64_list=tf.train.Int64List(value=[dim]))
			rank += 1
	return features
  
  
writer = tf.python_io.TFRecordWriter('%s.tfrecord' %'test')
for i in range(0,3):
    features={}
    feature_writer(features, 'scalar', scalars[i], 0)
    feature_writer(features, 'vector', vectors[i], 1)
    feature_writer(features, 'matrix', img, 1)
    tf_features = tf.train.Features(feature= features)
    tf_example = tf.train.Example(features = tf_features)
    tf_serialized = tf_example.SerializeToString()
    writer.write(tf_serialized)
writer.close()
  
def create_parser(df):
    
    names = df['name']
    types = df['type']
    shapes = df['shape']
    isbytes = df['isbyte']
    defaults = df['default']
    length_types = df['length_type']
    
    def parser(example_proto):
        
        def specify_features():
            specified_features = {}
            for i in np.arange(len(names)):
                # which type
                if isbytes[i]:
                    t = tf.string
                    s = ()
                else:
                    t = types[i]
                    s = shapes[i]
                # has default_value?
                if defaults[i] == np.NaN:
                    d = np.NaN
                else:
                    d = defaults[i]
                # length varies
                if length_types[i] =='fixed':
                    specified_features[names[i]] = tf.FixedLenFeature(s, t)
                elif length_types[i] =='var':
                    specified_features[names[i]] = tf.VarLenFeature(t)
                else:
                    raise TypeError("length_type is not one of 'var', 'fixed'")
            return specified_features
        
            
        # decode each parsed feature and reshape
        def decode_reshape():
            # store all decoded&shaped features
            final_features = {}
            for i in np.arange(len(names)):
                # exclude shape info
                if '_shape_' not in names[i]:
                    # decode
                    if isbytes[i]:
                        # from byte format
                        decoded_value = tf.decode_raw(parsed_example[names[i]], types[i])
                    else:
                        # Varlen value needs to be converted to dense format
                        if length_types[i] == 'var':
                            decoded_value = tf.sparse_tensor_to_dense(parsed_example[names[i]])
                        else:
                            decoded_value = parsed_example[names[i]]
                    # reshape
                    shape_values = [parsed_example[key] for key in parsed_example.keys() if '%s_shape_' %names[i] in key]
                    if len(shape_values)>0:
                        tf_shape = tf.stack(shape_values+[-1])
                        decoded_value = tf.reshape(decoded_value, tf_shape)
                    final_features[names[i]] = decoded_value
            return final_features
        
        # create a dictionary to specify how to parse each feature 
        specified_features = specify_features()
        # parse all features of an example
        parsed_example = tf.parse_single_example(example_proto, specified_features)
        final_features = decode_reshape()
        return final_features
    return parser
    
N = ["scalar","vector","matrix","matrix_shape_0"]
T = [tf.int64,tf.float32,tf.float32,tf.int64]
S = [(),(3,),(2,3),()]
I = [False,False,False,False]
L = ['fixed','fixed','fixed','fixed']

data_info = pd.DataFrame({'name':N,
             'type':T,
             'shape':S,
             'isbyte':I,
             'length_type':L},columns=['name','type','shape','isbyte','default','length_type'])
             
filenames = ["test.tfrecord"]
dataset = tf.data.TFRecordDataset(filenames)
_parse_function = create_parser(data_info)
dataset = dataset.map(_parse_function)
dataset = dataset.shuffle(buffer_size=10000)
#dataset = dataset.batch(6)
dataset = dataset.repeat(2)
#iterator = dataset.make_initializable_iterator()
iterator = dataset.make_one_shot_iterator()
next_element = iterator.get_next()
next_exmaple=next_element
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
