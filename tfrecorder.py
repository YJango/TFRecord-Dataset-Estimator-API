import pandas as pd
import numpy as np
import tensorflow as tf
import os

class TFrecorder(object):
    
    '''
    helper function for write and read TFrecord files
    
    Args:
        path: where to write and read
        
    '''
    def __init__(self, length_type ='fixed',retrieve_shape = False):
        '''
        Args: length_type: string, indicating how to parse the data
           should be one of 'var' or 'fixed'
        '''
        self.length_type = length_type
        self.retrieve_shape = retrieve_shape
        
    def feature_writer(self, df, value, features=None):
        '''
        Writes a single feature in features
        Args:
            value: an array : the value of the feature to be written
        
        Note:
            the tfrecord type will be as same as the numpy dtype
            if the feature's rank >= 2, the shape (type: int64) will also be added in features
            the name of shape info is: name+'_shape'
        Raises:
            TypeError: Type is not one of ('int64', 'float32')
        '''
        if features is None:
            features = self.features
        name = df['name']
        isbyte = df['isbyte']
        length_type = df['length_type']
        default = df['default']
        dtype = df['type']
        shape = df['shape']
        
        # get the corresponding type function
        if isbyte:
            feature_typer = lambda x : tf.train.Feature(bytes_list = tf.train.BytesList(value=[x.tostring()]))
        else:
            if dtype == 'int64' or dtype == np.int64:
                feature_typer = lambda x : tf.train.Feature(int64_list = tf.train.Int64List(value=x))
            elif dtype == 'float32' or dtype == np.float32:
                feature_typer = lambda x : tf.train.Feature(float_list = tf.train.FloatList(value=x))
            else:
                raise TypeError("Type is not one of 'np.int64', 'np.float32'")
        # check whether the input is (1D-array)
        # if the input is a scalar, convert it to list
        if len(shape)==0 and isbyte==False:
            features[name] = feature_typer([value])
        elif len(shape)==1:
            features[name] = feature_typer(value)
        # if # if the rank of input array >=2, flatten the input and save shape info
        elif len(shape) >1:
            features[name] = feature_typer(value.reshape(-1))
            # write shape info
            features['%s_shape' %name] = tf.train.Feature(int64_list=tf.train.Int64List(value=shape))
        return features

    def data_info_fn(self, one_example):
    
        data_info = pd.DataFrame(columns=['name','type','shape','isbyte','length_type','default'])
        i = 0
        for key in one_example:
            value = one_example[key]
            dtype = value.dtype
            shape = value.shape
            if len(shape)>1:
                data_info.loc[i] = {'name':key,
                                    'type':dtype,
                                    'shape':shape,
                                    'isbyte':True,
                                    'length_type': 'fixed',
                                    'default':np.NaN}
                i+=1
                data_info.loc[i] = {'name':key+'_shape',
                                    'type':'int64',
                                    'shape':(len(shape),),
                                    'isbyte':False,
                                    'length_type':'fixed',
                                    'default':np.NaN}
                i+=1
            else:
                data_info.loc[i] = {'name':key,
                                    'type':dtype,
                                    'shape':shape,
                                    'isbyte':False,
                                    'length_type': self.length_type,
                                    'default':np.NaN}
                i+=1
        return data_info
        
    def writer(self, path, examples, data_info = None, num_examples_per_file = None):
        if data_info is None:
            self.data_info = self.data_info_fn(examples[0])
        else:
            self.data_info = data_info
        self.path = path

        self.num_example = len(examples)
        self.num_feature = len(examples[0])
        if num_examples_per_file is not None:
            num_so_far = 0
            self.num_examples_per_file = num_examples_per_file
            writer = tf.python_io.TFRecordWriter('%s%s_%s.tfrecord' %(self.path, num_so_far, self.num_examples_per_file))
            print('%s%s_%s.tfrecord' %(self.path, num_so_far, self.num_examples_per_file))
        else:
            if '.tfrecord' not in self.path:
                self.path = self.path+'.tfrecord'
            writer = tf.python_io.TFRecordWriter('%s' %self.path)
            print('????')
        for e in np.arange(self.num_example):
            self.features={}
            for f in np.arange(self.num_feature):
                feature_name = self.data_info.loc[f]['name']
                if '_shape' not in feature_name:
                    self.feature_writer(self.data_info.loc[f], examples[e][feature_name])
            tf_features = tf.train.Features(feature= self.features)
            tf_example = tf.train.Example(features = tf_features)
            tf_serialized = tf_example.SerializeToString()
            writer.write(tf_serialized)
            if num_examples_per_file is not None:
                if e%num_examples_per_file ==0 and e!=0:
                    writer.close()
                    num_so_far = e
                    writer = tf.python_io.TFRecordWriter('%s%s_%s.tfrecord' %(self.path, num_so_far, e+self.num_examples_per_file))
                    print('%s%s_%s.tfrecord' %(self.path, num_so_far, e+self.num_examples_per_file))
        writer.close()
        self.data_csv = self.path.split('.tfrecor')[0]+'.csv'
        self.data_info.to_csv(self.data_csv,index=False)
        print('number of features in each example: %s' %self.num_feature)
        print('%s examples has been written to %s' %(self.num_example,self.path))
        print('saved data_info to %s' %self.data_csv)
        print(self.data_info)
        
    def create_parser(self, data_info, reshape):
        
        names = data_info['name']
        types = data_info['type']
        shapes = data_info['shape']
        isbytes = data_info['isbyte']
        defaults = data_info['default']
        length_types = data_info['length_type']
        
        if reshape is None:
            reshape = {}
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
                    if '_shape' not in names[i]:
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
                        if '%s_shape' %names[i] in parsed_example.keys():
                            tf_shape = parsed_example['%s_shape' %names[i]]
                            decoded_value = tf.reshape(decoded_value, tf_shape)
                        elif names[i] in reshape.keys():
                            if len(reshape[names[i]])>0:
                                decoded_value = tf.reshape(decoded_value, reshape[names[i]])
                        final_features[names[i]] = decoded_value
                    elif self.retrieve_shape:
                        final_features[names[i]] = parsed_example[names[i]]
                return final_features
            
            
            # create a dictionary to specify how to parse each feature 
            specified_features = specify_features()
            # parse all features of an example
            parsed_example = tf.parse_single_example(example_proto, specified_features)
            final_features = decode_reshape()
            
            
            
            return final_features
        return parser
    def get_filenames(self, path,shuffle=False):
        # get all file names 
        files= os.listdir(path) 
        filepaths = [path+file for file in files if not os.path.isdir(file) and '.tfrecord' in file]
        # shuffle
        if shuffle:
            ri = np.random.permutation(len(filepaths))
            filepaths = np.array(filepaths)[ri]
        return filepaths
    def get_dataset(self, paths, data_info, shuffle = True, shuffle_buffer=10000, batch_size = 1, epoch = 1, padding = None, reshape = None, num_parallel_calls=12, prefetch_buffer=1000):
        
        self.filenames = paths
            
        print('read dataframe from %s x %s' %(paths[0],len(paths)))
        data_info = pd.read_csv(data_info,dtype={'isbyte':bool})
        data_info['shape']=data_info['shape'].apply(lambda s: [int(i) for i in s[1:-1].split(',') if i !=''])
            
        self.shuffle = shuffle
        self.shuffle_buffer = shuffle_buffer
        self.batch_size = batch_size
        self.epoch = epoch
        self.padding = padding
        #print(data_info)
        dataset = tf.data.TFRecordDataset(self.filenames)
       
        self.parse_function = self.create_parser(data_info, reshape)
        self.dataset = dataset.map(self.parse_function, num_parallel_calls=None).prefetch(prefetch_buffer)
        self.dataset_raw = self.dataset
        if self.shuffle:
            self.dataset = self.dataset.shuffle(buffer_size=self.shuffle_buffer)
        if self.padding is not None:
            self.dataset = self.dataset.padded_batch(self.batch_size, padded_shapes = self.padding)
        else:
            self.dataset = self.dataset.batch(self.batch_size)
        self.dataset = self.dataset.repeat(self.epoch)
        return self.dataset
