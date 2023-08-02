

def get_layer_name(model):
    names = []
    change_name =[]
    layer_num = []
    l=1
    for layer in model.layers:
        names.append(layer.name)
    for i in names:
        if i.find('conv2d')!=-1:
              change_name.append('conv2d')
              layer_num.append(l)
              l=l+1
        elif i.find('activation') !=-1:
              change_name.append('activation')
              layer_num.append(l)
              l=l+1
        elif i.find('batchnormalization')!=-1:
              change_name.append('dropout')
              layer_num.append(l)
              l=l+1
        elif i.find('max_pooling2d')!=-1:
              change_name.append('maxpooling2d')
              layer_num.append(l)
              l=l+1
        elif i.find('dropout')!=-1:
              change_name.append('dropout')
              layer_num.append(l)
              l=l+1
        elif i.find('dense')!=-1:
              change_name.append('dense')
              layer_num.append(l)
              l=l+1
        else:
              change_name.append(i)
              layer_num.append(l)
              l=l+1
    return names,change_name,layer_num

def get_layer_filters(model):
    fil = []
    fil_size =[]
    conv_count = 0
    filter_size = []
    conv_strides = []
    check = False
    count = False
    num_layer = []
    s = 3
    name = ''
    pool_fil_size =[]
    pool_num_layer=[]
    pool_strides=[]
    l=1
    for layer in model.layers:
        if layer.name.find('conv2d')!=-1 :
            fil.append(layer.filters) 
            num_layer.append(l)
            fil_size.append(layer.kernel_size)
            conv_strides.append(layer.strides)
            conv_count +=1
        if layer.name.find('max_pooling')!=-1:
              pool_fil_size.append(layer.pool_size)
              pool_num_layer.append(l)
              pool_strides.append(layer.strides)
        l+=1
    for i in range(0, len(fil)-1):
        if fil[i] <= fil[i+1]:
            check= True
        else:
            check =False
            name = num_layer[i+1]
            break
    for i in range(len(fil_size)):
          if fil_size[i][0] == fil_size[i][1]:
                 filter_size.append(fil_size[i][0])
    count = all(i==3 for i in filter_size)
    if count:
          s = 1
    elif filter_size == sorted(filter_size,reverse=True):
          s = 2        
    
    return check,filter_size,fil,num_layer,s,pool_fil_size, pool_num_layer,pool_strides,conv_strides
    
def dense_layer(model):
    dense_count = 0
    dense_units = []
    dense_num = []
    dc = 1
    for layer in model.layers:
        if layer.name.find('dense')!=-1 :
            dense_num.append(dc)
            dense_units.append(layer.units) 
            dense_count +=1
        dc+=1
    return dense_count, dense_units, dense_num
      
class Theia(Callback):
    """Callback that terminates training when bug is encountered.
    """
    def __init__(self, inputs,inputs_test, batch_size,problem_type,input_type):
        super(Theia, self).__init__()
        self.inputs = inputs
        self.inputs_test = inputs_test
        
        #self.classes = classes
        self.input_type = input_type
        self.problem_type = problem_type
        self.fil = []
        self.counter = 0
        self.count=0
        self.check = False
        self.batch_size = batch_size
        
    def on_train_begin(self,logs=None):
        start_time = time.time()
        layer_name, change_name, layer_number = get_layer_name(self.model)
        if change_name[0]=='conv1d':
            image_row = self.model.layers[0].input_shape[1]
            image_column = self.model.layers[0].input_shape[2]
            channel = 0
              
        if change_name[0]=='conv2d':
          image_row = self.model.layers[0].input_shape[1]
          image_column = self.model.layers[0].input_shape[2]
          channel = self.model.layers[0].input_shape[3]
        if change_name[0] == 'dense':
              image_row=28
              image_column = 28
              channel =1
        
        
        if 'Dense' in str(self.model.layers[-1]):
          classes = self.model.layers[-1].units
        elif 'Dense' in str(self.model.layers[-2]):
          classes = self.model.layers[-2].units  
        c=0
        
        layer_count = len(layer_name)
        dense_count, dense_units, dense_num = dense_layer(self.model)
        check,filter_size,fil,num_layer,s, pool_fil_size, pool_num_layer,pool_strides,conv_strides= get_layer_filters(self.model)
        
        
        message=[]
        mes=[]
        
      #Normalization of input
        if self.input_type == 1:  # 1 for colored
              min_train = np.min(self.inputs[0][0])
              max_train = np.max(self.inputs[0][0])
              min_test = np.min(self.inputs_test[0][0])
              max_test = np.max(self.inputs_test[0][0])
        elif self.input_type == 0:   #0 for grayscale
              min_train = self.inputs.min()
              max_train = self.inputs.max()
              min_test = self.inputs_test.min()
              max_test = self.inputs_test.max()

        if  min_train < 0.0 or max_train > 1.0 :
              mes.append('Normalize the training data' )
              c+=1
        if  min_test < 0.0 or max_test > 1.0:
              mes.append('Normalize the test data' )
              c+=1
         #.........................................................................................................................                  
        # Inadequate Batch size 
        input_c =str(type(self.inputs))
        print('inout_c is'+ input_c)
        if ('DirectoryIterator' or 'DataFrameIterator')  in input_c:
            sam = self.inputs.samples
        else:
            sam = len(self.inputs)
        print(sam)
        if sam  >=20000:
          if self.batch_size not in range(32,257): 
                if self.batch_size < 32:
                        mes.append('Increase the batch size --> preferred 32 or 64 or 128 or 256')
                        c+=1
                elif self.batch_size >256:
                      mes.append('Decrease the batch size --> preferred 32 or 64 or 128 or 256')
                      c+=1
        else:
          if self.batch_size not in range(32,64): 
                if self.batch_size < 32:
                        mes.append('Increase the batch size --> preferred 32 or 64')
                        c+=1
                elif self.batch_size >64:
                      mes.append('Decrease the batch size --> preferred 32 or 64')
                      c+=1
              
       #.........................................................................................................................                    
      #   #Check learning rate
        if 'adadelta' in str(self.model.optimizer):
              if self.model.optimizer.learning_rate != 1.0:
                    mes.append('For Adadelta optimizer use learning rate 1.0')
                    c+=1
        elif self.model.optimizer.learning_rate >=0.00001 and  self.model.optimizer.learning_rate <= 0.01:
          pass
        elif self.model.optimizer.learning_rate > 0.01:
              mes.append('Decrease the learning rate') 
              c+=1
        elif self.model.optimizer.learning_rate < 0.00001:
              mes.append('Increase the learning rate')
              c+=1 
      #.........................................................................................................................                          
      #   #Non-saturating non-linearity
        i = 0
        d = 0
        act = 0
        sub = 'relu'
        activation=''
        print('This is change_name',change_name)
        for layer in self.model.layers:
            d += 1
            if i < (len(change_name)-1) :
                if 'dense' in change_name[0]:
                  if 'dense'  in change_name[i] :
                 
                      activation = str(layer.activation)
                      if 'batchnormalization' not in change_name[i+1]:
                          message.append('Layer ' + str(layer_number[i]) + ' : Missing Batch Normalization --> Add Batch Normanlization after this layer')
                          c+=1
                      if 'activation' in change_name [i+1]:
                          act = 1
                      elif 'activation' not in change_name [i+1]:
                          message.append('Layer ' + str(layer_number[i]) +' : Missing activation function --> Add activation function')
                          c+=1
                  elif act==1:
                          act=0
                  
                  elif 'activation' in change_name[i] :
                          message.append('Layer ' + str(layer_number[i]) + ' : Multiple activation function --> Use activation once')

                if 'conv2d' in change_name[0]:
                  if 'conv2d' in change_name [i]:
                          activation = str(layer.activation)
                          if 'batchnormalization' not in change_name[i+1]:
                              message.append('Layer ' + str(layer_number[i]) + ' : Missing Batch Normalization --> Add Batch Normanlization after this layer')
                              c+=1
                          if 'activation' in change_name [i+1]:
                                act = 1
                              
                          elif 'activation' not in change_name [i+1] and 'activation' not in change_name [i+2]:
                              message.append('Layer ' + str(layer_number[i]) +' : Missing activation function --> Add activation function')
                              c+=1
                          elif act==1 and str(layer.activation) in sub:
                                act=0
                              
                          elif act==1 and str(layer.activation) not in sub:
                              message.append('Layer ' + str(layer_number[i]) +' : Change activation function --> preferred ReLU')
                              c+=1
                          elif 'activation' in change_name[i] :
                              message.append('Layer ' + str(layer_number[i]) + ' : Multiple activation function --> Use activation once')
                              c+=1
                  elif 'dense'  in change_name[i] :
                      activation = str(layer.activation)
                      # if 'batchnormalization' not in change_name[i+1]:
                      #     message.append('Layer ' + str(layer_number[i]) + ' : Missing Batch Normalization --> Add Batch Normanlization after this layer')
                      #     c+=1
                      if 'activation' in change_name [i+1]:
                          act = 1
                      elif 'activation' not in change_name [i+1]:
                          message.append('Layer ' + str(layer_number[i]) +' : Missing activation function --> Add activation function')
                          c+=1
                  elif act==1:
                          act=0
                  
                  elif 'activation' in change_name[i] :
                          message.append('Layer ' + str(layer_number[i]) + ' : Multiple activation function --> Use activation once')
                if 'conv1d' in change_name[0]:
                      if 'conv1d' in change_name [i]:
                          activation = str(layer.activation)
                          if 'batchnormalization' not in change_name[i+1]:
                              message.append('Layer ' + str(layer_number[i]) + ' : Missing Batch Normalization --> Add Batch Normanlization after this layer')
                              c+=1
                          if 'activation' in change_name [i+1]:
                                act = 1
                              
                          elif 'activation' not in change_name [i+1]  and 'activation' not in change_name [i+2]:
                              message.append('Layer ' + str(layer_number[i]) +' : Missing activation function --> Add activation function')
                              c+=1
                         
                          # elif act==1 and str(layer.activation) not in sub:
                          #     message.append('Layer ' + str(layer_number[i]) +' : Change activation function --> preferred ReLU')
                          #     c+=1
                          elif 'activation' in change_name[i+1] and  'activation' in change_name[i+2]:
                              message.append('Layer ' + str(layer_number[i]) + ' : Multiple activation function --> Use activation once')
                              c+=1
            i+=1       
       #.........................................................................................................................                    
      #   #Mismatch between number of classes, last layer activation & loss function
        for layer in self.model.layers:
              print('Layer name is',layer.name)
              print('Again layer name',layer_name)
              print('Dense_units',dense_units)
              if layer.name == layer_name[-1] :
                    if dense_units[-1] ==1 and self.problem_type==0:
                          if 'linear' not in str(layer.activation) :
                                message.append('Layer ' + str(layer_number[-1]) +' : Regression problem --> Use linear activation')
                    elif dense_units[-1] >= 2 and self.problem_type ==1:
                          if 'softmax' in str(layer.activation):
                                if str(self.model.loss).find('categorical_crossentropy') !=-1:
                                  pass
                                else:
                                  mes.append('Change loss function --> Use categorical_crossentropy')
                                  c+=1

                          else:
                                if str(self.model.loss).find('categorical_crossentropy') !=-1:
                                  message.append('Layer ' + str(layer_number[-1]) +' : Wrong activation function --> Multiclass classification use softmax')
                                  c+=1
                                else:
                                  message.append('Layer ' + str(layer_number[-1]) +' :  Wrong activation function --> Multiclass classification use softmax')
                                  mes.append('Change loss function --> Use categorical_crossentropy')
                                  c+=1
                    else:
                          if 'sigmoid' in str(layer.activation):
                                if str(self.model.loss).find('binary_crossentropy') !=-1:
                                  pass
                                else:
                                  mes.append('Change loss function --> Use binary_crossentropy')
                                  c+=1
                          else:
                                if str(self.model.loss).find('binary_crossentropy') !=-1:
                                  message.append('Layer ' + str(layer_number[-1]) +' :  Wrong activation function --> Binary classification use sigmoid')
                                  c+=1
                                  
                                else:
                                  message.append('Layer ' + str(layer_number[-1]) +' : Wrong activation function --> Binary classification use sigmoid')
                                  mes.append('Change loss function --> Use binary_crossentropy')
                                  c+=1
     #.........................................................................................................................                  
      #Check dropout 
        i=0                       
        for i in range(0,len(change_name)-2):
              print('CHecking value of i',i)
              if 'dense' in change_name[i]:
                  j=i+1
                  dr = 0
                  while 'dense' not in change_name[j]:
                        if 'dropout' in change_name[j]:
                              dr+=1
                              
                        j+=1
                  if dr > 1 :
                        message.append('Layer ' + str(layer_number[j-1]) +' : Redundant Dropout --> Remove Dropout ****')
                        c+=1
                  elif dr ==0: 
                        message.append('Layer ' + str(layer_number[j-1]) +' : Missing Dropout --> Add Dropout')
                        c+=1
              if  'conv2d' in change_name[i]:
                  print('Thus is i',i)
                  j=i+1
                  dr = 0
                  while  'maxpooling2d' not in change_name[j]:
                        
                        if 'dropout' in change_name[j]:
                              dr+=1
                        if 'flatten' in change_name[j]:
                           break     
                        j+=1
                  
                  if dr >= 1 and 'dropout' in change_name[j+1] :
                        message.append('Layer ' + str(layer_number[j+1]) +' : Redundant Dropout --> Remove Dropout *')
                        c+=1
                  elif dr ==0 and 'dropout' not in change_name[j]: 
                        message.append('Layer ' + str(layer_number[j]) +' : Missing Dropout --> Add Dropout **')
                        c+=1
              if  'conv1d' in change_name[i]:
                  
                  j=i+1
                  dr = 0
                  while  'maxpooling1d' not in change_name[j]:
                        
                        if 'dropout' in change_name[j]:
                              dr+=1
                        if 'flatten' in change_name[j]:
                           break     
                        j+=1
                 
                  if dr >= 1 and 'dropout' in change_name[j-1] :
                        message.append('Layer ' + str(layer_number[j-1]) +' : Redundant Dropout --> Remove Dropout *')
                        c+=1
                  elif dr ==0 and 'dropout' not in change_name[j]: 
                        message.append('Layer ' + str(layer_number[j]) +' : Missing Dropout --> Add Dropout **')
                        c+=1
              i=j+1  
        #.........................................................................................................................                              
      #   #Trade-off between convolution layer and fully connected layer
        co = 0
        de = 0
        den_layer = []
        if channel == 3 and 'conv2d'  in change_name[0]:
            for l in range(len(change_name)):
                if 'conv2d' in change_name[l]:
                    co +=1
                    conv_layer = layer_number[l]
                elif 'dense' in change_name[l]:
                    de+=1 
                    den_layer.append(layer_number[l])
                
          
            if co < 3 and de >3:
                message.append('Layer '+ str(conv_layer) +' : Increase number of convolution layers')
                message.append('Layer '+ str(den_layer[-2]) + ' : Decrease number of hidden dense layers --> preferred one or two') 
                c+=1
            elif co < 3:
                #print('Layer '+ str(conv_layer) +' : Increase number of convolution layers')
                message.append('Layer '+ str(conv_layer) +' : Increase number of convolution layers')
                
                c+=1
            elif de >3:
                #print('Layer: '+ str(den_layer[-2]) + ' Decrease number of hidden dense layers preferred one or two')
                message.append('Layer '+ str(den_layer[-2]) + ' : Decrease number of hidden dense layers --> preferred one or two')
                c+=1
            
        if channel == 1 and 'conv2d'  in change_name[0]:
             for l in range(len(change_name)):
                if 'conv2d' in change_name[l]:
                    co +=1
                    conv_layer = layer_number[l]
                elif 'dense' in change_name[l]:
                    de+=1 
                    den_layer.append(layer_number[l])
                    
             if co < 2 and de > 3:
                 message.append('Layer '+ str(conv_layer) +' : Increase number of convolution layers')
                 message.append('Layer '+ str(den_layer[-2]) + ' : Decrease number of hidden dense layers --> preferred one or two') 
                 c += 1
             elif co < 2:
                 message.append('Layer '+ str(conv_layer) +' : Increase number of convolution layers')
                 c+=1
               
             elif de>3:
                 
                message.append('Layer '+ str(den_layer[-2]) + ' : Decrease number of hidden dense layers --> preferred one or two')   
                c += 1 
        #.........................................................................................................................                   
          c+=1 

      #   #Global feature extraction
        if dense_count == 2 and 'conv2D' in change_name[0] :
            if channel == 3:
              if dense_units[0] not in range(128,513): 
                  if dense_units[0] > 512:
                      message.append('Layer ' + str(dense_num[0]) + ' : Decrease the units in dense layer --> preferred 512 or 256 or 128') 
                      c+=1
                  if dense_units[0] < 128:
                      message.append('Layer ' + str(dense_num[0]) + ' : Increase the units in dense layer --> preferred 512 or 256 or 128') 
                      c+=1
            if channel == 1 :
                if dense_units[0] not in range(64,129): 
                  if dense_units[0] > 128:
                      message.append('Layer ' + str(dense_num[0]) + ' : Decrease the units in dense layer --> preferred 128 or 64') 
                      c+=1
                  if dense_units[0] < 64:
                              message.append('Layer ' + str(dense_num[0]) + ' : Increase the units in dense layer --> preferred 128 or 64') 
                              c+=1
                 
        elif dense_count <= 3 and  'conv2D' in change_name[0]:
              if dense_units[0] >= dense_units[1]:
                  if channel == 3:
                          
                          if dense_units[0] not in range(256,513): 
                                if dense_units[0] > 512:
                                      message.append('Layer ' + str(dense_num[0]) + ' :Decrease the units in dense layer --> preferred 512 or 256 or 128') 
                                      c+=1
                                if dense_units[0] < 256:
                                      message.append('Layer ' + str(dense_num[0]) + ' :Increase the units in dense layer --> preferred 512 or 256 or 128') 
                                      c+=1
                          if dense_units[1] not in range(128,257):
                                if dense_units[1] > 256:
                                      message.append('Layer ' + str(dense_num[1]) + ' :Decrease the units in dense layer --> preferred 256 or 128') 
                                      
                                if dense_units[1] < 128:
                                      message.append('Layer ' + str(dense_num[1]) + ' :Increase the units in dense layer --> preferred 256 or 128') 
                                      
                  elif channel == 1:
                        if dense_units[0] not in range(64,129): 
                            if dense_units[0] > 128:
                                  message.append('Layer ' + str(dense_num[0]) + ' : Decrease the units in dense layer --> preferred 128 or 64') 
                                  c+=1
                            if dense_units[0] < 64:
                                  message.append('Layer ' + str(dense_num[0]) + ' : Increase the units in dense layer --> preferred 128 or 64') 
                                  c+=1
                            if dense_units[1] not in range(64,129):
                                if dense_units[1] > 128:
                                  message.append('Layer ' + str(dense_num[1]) + ' : Decrease the units in dense layer --> preferred 64') 
                                  
                                if dense_units[1] < 64:
                                  message.append('Layer ' + str(dense_num[1]) + ' : Increase the units in dense layer --> preferred 64') 
              else:
                  
                  message.append('Layer ' + str(dense_num[0]) + ' has less units than ' + 'Layer ' + str(dense_num[1])+' : Keep the units same or decrease units while going deeper')
                  
      
   
                          
        # #Inaccurate number of filters
        ck =0 
        if check==1:
          if channel == 3:
            for i in range(len(fil)):
                if fil[i] in range(32,513):
                    pass
                elif fil[i] < 32:
                    message.append('Layer '+ str(num_layer[i]) +' : Increase the number of filters --> preferred between 32-512')
                    c+=1
                elif fil[i] > 512:
                    message.append('Layer '+ str(num_layer[i]) +' : Decrerase the number of filters --> preferred between 32-512')
                    c+=1

          if channel == 1:
            
            for i in range(len(fil)):
                if fil[i] in range(6,65):
                    pass
                elif fil[i] < 6:
                    message.append('Layer '+ str(num_layer[i]) +' : Increase the number of filters --> preferred between 6-64')
                    c+=1
                elif fil[i] > 64:
                    message.append('Layer '+ str(num_layer[i]) +' : Decrease the number of filters --> preferred between 6-64')
                    c+=1

        
        
        else:
            if channel == 3:
              if fil[0] < 32:
                  message.append('Layer '+ str(num_layer[0]) +' : Increase the number of filters --> preferred between 32-512')
                  ck=1
                  c+=1
              elif fil[0] > 64:
                  message.append('Layer '+ str(num_layer[0]) +' : Decrease the number of filters --> preferred between 32-64')  
                  ck=1
                  c+=1
              for i in range(1,len(fil)):
                    
                    if fil[i] not in range(32,513):
                      
                      message.append('Layer '+str(num_layer[i])+' : Increase the number of filters --> preferred between 32-512')
                      c+=1
                    elif ck == 0:
                      message.append('Layer '+str(num_layer[i])+' : Increase number of filters while going deep --> preferred between 32-512')
                      c+=1 
            if channel == 1 and 'conv2d' in change_name[0]:
              
              if fil[0] not in range(8,65):
                  message.append('Layer '+ str(num_layer[i]) +' : Increase the number of filters --> preferred between 8-64')
              for i in range(1,len(fil)):
                    
                          
                    if fil[i] not in range(8,65):
                        print('here')
                        message.append('Layer '+str(num_layer[i])+' : Increase number of filters while going deep --> preferred between 8-64') 
                        c+=1
                    else:
                        print('now here')
                        message.append('Layer '+str(num_layer[i])+' : Increase number of filters while going deep --> preferred between 8-64')
                        c+=1
        res = sorted(message, key = lambda x: int(x.split()[1]))
        for i in res:
          print(i)
        res1= sorted(mes)
        for i in res1:
            print(i)
        print("--- %s seconds ---" % (time.time() - start_time))
        
        if c>=1:
            self.model.stop_training = True
            sys.exit(1) 