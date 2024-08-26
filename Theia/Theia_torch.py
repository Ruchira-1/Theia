import sys
import torch

class Theia_callback:
    
    def __init__(self):
        pass
    def check(train_data, test_data, model, loss, optimizer,batch_size,problem_type,input_type):
            mes = []
            mes1 = []
            
            c = 0
            #Data
            #For dimension and channels
            train_sample = next(iter(train_data)) 
            image, label = train_sample  
            image_shape = list(image.shape)
            dimension = [image_shape[1],image_shape[2]]
            channels = image_shape[0]

            #For input range computation
            test_sample = next(iter(test_data)) 
            image1, label1 = test_sample 
            value1, indices = (torch.max(image[0],dim=1))
            train_upper_range = max(value1.tolist())
            value2, indices = torch.min(image[0],dim=1)
            train_lower_range = min(value2.tolist())
            
            value3, indices = (torch.max(image1[0],dim=1))
            test_upper_range = max(value3.tolist())
            value4, indices = torch.min(image1[0],dim=1)
            test_lower_range = min(value4.tolist())
            


            #For number of samples
            no_of_samples = len(train_data)


            #Model
            filters = []
            # kernel_size = []
            # conv_strides = []
            # pool_size = []
            units = []
            activation = []
            layer_name = []
            ConvCount = 0
            DenseCount = 0
            units_out = []
            for  m in model.modules():
                    layer_name.append(type(m).__name__)
                    if 'Conv2d' == type(m).__name__:
                        ConvCount = ConvCount + 1
                        filters.append(m.out_channels)
                        # kernel_size.append(m.kernel_size)
                        # conv_strides.append(m.stride)
                    # if 'MaxPool2d' == type(m).__name__:
                    #     pool_size.append((m.kernel_size))
                    if 'Linear' == type(m).__name__:
                        DenseCount = DenseCount + 1
                        units.append(m.out_features)
                        units_out.append(m.out_features)
                   
            layer_name = [i for i in layer_name if i != 'Sequential']
            
            #Learner
            learning_rate = optimizer.param_groups[0]['lr']
            loss = loss.__class__.__name__
            batch_size = batch_size
            print('Loss is',loss)
            print('Batch size is',batch_size)

    
        # #Normalization of input
            if  (train_lower_range != 0.0 and train_upper_range != 1.0) or (train_lower_range != -1.0 and train_upper_range != 1.0):
                mes1.append('Normalize the training data')
                c+=1
            if  (test_lower_range != 0.0 and test_upper_range != 1.0) or (test_lower_range != -1.0 and test_upper_range != 1.0):
                mes1.append('Normalize the test data')
                c+=1


            #Incorrect number of filters
            ck = 0 
            j=0
            #if check==1:
            if input_type == 3:
                for i in range(len(layer_name)):
                    if layer_name[i] == 'Conv2d' :
                        if filters[j] in range(16,513):
                            pass
                        elif filters[j] < 16:
                            mes.append('Layer '+ str(i) +' : Increase the number of filters --> preferred between 16-512')
                            c+=1
                        elif filters[j] > 512:
                            mes.append('Layer '+ str(i) +' : Decrease the number of filters --> preferred between 16-512')
                            c+=1
                        j+=1

            if input_type == 1:
                
                for i in range(len(layer_name)):
                    if layer_name[i] == 'Conv2d' :
                        if filters[j] in range(6,256):
                            pass
                        elif filters[j] < 6:
                            mes.append('Layer '+ str(i) +' : Increase the number of filters --> preferred between 6-256')
                            c+=1
                        elif filters[j] > 256:
                            mes.append('Layer '+ str(i) +' : Decrease the number of filters --> preferred between 6-256')
                            c+=1
                        j+=1

            
                
            #Choice of nonlinearity
            j=0
            for i in layer_name:
                if i == 'Conv2d' :
                    if 'BatchNorm2d' not in layer_name[j+1]:
                              mes.append('Layer ' + str (j+1) + ' : Missing Batch Normalization --> Add Batch Normanlization after this layer')
                              c+=1
                    if layer_name[j+1] == 'ReLU' or layer_name[j+2] == 'ReLU':
                        pass
                    elif layer_name[j+1] == 'MaxPool2d':
                        mes.append('Layer ' + str (j+1) + ' : Missing activation function --> Add activation function')
                        c+=1
                    elif layer_name[j+1] in ('ReLU','Tanh','Sigmoid') and layer_name[j+2] in ('ReLU','Tanh','Sigmoid'):
                        mes.append('Layer ' + str(j+2) +' : Multiple activations ')
                        c+=1
                    else:
                        mes.append('Layer ' + str(j+1) +' : Change activation function --> preferred ReLU')
                        c+=1
                elif i == 'Conv1d' or i == 'Linear' and j<len(layer_name)-1:
                    if 'BatchNorm1d' not in layer_name[j+1]:
                              mes.append('Layer ' + str (j+1) + ' : Missing Batch Normalization --> Add Batch Normanlization after this layer')
                              c+=1
                
                j+=1


            j=0
            if layer_name[-1] != 'Linear':
                size = layer_name[:-2]
            else:
                size = layer_name[:-1]
            for i in size:
                if i == 'Linear' :
                    if layer_name[j+1] == 'ReLU':
                        pass
                    elif layer_name[j+1] != 'ReLU' :
                        mes.append('Layer ' + str (j) + ' :Missing activation function --> Add activation function')
                        c+=1
                    else:
                        mes.append('Layer ' + str(j+1) +' : Change activation function --> preferred ReLU')
                        c+=1
                j+=1   

             #Check dropout 
            i=0                       
            for i in range(0,len(layer_name)-2):
                print('CHecking value of i',i)
                if 'Linear' in layer_name[i]:
                    j=i+1
                    dr = 0
                    while 'Linear' not in layer_name[j]:
                            if 'Dropout' in layer_name[j]:
                                dr+=1
                                
                            j+=1
                    if dr > 1 :
                            mes.append('Layer ' + str(j-1) +' : Redundant Dropout --> Remove Dropout ****')
                            c+=1
                    elif dr ==0: 
                            mes.append('Layer ' + str(j-1) +' : Missing Dropout --> Add Dropout')
                            c+=1
                if  'Conv2d' in layer_name[i]:
                    print('Thus is i',i)
                    j=i+1
                    dr = 0
                    while  'MaxPool2d' not in layer_name[j]:
                            
                            if 'Dropout' in layer_name[j]:
                                dr+=1
                            # if 'flatten' in layer_name[j]:
                            #     break     
                            j+=1
                    
                    if dr >= 1 and 'Dropout' in layer_name[j+1] :
                            mes.append('Layer ' + str(j+1) +' : Redundant Dropout --> Remove Dropout')
                            c+=1
                    elif dr ==0 and 'Dropout' not in layer_name[j]: 
                            mes.append('Layer ' + str(j) +' : Missing Dropout --> Add Dropout')
                            c+=1
                if  'Conv1d' in layer_name[i]:
                    
                    j=i+1
                    dr = 0
                    while  'MaxPool1d' not in layer_name[j]:
                            
                            if 'Dropout' in layer_name[j]:
                                dr+=1
                            # if 'Flatten' in layer_name[j]:
                            #     break     
                            # j+=1
                    
                    if dr >= 1 and 'Dropout' in layer_name[j-1] :
                            mes.append('Layer ' + str(j-1) +' : Redundant Dropout --> Remove Dropout *')
                            c+=1
                    elif dr ==0 and 'Dropout' not in layer_name[j]: 
                            mes.append('Layer ' + str(j) +' : Missing Dropout --> Add Dropout **')
                            c+=1
                i=j+1 

            



            # Insufficient Downsampling
            convpool = []
            count = 0
            for j in range(len(layer_name)):    
                if layer_name[j] == 'Conv2d' or layer_name[j] == 'AvgPool2d' or layer_name[j] == 'MaxPool2d':
                    convpool.append((j,layer_name[j]))
            for i in range (len(convpool)):
                if convpool[i][1] == 'Conv2d':
                    count +=1
                if convpool[i][1] == 'AvgPool2d' or convpool[i][1] == 'MaxPool2d':
                    if count > 4 :
                        mes.append('Layer ' + str(convpool[i][0])+ ' : Add pooling layer')
                        c+=1
                    count = 0


            #Inappropriate number of convolution layers  
            if input_type == 1 and ConvCount<2:
                    mes1.append('Add more convolution layers')
                    c+=1
            if input_type == 3 and ConvCount<3:
                    mes1.append('Add more convolution layers')
                    c+=1


            #Global feature extraction
            if layer_name[0] == 'Conv2d':
                flag = 0
                layer_num = []
                test_list1 = units[:]
                test_list1.sort(reverse = True)
                if test_list1 == units:
                        flag = 1
                for j in range(len(layer_name)):
                    if layer_name[j] == 'Linear':
                        layer_num.append(j)
            
                i = 0
                for j in layer_num[:-1]:
                    if flag == 0:
                        mes.append('Layer ' + str(j)+ ' : Keep the units same or decrease units while going deeper')
                        c+=1
                    elif units[i] < 64:
                        mes.append('Layer ' + str(j) + ' : Increase the units in dense layer --> preferred 64-1024') 
                        c+=1
                    elif units[i] >1024:
                        mes.append('Layer ' + str(j) + ' : Decrease the units in dense layer --> preferred 64-1024') 
                        c+=1
                    i+=1

            
            #Improper Number of Fully-connected Layers
            if DenseCount > 4 and layer_name[0]=='Conv2d':
                    mes1.append('Decrease number of Linear layers --> preferred 1-3')
                    c+=1

            #Mismatch between number of classes, last layer activation and loss function
            no_of_classes = units[-1]
            last_layer = layer_name[-1]
            if no_of_classes == 1 and problem_type==0:
                if  last_layer == 'Sigmoid' or last_layer == 'Softmax':
                    mes.append('Layer ' + str(len(layer_name)-1) + ' : Regression Problem --> Remove activation ')
                    c+=1
            if no_of_classes == 1 and problem_type==1:
                if loss == 'BCELoss' and last_layer != 'Sigmoid':
                    mes.append('Layer ' + str(len(layer_name)-1) + ' : Change loss function to BCEWithLogitsLoss or add Sigmoid as last layer activation')
                    c+=1
                elif loss == 'BCEWithLogitsLoss' and last_layer in ('Sigmoid','Softmax'):
                    mes.append('Layer ' + str(len(layer_name)-1) + ' : BCEWithLogitsLoss has in-build sigmoid activation, remove last layer activation')
                    c+=1   
                else:
                    mes1.append('Binary classification change loss function --> use BCELoss or BCEWithLogitsLoss')
                    c+=1 
            if no_of_classes >2 and problem_type==1:
                if loss == 'CrossEntropyLoss' and last_layer in ('Sigmoid','Softmax'):
                    mes.append('Layer ' + str(len(layer_name)-1) + ' : CrossEntropyLoss has in-build softmax activation, remove last layer activation')
                    c+=1
                elif loss != 'CrossEntropyLoss' and last_layer in ('Sigmoid','Softmax'):
                    mes1.append('Multi-class classification change loss function --> use CrossEntropyLoss')
                    mes.append('Layer ' + str(len(layer_name)-1) + ' : Remove last layer activation')
                    c+=1
                elif loss == 'CrossEntropyLoss':
                    pass
                else:
                    mes1.append('Multi-class classification change loss function --> use CrossEntropyLoss')
                    c+=1 

            #Indequate Batch Size   
            
            if batch_size > 256: 
                    mes1.append('Decrease the batch size --> preferred 32,64,128,256')
                    c+=1
           
            if batch_size < 32: 
                    mes1.append('Increase the batch size --> preferred 32,64,128,256')
                    c+=1

            #Learning rate out-of-bounds ≥ 0.0001 and ≤ 0.01
            if learning_rate >= 0.0001 and  learning_rate <= 0.01:
             pass
            elif learning_rate > 0.01:
                mes1.append('Decrease the learning rate --> preferred between 0.01 to 0.0001') 
                c+=1
            elif learning_rate < 0.0001:
                mes1.append('Increase the learning rate --> preferred between 0.01 to 0.0001')
                c+=1           
                
            res = sorted(mes, key = lambda x: int(x.split()[1]))
            for i in res:
                print(i)
            res1= sorted(mes1)
            for i in res1:
                print(i)
            
            if c>=1:
            
                sys.exit(1) 