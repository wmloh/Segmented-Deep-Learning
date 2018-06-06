from util.util import to_one_hot

class Operation:
	
    '''
    The Operation object is an abstraction to piece multiple Model objects.
    See https://github.com/wmloh/Segmented-Deep-Learning for more details.
    '''

    names = list()

    def __init__(self, input_dim, output_dim, one_hot=True, name=None):
        
        '''
        Int, Int, Bool, Str/None -> OPERATION

        Initializes an Operation object.
        Parameters:
        * input_dim - number of input data
        * output_dim - number of classes
        * one_hot - representation of predictions
        * name - unique name of object
        '''
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.model_list = list()
        self.layer_count = 0
        self.one_hot = one_hot
        self.length = 0
        if name == None:
            self.name = str(id(self))
        else:
            assert(type(name) is str)
            if name in self.names:
                raise NameError('This name already exists in another Operation object')
            self.name = name
            self.names.append(name)
        
    def __call__(self, x):

        '''
        The Operation object is callable, i.e. it acts like a function
        to predict the class with the given <x> and models
        '''
        return self.calc(x)

    def __str__(self):

        '''
        User-defined print function that prints the architecture of models
        in the Operation object
        '''
        if self.length == 0:
            return 'OPERATION: %s' % (self.name)
        else:
            arch = '"input[dim %i]" -> ' % self.input_dim
            for i in self.model_list:
                arch += '{' + ', '.join([j.name for j in i]) + '} -> '
            arch += '"output[dim %i]"' % self.output_dim
            return 'OPERATION: %s\n%s' % (self.name, arch)

    def __contains__(self, name):

        '''
        Str -> Bool

        Returns true if and only if there is a Model object in
        the Operation object called <name>
        '''
        for i in self.model_list:
            for j in i:
                if name == j.name:
                    return True
        return False

    def __len__(self):

        '''
        Returns the number of Models in the Operation object
        '''
        return self.length

    def __getitem__(self, layer):

        '''
        Prints the details of each model in the given layer
        by accessing items of the object, i.e. ops[index]
        '''
        print('\n\n'.join([i.__repr__() for i in self.model_list[layer]]))

    @property
    def check(self):
        '''
        Attribute of Operation object that return True if and only if
        the outputs of a layer correspond to the inputs of the next
        layer.
        Used as a debugging tool.
        '''
        for layer in range(self.layer_count-1):
            output_count = len(self.model_list[layer])
            input_count = sum([i.input_dim for i in self.model_list[layer+1]])
            if output_count != input_count:
                return False
        return True
    
    def add(self, model, layer):
        
        '''
        MODEL, Int -> None

        Requires:
        * Models added must adhere to the input_dim and output_dim of the 
                Operation object and other models.
        * layer >= 1

        Adds a model to the end of a given layer of the Operation object.
        '''
        if self.layer_count < layer:
            self.add_layers(layer)
            
        self.model_list[layer-1].append(model)
        self.length += 1
        
    def add_layers(self, layer):

        '''
        Int -> None

        Requires:
        * self.layer_count <= layer

        Increases the number of layers up to <layers> by adding
        empty layers.
        '''
        for i in range(layer - len(self.model_list)):
            self.model_list.append(list())
        self.layer_count = layer
            
    def remove(self, layer, position, display=True):
        
        '''
        Int, Int -> None/MODEL

        Requires:
        * Both layer and position must exist
        * layer >= 1
        * position >= 0

        Removes the model in a given layer at a given position from
        the top.
        '''
        layer -= 1
        if layer < self.layer_count and position < len(self.model_list[layer]):
            model = self.model_list[layer].pop(position)
            if display:
                print('%s removed from layer %i and pos %i' % (model.name,
                                                               layer,
                                                               position))
            
        else:
            raise ValueError('At least one of the layer or position inputs does not exist')
        

    def calc(self, x):
        
        '''
        list(NP.ARRAY) -> Int/NP.ARRAY

        Passes <x> into the first layer Model in Operation object and every
        output is passed into the subsequent models along the chain.

        Note: Recommended to call the check attribute before calling this
        function for the first time
        '''
        output = x
        for i in self.model_list:
            
            lst = []
            count = 0
            for j in i:
                lst.append([count, count + j.input_dim])
                count += j.input_dim
            output = [i[k].predict(output[l[0]:l[1]],
                                   display=False,
                                   one_hot=False) for k, l in enumerate(lst)]
            output = [int(i) for i in output]

        if self.one_hot:
            return to_one_hot(output[0], self.output_dim)
        return output[0]

    def change_dim(self, dim_in, dim_out):
        
        '''
        Int, Int -> None

        Changes input_dim and output_dim of the Operation object.
        '''
        self.input_dim = dim_in
        self.output_dim = dim_out
