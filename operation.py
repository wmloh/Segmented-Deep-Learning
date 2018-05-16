class Operation:
	
	'''
	The Operation object is an abstraction to piece multiple Model objects.
	See https://github.com/wmloh/Segmented-Deep-Learning for more details.
	'''

    def __init__(self, input_dim, output_dim):
    	'''
		(Int, Int) -> OPERATION

		Initializes Operation object with <input_dim> number of input values
		it can receive and <output_dim> number of classes.
    	'''
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.model_lis = list()

    def add(self, model):
    	'''
		MODEL -> None

		Requires:
		* Models added must adhere to the input_dim and output_dim of the 
			Operation object and other models.

		Adds a model to the end of the Operation object.
    	'''
        self.model_list.append(model)

    def remove(self):
    	'''
		Requires:
		* Operation object must be non-empty.

		Removes the most recently added model in Operation object.
    	'''
        self.model_list.pop()

    def calc(self, x):
    	'''
		NP.ARRAY -> Int/NP.ARRAY

		Passes <x> into the first Model in Operation object and every
		output is passed into the subsequent models along the chain.
    	'''
        output = x
        for i in self.model_list:
            output = i.predict(output, display=False)
        return output

    def change_dim(self, dim_in, dim_out):
    	'''
		Int, Int -> None

		Changes input_dim and output_dim of the Operation object.
    	'''
        self.input_dim = dim_in
        self.output_dim = dim_out
