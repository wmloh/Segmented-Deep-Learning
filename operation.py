class Operation:

    input_dim = 0
    output_dim = 0
    model_list = list()
    
    def __init__(self, input_dim, output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim

    def add(self, model):
        self.model_list.append(model)

    def remove(self):
        self.model_list.pop()

    def calc(self, x):
        output = x
        for i in self.model_list:
            output = i.predict(output, display=False)
        return output

    def change_dim(self, dim_in, dim_out):
        self.input_dim = dim_in
        self.output_dim = dim_out
