from model import Model
import matplotlib.pyplot as plt

def error_vs_reg(hl_dim, dataset_fnc, lambda_range, train_size=500, epoch=2000, test_size=1000):
    '''
    list(Int), (Int -> NP.ARRAY, NP.ARRAY), list(Int), Int, Int, Int ->
        list(Float), list(Float), list(list(NP.ARRAY)), NP.ARRAY, NP.ARRAY

    Evaluates the model with <hl_dim> specified, on datasets given by
    <dataset_fnc> which is passed only one size parameter, against
    a list of regularization strengths <lambda_range>.

    Returns:
    * Validation loss values
    * Train loss values
    * Parameters of model for each lambda
    * X dataset for validation
    * y dataset for validation
    '''
    X, y = dataset_fnc(train_size)
    test_X, test_y = dataset_fnc(test_size)
    input_dim = X.shape[1]
    
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    
    train_loss = list()
    val_loss = list()
    params = list()

    for i in lambda_range:
        m = Model(input_dim, 2, hl_dim)
        
        m.train(X, y, epoch=epoch, reg_strength=i)
        x = m.show_params(display=False)
        
        p1, p2 = x
        params.append([p1, p2])
        
        train_loss.append(m.validate(X, y, reg_strength=0))
        val_loss.append(m.validate(test_X, test_y, reg_strength=0))

        del m
    
    ax1.plot(lambda_range, train_loss, label='Training loss')
    ax1.plot(lambda_range, val_loss, label='Test loss')
    ax1.legend()
    fig.show()

    return val_loss, train_loss, params, test_X, test_y

