from keras.constraints import Constraint
from keras import backend as K

# ------------------------------------------------------------------------------------------------
class AdversarialOptimizerSimultaneousWithLoops(object):
    """
    Perform simultaneous updates for each player in the game.
    """

    def __init__(self,nloops=None):
        self.nloops = nloops
        
    def make_train_function(self, inputs, outputs, losses, params, optimizers, constraints, model_updates,
                            function_kwargs):
        return K.function(inputs,
                          outputs,
                          updates=self.call(losses, params, optimizers, constraints) + model_updates,
                          **function_kwargs)

    def call(self, losses, params, optimizers, constraints):
        updates = []
        if self.nloops is None:
            nloops = [1]*len(losses)
        else:
            nloops = self.nloops
        for nloops, loss, param, optimizer, constraint in zip(nloops, losses, params, optimizers, constraints):
            for loop in range(nloops):
                updates += optimizer.get_updates(param, constraint, loss)
        # print(updates)
        return updates


# ------------------------------------------------------------------------------------------------
class WeightClip(Constraint):
    '''Clips the weights incident to each hidden unit to be inside a range
    '''
    def __init__(self, c=2, **kwargs):
        self.c = c
        print('WeightClip',self.c)
        
    def __call__(self, p):
        print('calling WeightClip',self.c)
        return K.clip(p, -self.c, self.c)

    def get_config(self):
        return {'name': self.__class__.__name__,
                'c': self.c}


# ------------------------------------------------------------------------------------------------
def wgan_loss(y_true, y_pred): # y = 1:true, -1:fake
    return -K.mean(y_true * y_pred, axis = -1)


