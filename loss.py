import torch.nn
import torch
import numpy as np

def divide_no_nan(a, b):
    """
    a/b where the resulted NaN or Inf are replaced by 0.
    """
    result = a / b
    result[result != result] = .0
    result[result == np.inf] = .0
    return result

class NaNMSELoss():
    def __init__(self, cfg):
        super().__init__()
        self.model_name = cfg["modelname"]

    def fit(self, y_pred,y_true,lossmse):
        mask = y_true == y_true #海洋为None，不得true
        y_true = y_true[mask]
        y_pred = torch.squeeze(y_pred[mask])
        loss = torch.sqrt(lossmse(y_true, y_pred))
        return loss

class MeanBiasError( ):
    def __init__(self, cfg):
        super().__init__(name="mean_bias_error")

    def fit(self, y_true, y_pred):
        mask = y_true == y_true
        y_true = y_true[mask]
        y_pred = torch.squeeze(y_pred[mask])
        bias_error = torch.sub(y_true, y_pred)
        size_value = y_true.size()[0]
        mbe_loss = torch.mean(torch.sum(bias_error) / size_value)
        return mbe_loss


class RelativeAbsoluteError( ):
    def __init__(self, cfg):
        super().__init__(name="relative_absolute_error")

    def fit(self, y_true, y_pred):
        mask = y_true == y_true
        y_true = y_true[mask]
        y_pred = torch.squeeze(y_pred[mask])
        true_mean = torch.mean(y_true)
        squared_error_num = torch.sum(torch.abs(torch.sub(y_true, y_pred)))
        squared_error_den = torch.sum(torch.abs(torch.sub(y_true, true_mean)))
        rae_loss = torch.div(squared_error_num, squared_error_den)
        return rae_loss


class RelativeSquaredError( ):
    def __init__(self, cfg):
        super().__init__(name="relative_squared_error")

    def fit(self, y_true, y_pred):
        mask = y_true == y_true
        y_true = y_true[mask]
        y_pred = torch.squeeze(y_pred[mask])
        squared_error_num = torch.sum(torch.square(torch.sub(y_true, y_pred)))
        squared_error_den = torch.sum(torch.square(torch.sub(y_true, y_pred)))
        rse_loss = torch.div(squared_error_num, squared_error_den)
        return rse_loss


class NormalizedRootMeanSquaredError( ):
    def __init__(self, cfg):
        super().__init__(name="normalized_root_mean_squared_error")

    def fit(self, y_true, y_pred):
        mask = y_true == y_true
        y_true = y_true[mask]
        y_pred = torch.squeeze(y_pred[mask])
        squared_error = torch.square(torch.sub(y_true, y_pred))
        sum_squared_error = torch.sum(squared_error)
        size_value = y_true.size()[0]
        rmse = torch.sqrt(sum_squared_error / size_value)
        nrmse_loss = torch.div(rmse, backend.std(y_pred))
        return nrmse_loss


class RelativeRootMeanSquaredError( ):
    def __init__(self, cfg):
        super().__init__(name="relative_root_mean_squared_error")

    def fit(self, y_true, y_pred):
        mask = y_true == y_true
        y_true = y_true[mask]
        y_pred = torch.squeeze(y_pred[mask])
        num = torch.sum(torch.square(torch.sub(y_true, y_pred)))
        den = torch.sum(torch.square(y_pred))
        squared_error = torch.div(num, den)
        rrmse_loss = torch.sqrt(squared_error)
        return rrmse_loss


class RootMeanSquaredLogarithmicError( ):
    def __init__(self, cfg):
        super().__init__(name="root_mean_squared_logarithmic_error")

    def fit(self, y_true, y_pred):
        mask = y_true == y_true
        y_true = y_true[mask]
        y_pred = torch.squeeze(y_pred[mask])
        #y_pred = tf.convert_to_tensor(y_pred)
        #y_true = tf.cast(y_true, y_pred.dtype)
        first_log = torch.log(y_pred + 1.0)
        second_log = torch.log(y_true + 1.0)

        return torch.sqrt(torch.mean(
            torch.sqrt(first_log - second_log)
        ))
    
class MeanSquaredLogarithmicError( ):
    def __init__(self, cfg):
        super().__init__(name="mean_squared_logarithmic_error")

    def fit(self, y_true, y_pred):
        mask = y_true == y_true
        y_true = y_true[mask]
        y_pred = torch.squeeze(y_pred[mask])
        #y_pred = tf.convert_to_tensor(y_pred)
        #y_true = tf.cast(y_true, y_pred.dtype)
        first_log = torch.log(y_pred + 1.0)
        second_log = torch.log(y_true + 1.0)

        return torch.mean(
            torch.sqrt(first_log - second_log)
        )

class Huber():
    def __init__(self, cfg):
        super().__init__()
        self.model_name = cfg["modelname"]
        
    def fit(self, y_pred,y_true):
        mask = y_true == y_true #海洋为None，不得true
        y_true = y_true[mask]
        y_pred = torch.squeeze(y_pred[mask])
        lossmse = torch.nn.SmoothL1Loss(reduce = False,size_average=False)
        loss = lossmse(y_true, y_pred)
        return loss
    
class Huber1():
    def __init__(self, cfg):
        super().__init__()
        self.model_name = cfg["modelname"]
    def fit(self, y_pred,y_true):
        mask = y_true == y_true #海洋为None，不得true
        y_true = y_true[mask]
        y_pred = torch.squeeze(y_pred[mask])
        
        residual = torch.abs(y_true - y_pred)
        mask = residual <1.0
        
        loss = torch.where (mask,0.5*residual**2,1.0*residual-0.5*1.0**2)
        return loss.mean()
    
class MeanAbsolutePercentError():
    def __init__(self, cfg):
        super().__init__()
        self.model_name = cfg["modelname"]
        
    def fit(self, y_pred,y_true):
        mask = y_true == y_true #海洋为None，不得true
        y_true = y_true[mask]
        y_pred = torch.squeeze(y_pred[mask])
        AbsolutePercentError = (torch.abs(y_pred-y_true)+1e-7)/(torch.abs(y_true)+1e-7)
        
        loss = torch.mean(AbsolutePercentError)
        return loss

    
class  RootMeanSquaredError():
    def __init__(self, cfg):
        super().__init__()
        self.model_name = cfg["modelname"]
    def fit(self, y_pred,y_true):
        mask = y_true == y_true #海洋为None，不得true
        y_true = y_true[mask]
        y_pred = torch.squeeze(y_pred[mask]) 
        loss = torch.sum((y_pred - y_true)*(y_pred - y_true))        
        loss = loss / y_pred.size()[0]      
    
        #loss = torch.sum(torch.square(y_pred - y_true) , axis= 1)/(y_pred.size()[1]) #Taking the mean of all the squares by dividing it with the number of outputs i.e 20 in my case
        loss = torch.sqrt(loss)
        #loss = torch.sum(loss)/predicted_x.size()[0]  #averaging out by batch-size
        return loss
    
class  LogCosh():
    def __init__(self, cfg):
        super().__init__()
        self.model_name = cfg["modelname"]
        
    def fit(self, y_pred,y_true):
        mask = y_true == y_true #海洋为None，不得true
        y_true = y_true[mask]
        y_pred = torch.squeeze(y_pred[mask])        
    
        loss = torch.log(torch.cosh(y_pred - y_true))
        return torch.mean(loss)
    
class SquarMeanAbsolutePercentError():
    def __init__(self, cfg):
        super().__init__()
        self.model_name = cfg["modelname"]
        
    def fit(self, y_pred,y_true):
        mask = y_true == y_true #海洋为None，不得true
        y_true = y_true[mask]
        y_pred = torch.squeeze(y_pred[mask])
        return 200 * torch.mean(divide_no_nan(torch.abs(y_pred - y_true),
                                          torch.abs(y_pred) + torch.abs(y_true)) * mask)