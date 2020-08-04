import numpy as np

def deviance(mu, y, loss_type = 'poisson'):
    """Calculate Poisson-devinace-explained between pairs of columns from the matrices mu and y.
    author: Shih-Yi
    The version here has improved numerical stability.
    Input: first dimension is number of timepoints and second dimension is number of neurons
    mu: model prediction
    y: true value
    loss_type: 'poisson', 'gaussian', or 'binominal' ('exponential' is numerically unstable)
    Output:
    dev: fraction deviance explained for individual neurons
    d_model: raw deviance of the model
    d_null: raw deviance of the null model (predict every point with the mean of y)
    """
    def stable(x):
        return x + 10 * np.finfo(x.dtype).tiny
    assert (mu.shape == y.shape), "Shapes " + str(mu.shape) + " and " + str(y.shape) + " don't match!"
    mean_y = np.mean(y, axis=0)
    log_y = np.log(stable(y))
    if loss_type == 'poisson':
        d_model = 2.0 * np.sum(y * (log_y - np.log(stable(mu))) + mu - y, axis=0)
        d_null = 2.0 * np.sum(y * (log_y - np.log(stable(mean_y))) + mean_y - y, axis=0)
    elif loss_type == 'gaussian':
        d_model = np.sum((y - mu)**2, axis=0)
        d_null = np.sum((y - mean_y)**2, axis=0)
    elif loss_type == 'exponential':
        d_model = 2.0 * np.sum(np.log(stable(mu)) - log_y + y * (y - mu), axis=0)
        d_null = 2.0 * np.sum(np.log(stable(mean_y)) - log_y + y * (y - mean_y), axis=0)
    elif loss_type == 'binominal':
        d_model = 2.0 * np.sum(-y*np.log(stable(mu))-(1.-y)*np.log(stable(1.-mu))
                               +y*np.log(stable(y))+(1.-y)*np.log(stable(1.-y)), axis=0)
        d_null = 2.0 * np.sum(-y*np.log(stable(mean_y))-(1.-y)*np.log(stable(1.-mean_y))
                               +y*np.log(stable(y))+(1.-y)*np.log(stable(1.-y)), axis=0)
    dev = 1.0 - d_model/stable(d_null)
    if isinstance(dev, type(y)): # if dev is still an ndarray (skip if is a single number)
        dev[mean_y == 0] = 0  # If mean_y == 0, we get 0 for model and null deviance, i.e. 0/0 in the deviance fraction.
    return dev, d_model, d_null