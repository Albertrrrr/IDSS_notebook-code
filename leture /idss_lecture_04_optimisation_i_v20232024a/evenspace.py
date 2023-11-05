import numpy as np
import matplotlib.pyplot as plt

# same loss as before
def pt_fit_loss(x, shape):    
    # reshape back to the original shape
    x = x.reshape(shape)
    # compute pairwise distances
    distances = np.linalg.norm(x[:,:,None] - x[:,:,None].T, axis=1, ord=2)
    # compute difference between pairwise distance and 0.5
    return np.sum((0.8-distances)**2) 


def pt_fit_plot(res, shape, temperature_fn=None, title="", figsize=6):
    
    fig = plt.figure(figsize=(figsize,figsize))
    ax = fig.add_subplot(1,1,1)
    xs = np.linspace(0,1.0, res.iters)

    # plot each iteration
    for i in range(res.iters):
        ax.cla()
        ax.set_title(title)
        
        try: 
            # show the loss up to this time point
            ax.plot(xs[:i], res.best_losses[:i]*2e-3, 'r-')
            ax.text(xs[i], res.best_losses[i]*2e-3, 'Loss', color='r')

            if temperature_fn is not None:
                ax.plot(xs[:i], temperature_fn(np.arange(len(xs[:i]))), 'g-')
                ax.text(xs[i], temperature_fn(i), 'Temperature', color='g')

            layout = res.best_thetas[i].reshape(shape)
            # hold the axis steady
            ax.set_xlim(-0.5,1.5)
            ax.set_ylim(-0.5,1.5)
            # plot the points
            ax.scatter(layout[:,0], layout[:,1])
            ax.axis("off")
            ax.set_aspect(1.0)
            fig.canvas.draw()
        except: 
            pass 
            #print('Only found {}/{} iterations'.format(res.best_losses.shape[0],xs.shape[0]))
    