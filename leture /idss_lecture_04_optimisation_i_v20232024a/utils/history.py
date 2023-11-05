import numpy as np
import matplotlib.pyplot as plt

def linear_regression_plot(res, gradient, offset, x, y, opt_title=""):
    
    # generate a grid
    div = np.linspace(-5,5,50)
    mx, my = np.meshgrid(div,div)

    # we need to compute distance for *each* data point (for 20 of them)

    mmx = np.tile(mx[:,:,None], (1,len(x)))
    mmy = np.tile(my[:,:,None], (1,len(x)))

    # cost function: |y - f'(x)|^2
    cost_fn = np.sum(((x * mmy + mmx) - y)**2, axis=2)

    fig = plt.figure(figsize=(8,16))
    # plot the parameters
    ax = fig.add_subplot(3,1,1)       
    ax.contour(my,mx,np.log(cost_fn),20, colors='k', linewidths=0.5, alpha=0.5)
    
    ax.plot(res.all_theta[:,0], res.all_theta[:,1], 'o', label="Test points", alpha=0.1)
    ax.plot(res.best_thetas[:,0], res.best_thetas[:,1], 'o', label="Best points found", alpha=0.5)
    ax.plot(res.theta[0], res.theta[1], 'o', label="Minimum found")
    ax.plot(gradient, offset, 'o', label="True minimum")
    ax.legend()
    ax.set_frame_on(False)
    ax.set_title("Parameters")
    ax.set_xlabel("Gradient $c$")
    ax.set_ylabel("Offset $m$")
    
    # now plot the lines themselves
    ax = fig.add_subplot(3,1,2)    
    ax.scatter(x,y, label="Data")
    xs = np.linspace(np.min(x), np.max(x), 5)
    ys = xs*gradient + offset
    ax.plot(xs, xs*res.theta[0]+res.theta[1], label='Minimum found', c='C2')        
    ax.set_frame_on(False)
    for m,c in res.best_thetas:
        ax.plot(xs, xs*m+c, c='C1', alpha=0.1)
    
    ax.set_ylim(ys[0], ys[-1])   
    ax.plot(xs, ys, label='True minimum', c='r')
    ax.set_title("Predictions")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.legend()
    fig.suptitle("{opt_title} over possible line configurations".format(opt_title=opt_title))

    ax = fig.add_subplot(3,1,3)
    ax.plot(res.all_loss, label="History")
    ax.plot(res.best_iters, res.best_losses, 'o', label="Lowest so far")
    ax.legend()
    ax.set_xlabel("Iteration")
    ax.set_ylabel("$L(\\theta)$")
    ax.set_frame_on(False)
    ax.set_title("Objective function")

                 

class History:
    def __init__(self):
        self.all_theta = []
        self.all_loss = []
        self.best_thetas = []
        self.best_losses = []
        self.best_iters = []
        self.all_best = []
        self.best = np.inf
        self.best_theta = None
        self.loss_trace = []
        self.iters = 0
        
    
    def track(self, theta, loss, force=False):
        self.all_theta.append(theta)
        self.all_loss.append(loss)
        self.iters += 1
        is_best = False
        if loss<self.best or force:
            self.best = loss
            self.best_theta = theta
            self.best_losses.append(self.best)
            self.best_thetas.append(self.best_theta)
            self.best_iters.append(self.iters)
            is_best = True
        self.loss_trace.append(self.best)
        self.all_best.append(is_best)
        return is_best
    
    def finalise(self):
        self.all_theta = np.array(self.all_theta)
        self.all_loss = np.array(self.all_loss)
        self.best_losses = np.array(self.best_losses)
        self.best_thetas = np.array(self.best_thetas)
        self.best_iters = np.array(self.best_iters)
        self.theta = np.array(self.best_theta)
        self.loss = np.array(self.best)
        self.loss_trace = np.array(self.loss_trace)
        return self