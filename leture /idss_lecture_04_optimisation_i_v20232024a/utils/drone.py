import utils.transformations as tr
import utils.tkanvas as tkanvas
import numpy as np

def transform(pts, m):    
    transformed = pts @ m.T
    return transformed

def pdivide(pts):        
    return (pts[:,:].T / pts[:,-1]).T

def viewport(w,h):
    return np.array([[w/2, 0, 0, w/2],
                    [0, h/2, 0, h/2],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0]])

class View3D:
    def __init__(self, w, h):
        self.viewport = viewport(w,h)
        self.model = np.eye(4)
        self.camera = np.eye(4)
                
    def project(self, pts):
        hpts = np.concatenate((pts, np.ones(pts.shape[0])[:,None]), axis=1)
        return transform(pdivide(transform(transform(hpts, self.model), self.camera)), self.viewport)[:,0:2]
    
    def scale(self, scale):
        self.model = self.model @ (np.eye(4)*scale)
        
    def rotate(self, angle, axis):
        self.model =  self.model @ tr.rotation_matrix(np.radians(angle), axis) 


class DroneViewer:

    def __init__(self, flight_path, reference, rate=50, view_rotate=0.2):        
        self.drone_pts = np.array([[-0.1,-0.1,0], [-0.1, 0.1, 0], [0.1,0.1,0], [0.1, -0.1, 0]]) * np.array([3,4,3])
        self.ground_pts = np.array([[-10,-10,0], [-10, 10, 0], [10,10,0], [10, -10, 0]]) * np.array([2,2,2])
        

        self.t = 0
        self.flight_path = flight_path
        self.reference = reference
        self.rate = rate
        self.view_rotate = view_rotate
        canvas = tkanvas.TKanvas(w=800, h=600, tick_fn=self.rotate, draw_fn=self.draw)
        self.v = View3D(canvas.w, canvas.h)
        self.v.camera[3,2] = -1
        self.v.camera[0,3] = 0
        self.v.camera[1,3] = 0
        self.v.model[2,3] = -2
        self.v.model[1,3] = -1
        self.v.model[0,0] = 0.15
        self.v.model[1,1] = 0.15
        self.v.model[2,2] = 0.15

        self.v.rotate(30, (1,0,0))
        
    def rotate(self,dt):
        self.v.rotate(self.view_rotate, (0,0,1))
    
    def draw(self, kanvas):
        
        drone = self.flight_path[self.t,:][None,:]
        ref = self.reference[self.t,:][None,:]
        operator = np.array([[0,0,0], 
                        [0,0,1.8]])
        
        if self.t < len(self.flight_path)-1:
            self.t += 1
        
        # drawing code
        # background...
        kanvas.clear()
        kanvas.rectangle(0,0,kanvas.w,kanvas.h,fill='darkgreen')
        
        # labels
        kanvas.text(20,20, text="%.2f seconds" % (self.t/self.rate), fill='white', anchor='w')     
        if drone[0,2]<0.05:
            kanvas.text(20,50, text="[LANDED]", fill='orange', anchor='w')     
            
        # ground area
        ground_poly = kanvas.polygon(self.v.project(self.ground_pts), fill='green')    
        
        # drone + shadow
        drone_shadow = drone * np.array([1,1,0])
        kanvas.polygon(self.v.project(self.drone_pts+drone_shadow), fill='darkgreen')                     
        kanvas.polygon(self.v.project(self.drone_pts+drone), fill='red')     
        drone_pos = self.v.project(drone)
        ref_pos = self.v.project(ref)
        ctr_pos = self.v.project(operator)
        
        path = self.v.project(self.reference[0:self.t])
        shadow_path = self.v.project(self.reference[0:self.t]*np.array([1,1,0]))
        kanvas.polygon(path, outline='blue', fill='')     
        kanvas.polygon(shadow_path, outline='darkgreen', fill='')  
        kanvas.circle(drone_pos[0,0], drone_pos[0,1], 4, fill='orange')     
        kanvas.circle(ref_pos[0,0], ref_pos[0,1], 4, fill='blue')     
        
        kanvas.line(ctr_pos[0,0], ctr_pos[0,1], ctr_pos[1,0], ctr_pos[1,1], fill='white')     
        kanvas.text(drone_pos[0,0]+20, drone_pos[0,1], text="Drone", fill="white")  
        kanvas.text(ref_pos[0,0]+20, ref_pos[0,1], text="Reference", fill="white")  
        

def pid_controller(vec, n_dim=3):
    p,i,d = tuple(vec)
    i_state = np.zeros(n_dim)
    old_error = np.zeros(n_dim)    
    def controller(state, reference):  
        nonlocal i_state
        error = reference - state
        p_component = p * error
        i_state +=  error        
        deriv = error - old_error
        old_error[:] = error        
        return p * error + i * i_state + d * deriv
    return controller
    
def simulate(controller, reference, rate=50):
    ## If you are looking at this and thinking "this is totally unrealistic",
    ## you are correct. It isn't very realistic. 
    
    ts = np.arange(len(reference)) / rate
    disturbance = np.array([np.cos(ts*0.035), np.sin(ts*0.02), np.cos(ts*0.05)]).T + np.array([-0.1,4,0.0])  

    x = np.zeros(9,)
    dt = 0.002    
    drone_thrust = 55
    noise = 15
    air_resistance = 0.99
    gravity = 0.08
    # define the dynamics, observation, control and disturbance matrices
    dynamics = np.diag(np.ones(9), 0) + np.diag(np.ones(6), 3)*dt
        
    
    observe_matrix = np.array([[1,0,0,0,0,0,0,0,0], 
                               [0,1,0,0,0,0,0,0,0], 
                               [0,0,1,0,0,0,0,0,0], ]).T
    
    c_gain = 0.1
    control_matrix = np.array([[0,0,0, c_gain,0,0, 0,0,0], 
                               [0,0,0, 0,c_gain,0, 0,0,0],
                               [0,0,0, 0,0,c_gain, 0,0,0]
                              ]).T
    
    d_gain = 0.6
    disturbance_matrix = np.array([[0,0,0,0,0,d_gain,0,0,0], 
                                   [0,0,0,d_gain,0,0,0,0,0],
                                    [0,0,0,0,d_gain,0,0,0,0]
                                  ]).T
    xs = []
    for i in range(len(reference)):
        
        inp = controller(x @ observe_matrix, reference[i,:])
        # limit thrust
        inp = np.clip(inp,-drone_thrust,drone_thrust)
        
        d = disturbance[i] + np.random.normal(0,noise, disturbance[i].shape)
        # no disturbance on ground
        if x[2]<0.05:
            d *= 0
            x[3:5] *= 0.2 # extreme drag on ground
            x[2] = np.maximum(x[2],0)
        x = dynamics @ x + control_matrix @ inp + disturbance_matrix @  d
        
            
        x[3:6] *= air_resistance # air resistance
        x[8] -= gravity # gravity
        xs.append(x)
    return np.array(xs)[:,0:3]