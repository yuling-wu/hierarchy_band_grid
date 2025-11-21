import numpy as np
import brainpy as bp
import brainpy.math as bm
from brainpy.types import Shape
from funcs import map2pi, softmax
bm.random.seed()
import jax
  

class Band_cells(bp.DynamicalSystem):
    def __init__(self, tau=0.1, size=180, J0=0.,k=5e-4,a=1/4*bm.pi, z_min=-bm.pi,z_max=bm.pi, **kwargs):
        super(Band_cells, self).__init__(**kwargs)

        self.tau = tau
        self.k = k
        self.a = a
        # feature space
        self.z_min = z_min
        self.z_max = z_max
        self.z_range = z_max - z_min
        self.x = bm.linspace(z_min, z_max, size, endpoint=False)  # The encoded feature values
        self.rho = size / self.z_range  # The neural density
        self.dx = self.z_range / size  # The stimulus density

        # variables
        self.r = bm.Variable(bm.zeros(size))
        self.u = bm.Variable(bm.zeros(size))
        self.input = bm.Variable(bm.zeros(size))
        self.center = bm.Variable(bm.zeros(1,)) # The center of the bump

        self.J = J0*self.Jc()  # The connection strength
        self.conn_mat = self.make_conn() # The connection matrix

    # critical connection strength
    def Jc(self):
        Jc = bm.sqrt(8*bm.sqrt(2*bm.pi)*self.k*self.a/self.rho)
        return Jc

    # truncate the distance into the range of feature space
    def dist(self, d):
        d = bm.remainder(d, self.z_range)
        d = bm.where(d > 0.5 * self.z_range, d - self.z_range, d)
        return d
    
    # make the connection matrix
    def make_conn(self):
        dis = self.x[:, None] - self.x[None, :]
        d = self.dist(dis)
        conn = self.J * bm.exp(-0.5 * bm.square(d / self.a)) / (bm.sqrt(2 * bm.pi) * self.a)
        return conn
    
    # decode the neural activity
    def decode(self, r, axis=0):
        expo_r = bm.exp(1j * self.x) * r
        return bm.angle(bm.sum(expo_r,axis=axis) / bm.sum(r,axis=axis))
    
    # choose the activation function
    def activate(self, x):
        return bm.relu(x)

    def dist(self, d):
        d = bm.remainder(d, self.z_range)
        d = bm.where(d > 0.5 * self.z_range, d - self.z_range, d)
        return d


    def update(self,input):
        self.input.value = input
        dt = bp.share['dt']
        I_rec = bm.dot(self.conn_mat, self.r)
        self.r.value = self.activate(self.u) 
        self.u.value = self.u + (-self.u + I_rec + self.input) / self.tau * dt
        self.input[:] = 0.
        self.center[0] = self.decode(self.u)
    

# Grid cell model modules
class Grid_cells(bp.DynamicalSystem):
    def __init__(self, num, tau=0.1, k=5e-3,
                 a=bm.pi / 4, A=1., J0=1.):
        super(Grid_cells, self).__init__()

        self.num = num**2
        # dynamics parameters
        self.tau = tau  
        self.k = k  # Degree of the rescaled inhibition
        self.a = a  # Half-width of the range of excitatory connections
        self.A = A  # Magnitude of the external input
        self.J0 = J0  # maximum connection value

        # feature space
        self.x_range = 2*bm.pi
        self.x = bm.linspace(-bm.pi, bm.pi, num, endpoint=False) 
        x_grid, y_grid = bm.meshgrid(self.x,self.x)
        self.x_grid = x_grid.flatten()
        self.y_grid = y_grid.flatten()
        self.value_grid = bm.stack([self.x_grid, self.y_grid]).T
        self.rho = self.num / (self.x_range **2)  # The neural density
        self.dxy = 1 / self.rho  # The stimulus density

        # initialize conn matrix
        self.conn_mat = self.make_conn()

        # initialize dynamical variables
        self.r = bm.Variable(bm.zeros(self.num))
        self.u = bm.Variable(bm.zeros(self.num))
        self.input = bm.Variable(bm.zeros(self.num))
        self.center = bm.Variable(bm.zeros(2,))

    def circle_period(self,d):
        d = bm.where(d>bm.pi, d-2*bm.pi, d)
        d = bm.where(d<-bm.pi, d+2*bm.pi, d)
        return d

    def dist(self,d):
        d = map2pi(d)
        delta_x = d[:,0]
        # delta_y = (d[:,1] - 1/2 * d[:,0]) * 2 / bm.sqrt(3)
        delta_y = d[:,1]
        dis = bm.sqrt(delta_x**2+delta_y**2)
        return dis
    
    def make_conn(self):
        @jax.vmap
        def get_J(v):
            d = self.dist(v - self.value_grid)
            Jxx = self.J0 * bm.exp(-0.5 * bm.square(d / self.a)) / (bm.sqrt(2 * bm.pi) * self.a)
            return Jxx
        return get_J(self.value_grid)

    def get_center(self):
        exppos_x = bm.exp(1j * self.x_grid)
        exppos_y = bm.exp(1j * self.y_grid)
        r = bm.where(self.r>bm.max(self.r)*0.1, self.r, 0)
        self.center[0] = bm.angle(bm.sum(exppos_x * r))
        self.center[1] = bm.angle(bm.sum(exppos_y * r))


    def reset_state(self):
        self.r.value = bm.Variable(bm.zeros(self.num))
        self.u.value = bm.Variable(bm.zeros(self.num))
        self.input.value = bm.Variable(bm.zeros(self.num))
        self.center.value = 2 * bm.pi * bm.random.rand(2) - bm.pi

    def update(self, input):
        self.input = input
        Irec = bm.matmul(self.conn_mat, self.r)
        # Update neural state
        du = (-self.u + Irec+ self.input) / self.tau * bm.dt
        u = self.u + du
        self.u.value = bm.where(u > 0, u, 0)
        r1 = bm.square(self.u)
        r2 = 1.0 + self.k * bm.sum(r1)
        self.r.value = r1 / r2
        self.get_center()



class Path_integration_module(bp.DynamicalSystem):
    def __init__(self, angle, spacing, size=180, z_min=-bm.pi,z_max=bm.pi, place_center=bm.zeros((900,2))):
        super(Path_integration_module, self).__init__()
        self.size = size # The number of neurons in each neuron group except DN

        # feature space
        self.z_min = z_min
        self.z_max = z_max
        self.z_range = z_max - z_min
        self.x = bm.linspace(z_min, z_max, size,endpoint=False)  # The encoded feature values
        self.rho = size / self.z_range  # The neural density
        self.dx = self.z_range / size  # The stimulus density
        self.spacing = spacing
        self.angle = angle
        self.proj_k_x = bm.array([bm.cos(angle+np.pi/3), bm.sin(angle+np.pi/3)]) * 2 * bm.pi/spacing
        self.proj_k_y = bm.array([bm.cos(angle), bm.sin(angle)]) * 2 * bm.pi/spacing

        # shifts
        self.phase_shift = 1/9*bm.pi*0.76 # the shift of the connection from PEN to EPG

        # neurons
        self.Band_x_l = Band_cells(size=size) 
        self.Band_x_r = Band_cells(size=size) 
        self.Band_y_l = Band_cells(size=size)
        self.Band_y_r = Band_cells(size=size)
        self.Grid_cells = Grid_cells(num=int(size/6))

        self.center_ideal = bm.Variable(bm.zeros(1)) # The center of v-
        self.center = bm.Variable(bm.zeros(1)) # The center of v-
        self.v_x = bm.Variable(bm.zeros(1)) # The x component of velocity
        self.v_y = bm.Variable(bm.zeros(1)) # The y component of velocity
        self.place_center = place_center
        self.make_conn_b2g()
        self.make_Wg2p()
        self.gain = 0.2
    
    def make_Wg2p(self):
        phase_place = self.Postophase(self.place_center)
        phase_grid = self.Grid_cells.value_grid
        d = phase_place[:,np.newaxis,:] - phase_grid[np.newaxis,:,:]
        d = map2pi(d)
        delta_x = d[:,:,0] 
        delta_y = (d[:,:,1] - 1/2 * d[:,:,0]) * 2 / bm.sqrt(3)
        # delta_x = d[:,:,0] + d[:,:,1]/2
        # delta_y = d[:,:,1] * bm.sqrt(3) / 2
        dis = bm.sqrt(delta_x**2+delta_y**2)
        Wg2p = bm.exp(-0.5 * bm.square(dis / self.Grid_cells.a)) / (bm.sqrt(2 * bm.pi) * self.Grid_cells.a)
        self.Wg2p = Wg2p

    def Postophase(self,pos):
        phase_x = bm.mod(bm.dot(pos,self.proj_k_x),2*bm.pi)-bm.pi
        phase_y = bm.mod(bm.dot(pos,self.proj_k_y),2*bm.pi)-bm.pi
        Phase = bm.array([phase_x, phase_y]).transpose()
        return Phase
    
    def get_input_grid(self, Phase):
        dis = self.dist(Phase - self.Grid_cells.value_grid)
        input = bm.exp(-0.5 * bm.square(dis / self.Grid_cells.a)) / (bm.sqrt(2 * bm.pi) * self.Grid_cells.a)
        return input
    
    def get_input_band(self, phase):
        dis = map2pi(phase - self.Band_x_l.x)
        input = bm.exp(-0.5 * bm.square(dis / self.Band_x_l.a)) / (bm.sqrt(2 * bm.pi) * self.Band_x_l.a)
        return input
    
    def make_conn_b2g(self):
        value_grid = self.Grid_cells.value_grid
        grid_x = value_grid[:, 0]
        grid_y = value_grid[:, 1]
        value_band_x = self.Band_x_l.x
        value_band_y = self.Band_y_l.x
        dis_x_l = map2pi(grid_x[:,None] - value_band_x[None,:] + self.phase_shift) 
        dis_x_r = map2pi(grid_x[:,None] - value_band_x[None,:] - self.phase_shift) 
        dis_y_l = map2pi(grid_y[:,None] - value_band_y[None,:] + self.phase_shift) 
        dis_y_r = map2pi(grid_y[:,None] - value_band_y[None,:] - self.phase_shift) 

        self.conn_x_l = bm.exp(-0.5 * bm.square(dis_x_l / self.Band_x_l.a)) / (bm.sqrt(2 * bm.pi) * self.Band_x_l.a)
        self.conn_x_r = bm.exp(-0.5 * bm.square(dis_x_r / self.Band_x_l.a)) / (bm.sqrt(2 * bm.pi) * self.Band_x_l.a)
        self.conn_y_l = bm.exp(-0.5 * bm.square(dis_y_l / self.Band_y_l.a)) / (bm.sqrt(2 * bm.pi) * self.Band_y_l.a)
        self.conn_y_r = bm.exp(-0.5 * bm.square(dis_y_r / self.Band_y_l.a)) / (bm.sqrt(2 * bm.pi) * self.Band_y_l.a)

    
    def dist(self,d):
        d = map2pi(d)
        delta_x = d[:,0]
        # delta_y = (d[:,1] - 1/2 * d[:,0]) * 2 / bm.sqrt(3)
        delta_y = d[:,1]
        dis = bm.sqrt(delta_x**2+delta_y**2)
        return dis

    def update(self,velocity,loc,loc_input_stre):
        self.v_x[0] = bm.dot(velocity, self.proj_k_x)
        self.v_y[0] = bm.dot(velocity, self.proj_k_y)
        # location input
        Phase_loc = self.Postophase(loc)
        loc_input = loc_input_stre * self.get_input_grid(Phase_loc)
        # # Input from four modules of band cells
        # ll_input = self.conn_x_l @ self.Band_x_l.r + self.conn_y_l @ self.Band_y_l.r
        # ll_input = bm.where(ll_input>bm.max(ll_input)/2, ll_input-bm.max(ll_input)/2, 0)
        # lr_input = self.conn_x_r @ self.Band_x_r.r + self.conn_y_l @ self.Band_y_l.r
        # lr_input = bm.where(lr_input>bm.max(lr_input)/2, lr_input-bm.max(lr_input)/2, 0)
        # rl_input = self.conn_x_l @ self.Band_x_l.r + self.conn_y_r @ self.Band_y_r.r
        # rl_input = bm.where(rl_input>bm.max(rl_input)/2, rl_input-bm.max(rl_input)/2, 0)
        # rr_input = self.conn_x_r @ self.Band_x_r.r + self.conn_y_r @ self.Band_y_r.r
        # rr_input = bm.where(rr_input>bm.max(rr_input)/2, rr_input-bm.max(rr_input)/2, 0)


        phase_x_l = map2pi(self.Band_x_l.center[0]-self.phase_shift)
        phase_x_r = map2pi(self.Band_x_r.center[0]+self.phase_shift)
        phase_y_l = map2pi(self.Band_y_l.center[0]-self.phase_shift)
        phase_y_r = map2pi(self.Band_y_r.center[0]+self.phase_shift)
        Phase_ll = bm.array([phase_x_l, phase_y_l]).transpose()
        Phase_lr = bm.array([phase_x_l, phase_y_r]).transpose()
        Phase_rl = bm.array([phase_x_r, phase_y_l]).transpose()
        Phase_rr = bm.array([phase_x_r, phase_y_r]).transpose()
        ll_input = self.get_input_grid(Phase_ll) * (2 * self.gain - self.v_x - self.v_y) 
        lr_input = self.get_input_grid(Phase_lr) * (2 * self.gain - self.v_x + self.v_y) 
        rl_input = self.get_input_grid(Phase_rl) * (2 * self.gain + self.v_x - self.v_y) #/ bm.sqrt(3)
        rr_input = self.get_input_grid(Phase_rr) * (2 * self.gain + self.v_x + self.v_y) #/ bm.sqrt(3)
        # total input to grid cells
        grid_input = loc_input + (ll_input + lr_input + rl_input + rr_input) * 12.1
        self.Grid_cells.update(input=grid_input)

        # update band cells
        Phase_grid = self.Grid_cells.center
        phase_x = map2pi(Phase_grid[0])
        phase_y = map2pi(Phase_grid[1])
        Band_input_x = self.get_input_band(phase_x) 
        Band_input_y = self.get_input_band(phase_y)
        self.Band_x_l.update(input=Band_input_x)
        self.Band_x_r.update(input=Band_input_x)
        self.Band_y_l.update(input=Band_input_y)
        self.Band_y_r.update(input=Band_input_y)
        grid_fr = self.Grid_cells.r
        # self.grid_output = bm.dot(self.Wg2p, grid_fr-bm.max(grid_fr)/2)
        self.grid_output = bm.dot(self.Wg2p, grid_fr)


class Hierarchical_network(bp.DynamicalSystemNS):
    def __init__(self, num_module, num_place):
        super(Hierarchical_network, self).__init__()
        self.num_module = num_module
        self.num_place = num_place ** 2
        # randomly sample num_place place field centers from a square arena (5m x 5m)
        x = bm.linspace(0,5,num_place)
        X, Y =  bm.meshgrid(x,x)
        self.place_center = bm.stack([X.flatten(), Y.flatten()]).T
        # self.place_center = 5 * bm.random.rand(num_place,2) 

        # load heatmaps_grid from heatmaps_grid.npz
        # data = np.load('heatmaps_grid.npz', allow_pickle=True)
        # heatmaps_grid = data['heatmaps_grid']
        # print(heatmaps_grid.shape)

        MEC_model_list = bm.NodeList([])
        # self.W_g2p_list = []
        spacing = np.linspace(2.5, 3.5, num_module) 
        angle = np.linspace(0, np.pi, num_module, endpoint=False)
        for i in range(num_module):
            MEC_model_list.append(Path_integration_module(spacing=spacing[i], angle=angle[i], place_center=self.place_center))
        self.MEC_model_list = MEC_model_list
        self.place_fr = bm.Variable(bm.zeros(self.num_place))
        self.grid_fr = bm.Variable(bm.zeros((num_module,30**2)))
        self.band_x_fr = bm.Variable(bm.zeros((num_module,180)))
        self.band_y_fr = bm.Variable(bm.zeros((num_module,180)))
        self.decoded_pos = bm.Variable(bm.zeros((2)))


    def update(self,velocity,loc,loc_input_stre=0.):
        grid_output = bm.zeros(self.num_place)
        for i in range(self.num_module):
            # update the band cell module
            self.MEC_model_list[i].update(velocity=velocity, loc=loc, loc_input_stre=loc_input_stre)
            self.grid_fr[i] = self.MEC_model_list[i].Grid_cells.u
            self.band_x_fr[i] = self.MEC_model_list[i].Band_x_l.r
            self.band_y_fr[i] = self.MEC_model_list[i].Band_y_l.r
            grid_output_module = self.MEC_model_list[i].grid_output
            # W_g2p = self.W_g2p_list[i]
            # grid_fr = self.MEC_model_list[i].Grid_cell.r
            # grid_output_module = bm.dot(W_g2p, grid_fr)
            grid_output += grid_output_module
        # update the place cell module
        grid_output = bm.where(grid_output > 0, grid_output, 0)
        u_place = bm.where(grid_output > bm.max(grid_output)/2, grid_output-bm.max(grid_output)/2, 0)
        # grid_output = grid_output**2/(1+bm.sum(grid_output**2))
        # max_id = bm.argmax(grid_output)
        # center = self.place_center[max_id]
        center = bm.sum(self.place_center * u_place[:,np.newaxis], axis=0)/(1e-5+bm.sum(u_place))
        self.decoded_pos = center
        self.place_fr = u_place ** 2 / (1+bm.sum(u_place**2))
        # self.place_fr = softmax(grid_output)
        

