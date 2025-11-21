import numpy as np
import brainpy as bp
import brainpy.math as bm
from brainpy.types import Shape
from funcs import map2pi, softmax
bm.random.seed()
import jax


# %matplotlib qt5

# define neuron model with gaussian recurrent connection
class GaussRecUnits(bp.dyn.NeuDyn):
    def __init__(self, size: Shape,tau=1.,J0=1.1,k=5e-4,a=2/9*bm.pi,z_min=-bm.pi,z_max=bm.pi,noise=2.):
        super().__init__(size=size)
        self.tau = tau  # The time constant
        self.k = k  # The inhibition strength
        self.a = a # The width of the Gaussian connection
        self.noise_0 = noise # The noise level

        # feature space
        self.z_min = z_min
        self.z_max = z_max
        self.z_range = z_max - z_min
        self.x = bm.linspace(z_min, z_max, size, endpoint=False)  # The encoded feature values
        self.rho = size / self.z_range  # The neural density
        self.dx = self.z_range / size  # The stimulus density

        self.J = J0*self.Jc()  # The connection strength
        self.conn_mat = self.make_conn() # The connection matrix

        # variables
        self.r = bm.Variable(bm.zeros(size)) # The neural firing rate
        self.u = bm.Variable(bm.zeros(size)) # The neural synaptic input
        self.input = bm.Variable(bm.zeros(size)) # The external input
        self.center = bm.Variable(bm.zeros(1,)) # The center of the bump

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
    
    # Initialize the neural activity
    def initialze(self):
        self.u = 10.*bm.exp(-0.5*bm.square((self.x-0)/self.a))/(bm.sqrt(2*bm.pi)*self.a)
        self.r = 30.*bm.exp(-0.5*bm.square((self.x-0)/self.a))/(bm.sqrt(2*bm.pi)*self.a)

    # decode the neural activity
    def decode(self, r, axis=0):
        expo_r = bm.exp(1j * self.x) * r
        return bm.angle(bm.sum(expo_r,axis=axis) / bm.sum(r,axis=axis))
    
    # update the neural activity
    def update(self, input):
        self.input.value = input
        dt = bp.share['dt']
        r1 = bm.square(self.u)
        r2 = 1.0 + self.k * bm.sum(r1)
        self.r.value = r1 / r2
        Irec = bm.dot(self.conn_mat, self.r)
        self.u.value = self.u + (-self.u + Irec + self.input ) / self.tau * dt
        self.input[:] = 0.
        self.center[0] = self.decode(self.u)
        
    

# define neuron model with non-recurrent connection
class NonRecUnits(bp.dyn.NeuDyn):
    def __init__(self, size: Shape, tau=0.1, z_min=-bm.pi, z_max=bm.pi, noise=2.):
        super().__init__(size=size)
        self.tau = tau
        self.noise_0 = noise

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
        self.r.value = self.activate(self.u) + self.noise_0 * bm.random.randn(self.num)
        self.u.value = self.u + (-self.u + self.input) / self.tau * dt
        self.input[:] = 0.
        return self.r
    

# the intact networks contains a group of EPG neurons (recurrent units), two P-EN neurons (non-recurrent units), one group of 
        # FC2 (recurrent units), two PFL3 (non-recurrent units) and two DN neurons (non-recurrent units)

class Band_cell_module(bp.DynamicalSystem):
    def __init__(self, angle, spacing, size=180, z_min=-bm.pi,z_max=bm.pi,  noise=2., **kwargs):
        super(Band_cell_module, self).__init__(**kwargs)
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
        self.proj_k = bm.array([bm.cos(angle-np.pi/2), bm.sin(angle-np.pi/2)]) * 2 * bm.pi/spacing

        # shifts
        self.phase_shift = 1/9*bm.pi*0.76 # the shift of the connection from PEN to EPG
        # self.PFL3_shift = 3/8*bm.pi # the shift of the connection from EPG to PFL3
        # self.PEN_shift_num = int(self.PEN_shift / self.dx) # the number of interval shifted
        # self.PFL3_shift_num = int(self.PFL3_shift / self.dx) # the number of interval shifted

        # neurons
        self.Band_cells = GaussRecUnits(size=size, noise=noise) #heading direction
        self.left = NonRecUnits(size=size, noise=noise)
        self.right = NonRecUnits(size=size, noise=noise)
        self.center_ideal = bm.Variable(bm.zeros(1)) # The center of v-
        self.center = bm.Variable(bm.zeros(1)) # The center of v-

        #weights
        self.w_L2S = 0.2
        self.w_S2L = 1.
        self.gain = 0.5
        self.synapses()
        # init heading direction
        self.Band_cells.initialze()


    def Postophase(self,pos):
        phase = bm.mod(bm.dot(pos,self.proj_k),2*bm.pi)-bm.pi
        return phase
    
    def get_stimulus_by_pos(self, pos):
        phase = self.Postophase(pos)
        d = self.dist(phase - self.x)
        return bm.exp(-0.25 * bm.square(d / self.Band_cells.a))
    
    # define the synapses
    def synapses(self):
        self.W_PENl2EPG = self.w_S2L*self.make_conn(self.phase_shift)
        self.W_PENr2EPG = self.w_S2L*self.make_conn(-self.phase_shift)
        # synapses
        self.syn_Band2Left = bp.dnn.OneToOne(self.size,self.w_L2S)
        self.syn_Band2Right = bp.dnn.OneToOne(self.size,self.w_L2S)
        self.syn_Left2Band = bp.dnn.Linear(self.size,self.size,self.W_PENl2EPG)
        self.syn_Right2Band = bp.dnn.Linear(self.size,self.size,self.W_PENr2EPG)

    # move the heading direction representation (for testing)
    def move_heading(self,shift):
        self.Band_cells.r.value = bm.roll(self.Band_cells.r,shift)
        self.Band_cells.u.value = bm.roll(self.Band_cells.u,shift)

    def get_center(self):
        exppos = bm.exp(1j * self.x)
        r = self.Band_cells.r
        self.center[0] = bm.angle(bm.sum(exppos * r))

    def dist(self, d):
        d = bm.remainder(d, self.z_range)
        d = bm.where(d > 0.5 * self.z_range, d - self.z_range, d)
        return d

    def make_conn(self,shift):
        d = self.dist(self.x[:,None]-self.x[None,:] + shift)
        conn =  bm.exp(-0.5 * bm.square(d / self.Band_cells.a)) / (bm.sqrt(2 * bm.pi) * self.Band_cells.a)
        return conn

    def update(self,velocity,loc,loc_input_stre):
        # location input
        loc_input = self.get_stimulus_by_pos(loc) * loc_input_stre

        v_phi = bm.dot(velocity, self.proj_k)
        center_ideal = self.center_ideal[0] + v_phi * bp.share['dt']
        self.center_ideal[0] = map2pi(center_ideal)
        # EPG output last time step
        Band_output = self.Band_cells.r
        # PEN input
        left_input = self.syn_Band2Left(Band_output)
        right_input = self.syn_Band2Right(Band_output)
        # PEN output and gain
        self.left.update(left_input)
        self.right.update(right_input)
        self.left.r.value = (self.gain+v_phi)*self.left.r
        self.right.r.value = (self.gain-v_phi)*self.right.r
        # EPG input
        Band_input = self.syn_Left2Band(self.left.r) + self.syn_Right2Band(self.right.r)
        # EPG output
        self.Band_cells.update(Band_input + loc_input)
        # self.Band_cells.update(loc_input)
        self.get_center()
    
    def reset(self):
        self.left.u.value = bm.zeros(self.size)
        self.right.u.value = bm.zeros(self.size)
    # FC2 (recurrent units), two PFL3 (non-recurrent units) and two DN neurons (non-recurrent units)



# Grid cell model modules
class Grid_cell(bp.DynamicalSystem):
    def __init__(self, num, angle, spacing, tau=0.1, tau_v=10.,  k=5e-3,
                 a=bm.pi / 9, A=1., J0=1., mbar=1.):
        super(Grid_cell, self).__init__()

        self.num = num**2
        # dynamics parameters
        self.tau = tau  # The synaptic time constant
        self.tau_v = tau_v
        # self.w_max = w_max
        self.ratio = bm.pi*2/spacing
        self.k = k  # Degree of the rescaled inhibition
        self.a = a  # Half-width of the range of excitatory connections
        self.A = A  # Magnitude of the external input
        self.J0 = J0  # maximum connection value
        self.m = mbar * tau / tau_v
        self.angle = angle

        # feature space
        self.x_range = 2*bm.pi
        self.x = bm.linspace(-bm.pi, bm.pi, num, endpoint=False) 
        x_grid, y_grid = bm.meshgrid(self.x,self.x)
        self.x_grid = x_grid.flatten()
        self.y_grid = y_grid.flatten()
        self.value_grid = bm.stack([self.x_grid, self.y_grid]).T
        self.rho = self.num / (self.x_range **2)  # The neural density
        self.dxy = 1 / self.rho  # The stimulus density
        self.coor_transform = bm.array([[1 , -1/bm.sqrt(3)],[0, 2/bm.sqrt(3)]])
        self.rot = bm.array([[bm.cos(self.angle), -bm.sin(self.angle)],[bm.sin(self.angle), bm.cos(self.angle)]])


        # initialize conn matrix
        self.conn_mat = self.make_conn()

        # initialize dynamical variables
        self.r = bm.Variable(bm.zeros(self.num))
        self.u = bm.Variable(bm.zeros(self.num))
        self.v = bm.Variable(bm.zeros(self.num))
        self.input = bm.Variable(bm.zeros(self.num))
        self.center = bm.Variable(bm.zeros(2,))

        # 定义积分器
        self.integral = bp.odeint(method='exp_euler', f=self.derivative)

    def circle_period(self,d):
        d = bm.where(d>bm.pi, d-2*bm.pi, d)
        d = bm.where(d<-bm.pi, d+2*bm.pi, d)
        return d
    
    # def dist(self,d):
    #     d = map2pi(d)
    #     delta_x = d[:,0] # - d[:,1]/bm.sqrt(3)
    #     delta_y = d[:,1] # * 2/bm.sqrt(3)
    #     dis = bm.sqrt(delta_x**2+delta_y**2)
    #     return dis
    # def dist(self,d):
    #     d = self.circle_period(d)
    #     dis = bm.matmul(self.coor_transform, bm.transpose(d)).T
    #     delta_x = dis[:, 0]
    #     delta_y = dis[:, 1]
    #     dis = bm.sqrt(delta_x ** 2 + delta_y ** 2)
    #     return dis



    def dist(self,d):
        d = map2pi(d)
        delta_x = d[:,0]
        delta_y = (d[:,1] - 1/2 * d[:,0]) * 2 / bm.sqrt(3)
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

    @property
    def derivative(self):
        du = lambda u, t, Irec: (-u + Irec + self.input - self.v) / self.tau
        dv = lambda v, t: (-v + self.m * self.u) / self.tau_v
        return bp.JointEq([du, dv])

    def reset_state(self):
        self.r.value = bm.Variable(bm.zeros(self.num))
        self.u.value = bm.Variable(bm.zeros(self.num))
        self.v.value = bm.Variable(bm.zeros(self.num))
        self.input.value = bm.Variable(bm.zeros(self.num))
        self.center.value = 2 * bm.pi * bm.random.rand(2) - bm.pi

    def update(self, input):
        self.input = input
        Irec = bm.matmul(self.conn_mat, self.r)
        # Update neural state
        u, v = self.integral(self.u, self.v, bp.share['t'], Irec, bm.dt)
        self.u.value = bm.where(u > 0, u, 0)
        self.v.value = v
        r1 = bm.square(self.u)
        r2 = 1.0 + self.k * bm.sum(r1)
        self.r.value = r1 / r2
        self.get_center()


class Path_integration_module(bp.DynamicalSystemNS):
    def __init__(self, spacing, angle, place_center=10 * bm.random.rand(512,2)):
        super(Path_integration_module, self).__init__()
        self.band_cell_x = Band_cell_module(angle=angle, spacing=spacing, noise=0.)
        self.band_cell_y = Band_cell_module(angle=angle+np.pi/3, spacing=spacing, noise=0.)
        self.band_cell_z = Band_cell_module(angle=angle+np.pi/3*2, spacing=spacing, noise=0.)
        self.Grid_cell = Grid_cell(num=20, angle=angle, spacing=spacing)
        self.proj_k_x = self.band_cell_x.proj_k
        self.proj_k_y = self.band_cell_y.proj_k
        self.place_center = place_center
        self.make_conn()
        self.make_Wg2p()
        self.num_place = place_center.shape[0]
        self.grid_output = bm.Variable(bm.zeros(self.num_place))
        self.coor_transform = bm.array([[1 , -1/bm.sqrt(3)],[0, 2/bm.sqrt(3)]])
            
    def Postophase(self,pos):
        phase_x = bm.mod(bm.dot(pos,self.proj_k_x),2*bm.pi)-bm.pi
        phase_y = bm.mod(bm.dot(pos,self.proj_k_y),2*bm.pi)-bm.pi
        Phase = bm.array([phase_x, phase_y]).transpose()
        return Phase
    
    def make_Wg2p(self):
        phase_place = self.Postophase(self.place_center)
        phase_grid = self.Grid_cell.value_grid
        d = phase_place[:,np.newaxis,:] - phase_grid[np.newaxis,:,:]
        d = map2pi(d)
        delta_x = d[:,:,0] 
        delta_y = (d[:,:,1] - 1/2 * d[:,:,0]) * 2 / bm.sqrt(3)
        # delta_x = d[:,:,0] + d[:,:,1]/2
        # delta_y = d[:,:,1] * bm.sqrt(3) / 2
        dis = bm.sqrt(delta_x**2+delta_y**2)
        Wg2p = bm.exp(-0.5 * bm.square(dis / self.band_cell_x.Band_cells.a)) / (bm.sqrt(2 * bm.pi) * self.band_cell_x.Band_cells.a)
        self.Wg2p = Wg2p


    def dist(self,d):
        d = map2pi(d)
        delta_x = d[:,0]
        delta_y = (d[:,1] - 1/2 * d[:,0]) * 2 / bm.sqrt(3)
        dis = bm.sqrt(delta_x**2+delta_y**2)
        return dis

    def get_input(self, Phase):
        dis = self.dist(Phase - self.Grid_cell.value_grid)
        input = bm.exp(-0.5 * bm.square(dis / self.band_cell_x.Band_cells.a)) / (bm.sqrt(2 * bm.pi) * self.band_cell_x.Band_cells.a)
        return input
    
    def make_conn(self):
        value_grid = self.Grid_cell.value_grid
        band_x = self.band_cell_x.x
        band_y = self.band_cell_y.x
        band_z = self.band_cell_z.x
        J0 = self.Grid_cell.J0 * 0.1
        grid_x = value_grid[:,0]
        grid_y = value_grid[:,1]
        # Calculate the distance between each grid cell and band cell
        grid_vector = bm.zeros(value_grid.shape)
        grid_vector[:,0] = value_grid[:,0]
        grid_vector[:,1] = (value_grid[:,1] - 1/2 * value_grid[:,0]) * 2 / bm.sqrt(3)
        z_vector = bm.array([-1/2, bm.sqrt(3)/2])
        grid_phase_z = bm.dot(grid_vector, z_vector)
        dis_x = self.band_cell_x.dist(grid_x[:,None] - band_x[None,:])
        dis_y = self.band_cell_y.dist(grid_y[:,None] - band_y[None,:])
        dis_z = self.band_cell_z.dist(grid_phase_z[:,None] - band_z[None,:])
        self.W_x_grid = J0 * bm.exp(-0.5 * bm.square(dis_x / self.band_cell_x.Band_cells.a)) / (bm.sqrt(2 * bm.pi) * self.band_cell_x.Band_cells.a)
        self.W_y_grid = J0 * bm.exp(-0.5 * bm.square(dis_y / self.band_cell_y.Band_cells.a)) / (bm.sqrt(2 * bm.pi) * self.band_cell_y.Band_cells.a)
        self.W_z_grid = J0 * bm.exp(-0.5 * bm.square(dis_z / self.band_cell_z.Band_cells.a)) / (bm.sqrt(2 * bm.pi) * self.band_cell_z.Band_cells.a)

    def update(self, velocity, loc, loc_input_stre=0.):
        self.band_cell_x.update(velocity=velocity, loc=loc, loc_input_stre=loc_input_stre)
        self.band_cell_y.update(velocity=velocity, loc=loc, loc_input_stre=loc_input_stre)
        self.band_cell_z.update(velocity=velocity, loc=loc, loc_input_stre=loc_input_stre)
        band_output = (self.W_x_grid @ self.band_cell_x.Band_cells.r + self.W_y_grid @ self.band_cell_y.Band_cells.r + self.W_z_grid @ self.band_cell_z.Band_cells.r)
        # band_output = (self.W_x_grid @ self.band_cell_x.Band_cells.r + self.W_y_grid @ self.band_cell_y.Band_cells.r)
        max_output = bm.max(band_output)
        band_output = bm.where(band_output > max_output/2, band_output-max_output/2, 0)
        phase_x = self.band_cell_x.center[0]
        phase_y = self.band_cell_y.center[0]
        Phase = bm.array([phase_x, phase_y]).transpose()
        # Phase = self.Postophase(loc)
        loc_input = self.get_input(Phase) * 5000
        self.Grid_cell.update(input=loc_input)
        grid_fr = self.Grid_cell.r
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
        spacing = np.linspace(2, 5, num_module) 
        for i in range(num_module):
            MEC_model_list.append(Path_integration_module(spacing=spacing[i], angle=0., place_center=self.place_center))
            # W_g2p = self.W_place2grid(heatmaps_grid[i*400:(i+1)*400])
            # self.W_g2p_list.append(W_g2p)
        self.MEC_model_list = MEC_model_list
        self.place_fr = bm.Variable(bm.zeros(self.num_place))
        self.grid_fr = bm.Variable(bm.zeros((num_module,20**2)))
        self.band_x_fr = bm.Variable(bm.zeros((num_module,180)))
        self.band_y_fr = bm.Variable(bm.zeros((num_module,180)))
        self.decoded_pos = bm.Variable(bm.zeros((2)))


    def update(self,velocity,loc,loc_input_stre=0.):
        grid_output = bm.zeros(self.num_place)
        for i in range(self.num_module):
            # update the band cell module
            self.MEC_model_list[i].update(velocity=velocity, loc=loc, loc_input_stre=loc_input_stre)
            self.grid_fr[i] = self.MEC_model_list[i].Grid_cell.u
            self.band_x_fr[i] = self.MEC_model_list[i].band_cell_x.Band_cells.r
            self.band_y_fr[i] = self.MEC_model_list[i].band_cell_y.Band_cells.r
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
        

