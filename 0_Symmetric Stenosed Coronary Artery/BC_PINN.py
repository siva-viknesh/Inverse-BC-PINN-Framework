# ********************** PHYSICS INFORMED NEURAL NETWORK - SOLVING & MODELING 3D PDE'S ******************************* #
# Author  : SIVA VIKNESH
# Email   : siva.viknesh@sci.utah.edu / sivaviknesh14@gmail.com
# Address : SCI INSTITUTE, UNIVERSITY OF UTAH, SALT LAKE CITY, UTAH, USA
# ******************************************************************************************************************** #

# ****** IMPORTING THE NECESSARY LIBRARIES
import os
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader, TensorDataset
import vtk
from vtk.util import numpy_support as VN

''' *********************** JUMP TO THE MAIN PROGRAM TO CONTROL THE PROGRAM PARAMETERS ***************************** '''

# ************************************** WRITING THE CUSTOM FUNCTIONS ************************************************ #

def SENSOR_LOCATION(sensor_file, Nslice, x_scale, y_scale, z_scale):

	x_location = x_locations = torch.linspace(0.01, x_scale-0.01, steps = Nslice)

	reader = vtk.vtkUnstructuredGridReader()
	reader.SetFileName(sensor_file)
	reader.Update()
	
	x_data = np.zeros (Nslice)
	y_data = np.zeros (Nslice)
	z_data = np.zeros (Nslice)

	for i, xl in enumerate (x_location):

		#****** PLANE CREATION
		plane = vtk.vtkPlane()
		plane.SetOrigin(xl, 0.0, 0.0)								# LOCATION OF SLICE ALONG X DIRECTION
		plane.SetNormal(1, 0, 0)									# SLICE IN YZ PLANE 
	
		#****** SLICE THE MESH AT THE CHOSEN PLANE
		cutter = vtk.vtkCutter()
		cutter.SetCutFunction(plane)
		cutter.SetInputConnection(reader.GetOutputPort())
		cutter.Update()

		#****** INTEGRATE THE VARIABLES TO GET CENTROID
		integrate = vtk.vtkIntegrateAttributes()
		integrate.SetInputConnection(cutter.GetOutputPort())
		integrate.Update()
		x_data [i] = integrate.GetOutput().GetBounds()[0]
		y_data [i] = integrate.GetOutput().GetBounds()[2]
		z_data [i] = integrate.GetOutput().GetBounds()[4]

		x_data = (x_data/x_scale).reshape(-1, 1)
		y_data = (y_data/y_scale).reshape(-1, 1)
		z_data = (z_data/z_scale).reshape(-1, 1)

	return x_data, y_data, z_data
	
def PINN(processor, x, y, z, xup, yup, zup, xbc_wall, ybc_wall, zbc_wall, xbc_wall_up, ybc_wall_up, zbc_wall_up, x_data, y_data, 
	z_data, x_data_up, y_data_up, z_data_up, P_data, P_data_up, x_inlet, y_inlet, z_inlet, x_scale, y_scale, z_scale, P_max, 
	P_min, vel_scale, density, diffusion, Nslice, learning_rate, learn_rate_a, step_epoch, step_eph_a, decay_rate, BClearning_rate, 
	BClearn_rate_a, BCstep_epoch, BCstep_eph_a, BCdecay_rate, batchsize, epochs, epoch_offset, input_VAR, neurons, auBC, avBC, 
	awBC, apBC, au1, av1, aw1, ap1, n, path, Flag_Batch, Flag_Resume, Flag_Dyn_LR, W_NSE, W_CONT, W_BC, W_DATA, W_INLET):

	# MESH SPATIAL DATA
	x = torch.Tensor(x).to(processor)
	y = torch.Tensor(y).to(processor)
	z = torch.Tensor(z).to(processor)

	xup = torch.Tensor(xup).to(processor)
	yup = torch.Tensor(yup).to(processor)
	zup = torch.Tensor(zup).to(processor)

	# INLET WALL BOUNDARY	
	x_inlet = torch.Tensor(x_inlet).to(processor)
	y_inlet = torch.Tensor(y_inlet).to(processor)
	z_inlet = torch.Tensor(z_inlet).to(processor)

	# BC INLET PROFILE
	xbc_inlet  = np.array ([0.001 /x_scale]).reshape (-1, 1)
	ybc_inlet  = np.array ([0.000 /y_scale]).reshape (-1, 1)
	zbc_inlet  = np.array ([0.000 /z_scale]).reshape (-1, 1)

	xbc_inlet  = torch.Tensor(xbc_inlet).to(processor)
	ybc_inlet  = torch.Tensor(ybc_inlet).to(processor)
	zbc_inlet  = torch.Tensor(zbc_inlet).to(processor)

	# WALL BOUNDARY
	xbc_wall = torch.Tensor(xbc_wall).to(processor)
	ybc_wall = torch.Tensor(ybc_wall).to(processor)
	zbc_wall = torch.Tensor(zbc_wall).to(processor)

	BC_in = torch.cat((xbc_wall, ybc_wall, zbc_wall), 1).to(processor)

	xbc_wall_up = torch.Tensor(xbc_wall_up).to(processor)
	ybc_wall_up = torch.Tensor(ybc_wall_up).to(processor)
	zbc_wall_up = torch.Tensor(zbc_wall_up).to(processor)
	

	BC_in_up = torch.cat((xbc_wall_up, ybc_wall_up, zbc_wall_up), 1).to(processor)

	# SENSOR LOCATION
	x_data   = torch.Tensor(x_data).to(processor)
	y_data   = torch.Tensor(y_data).to(processor)
	z_data   = torch.Tensor(z_data).to(processor)
	P_data   = torch.Tensor(P_data).to(processor)

	x_data_up   = torch.Tensor(x_data_up).to(processor)
	y_data_up   = torch.Tensor(y_data_up).to(processor)
	z_data_up   = torch.Tensor(z_data_up).to(processor)
	P_data_up   = torch.Tensor(P_data_up).to(processor)

	a  		 = Parameter(torch.ones(1))

	train_load_up = DataLoader(TensorDataset(xup, yup, zup), batch_size=batchsize, shuffle=True,  drop_last = False)
	train_load    = DataLoader(TensorDataset(x, y, z),       batch_size=batchsize, shuffle=True,  drop_last = False)
	del x, y, z, xup, yup, zup

	# ******************************************** NEURAL NETWORK **************************************************** #

	# ADAPTIVE ACTIVATION FUNCTION
	class CUSTOM_SiLU(nn.Module):																		
		
		def __init__(self, a):						
			super().__init__()
			self.a = a

		def forward(self, x):
			output = nn.SiLU()

			return output(self.a*n*x)

	# ******* X-VELOCITY COMPONENT: u

	class UBC_NN(CUSTOM_SiLU):

		def __init__(self):
			super().__init__(a)
			self.upstream = nn.Sequential(
				nn.Linear(input_VAR,neurons),

				CUSTOM_SiLU(a = auBC[0, : ]),
				nn.Linear(neurons,neurons),

				CUSTOM_SiLU(a = auBC[1, :]),
				nn.Linear(neurons,neurons),

				CUSTOM_SiLU(a = auBC[2, :]),
				nn.Linear(neurons,neurons),

				CUSTOM_SiLU(a = auBC[3, :]),
				nn.Linear(neurons,neurons),

				CUSTOM_SiLU(a = auBC[4, :]),
				nn.Linear(neurons,neurons),

				CUSTOM_SiLU(a = auBC[5, :]),
				nn.Linear(neurons,neurons),

				CUSTOM_SiLU(a = auBC[6, :]),
				nn.Linear(neurons,neurons),

				CUSTOM_SiLU(a = auBC[7, :]),
				nn.Linear(neurons,neurons),

				CUSTOM_SiLU(a = auBC[8, :]),
				nn.Linear(neurons,neurons),

				CUSTOM_SiLU(a = auBC[9, :]),
				nn.Linear(neurons,neurons),

				CUSTOM_SiLU(a = auBC[10, :]),
				nn.Linear(neurons,1),
			)

		def forward(self, x):
			output = self.upstream(x)
			return output	

	# ****** Y-VELOCITY COMPONENT: 

	class VBC_NN(CUSTOM_SiLU):
		def __init__(self):
			super().__init__(a)
			self.upstream = nn.Sequential(
				nn.Linear(input_VAR,neurons),

				CUSTOM_SiLU(a = avBC[0, :]),
				nn.Linear(neurons,neurons),

				CUSTOM_SiLU(a = avBC[1, :]),
				nn.Linear(neurons,neurons),

				CUSTOM_SiLU(a = avBC[2, :]),
				nn.Linear(neurons,neurons),

				CUSTOM_SiLU(a = avBC[3, :]),
				nn.Linear(neurons,neurons),

				CUSTOM_SiLU(a = avBC[4, :]),
				nn.Linear(neurons,neurons),

				CUSTOM_SiLU(a = avBC[5, :]),
				nn.Linear(neurons,neurons),

				CUSTOM_SiLU(a = avBC[6, :]),
				nn.Linear(neurons,neurons),

				CUSTOM_SiLU(a = avBC[7, :]),
				nn.Linear(neurons,neurons),

				CUSTOM_SiLU(a = avBC[8, :]),
				nn.Linear(neurons,neurons),

				CUSTOM_SiLU(a = avBC[9, :]),
				nn.Linear(neurons,neurons),

				CUSTOM_SiLU(a = avBC[10,:]),
				nn.Linear(neurons,1)
			)

		def forward(self, x):
			output = self.upstream(x)
			return output			

	# ****** Z-VELOCITY COMPONENT: w

	class WBC_NN(CUSTOM_SiLU):
		def __init__(self):
			super().__init__(a)
			self.upstream = nn.Sequential(
				nn.Linear(input_VAR,neurons),

				CUSTOM_SiLU(a = awBC[0, :]),
				nn.Linear(neurons,neurons),

				CUSTOM_SiLU(a = awBC[1, :]),
				nn.Linear(neurons,neurons),

				CUSTOM_SiLU(a = awBC[2, :]),
				nn.Linear(neurons,neurons),

				CUSTOM_SiLU(a = awBC[3, :]),
				nn.Linear(neurons,neurons),

				CUSTOM_SiLU(a = awBC[4, :]),
				nn.Linear(neurons,neurons),

				CUSTOM_SiLU(a = awBC[5, :]),
				nn.Linear(neurons,neurons),

				CUSTOM_SiLU(a = awBC[6, :]),
				nn.Linear(neurons,neurons),

				CUSTOM_SiLU(a = awBC[7, :]),
				nn.Linear(neurons,neurons),

				CUSTOM_SiLU(a = awBC[8, :]),
				nn.Linear(neurons,neurons),

				CUSTOM_SiLU(a = awBC[9, :]),
				nn.Linear(neurons,neurons),

				CUSTOM_SiLU(a = awBC[10,:]),
				nn.Linear(neurons,1)
			)

		def forward(self, x):
			output = self.upstream(x)
			return output		

	# ****** PRESSURE: p
	
	class PBC_NN(CUSTOM_SiLU):
		def __init__(self):
			super().__init__(a)
			self.upstream = nn.Sequential(
				nn.Linear(input_VAR,neurons),

				CUSTOM_SiLU(a = apBC[0, :]),
				nn.Linear(neurons,neurons),

				CUSTOM_SiLU(a = apBC[1, :]),
				nn.Linear(neurons,neurons),

				CUSTOM_SiLU(a = apBC[2, :]),
				nn.Linear(neurons,neurons),

				CUSTOM_SiLU(a = apBC[3, :]),
				nn.Linear(neurons,neurons),

				CUSTOM_SiLU(a = apBC[4, :]),
				nn.Linear(neurons,neurons),

				CUSTOM_SiLU(a = apBC[5, :]),
				nn.Linear(neurons,neurons),

				CUSTOM_SiLU(a = apBC[6, :]),
				nn.Linear(neurons,neurons),

				CUSTOM_SiLU(a = apBC[7, :]),
				nn.Linear(neurons,neurons),

				CUSTOM_SiLU(a = apBC[8, :]),
				nn.Linear(neurons,neurons),

				CUSTOM_SiLU(a = apBC[9, :]),
				nn.Linear(neurons,neurons),

				CUSTOM_SiLU(a = apBC[10, :]),
				nn.Linear(neurons,1)
			)

		def forward(self, x):
			output = self.upstream(x)
			return output
	
	
	# ******* X-VELOCITY COMPONENT: u

	class U_VEL_NN(CUSTOM_SiLU):

		def __init__(self):
			super().__init__(a)
			self.upstream = nn.Sequential(
				nn.Linear(input_VAR,neurons),

				CUSTOM_SiLU(a = au1[0, : ]),
				nn.Linear(neurons,neurons),

				CUSTOM_SiLU(a = au1[1, :]),
				nn.Linear(neurons,neurons),

				CUSTOM_SiLU(a = au1[2, :]),
				nn.Linear(neurons,neurons),

				CUSTOM_SiLU(a = au1[3, :]),
				nn.Linear(neurons,neurons),

				CUSTOM_SiLU(a = au1[4, :]),
				nn.Linear(neurons,neurons),

				CUSTOM_SiLU(a = au1[5, :]),
				nn.Linear(neurons,neurons),

				CUSTOM_SiLU(a = au1[6, :]),
				nn.Linear(neurons,neurons),

				CUSTOM_SiLU(a = au1[7, :]),
				nn.Linear(neurons,neurons),

				CUSTOM_SiLU(a = au1[8, :]),
				nn.Linear(neurons,neurons),

				CUSTOM_SiLU(a = au1[9, :]),
				nn.Linear(neurons,neurons),

				CUSTOM_SiLU(a = au1[10, :]),
				nn.Linear(neurons,1),
			)

		def forward(self, x):
			output = self.upstream(x)
			return output	

	# ****** Y-VELOCITY COMPONENT: 

	class V_VEL_NN(CUSTOM_SiLU):
		def __init__(self):
			super().__init__(a)
			self.upstream = nn.Sequential(
				nn.Linear(input_VAR,neurons),

				CUSTOM_SiLU(a = av1[0, :]),
				nn.Linear(neurons,neurons),

				CUSTOM_SiLU(a = av1[1, :]),
				nn.Linear(neurons,neurons),

				CUSTOM_SiLU(a = av1[2, :]),
				nn.Linear(neurons,neurons),

				CUSTOM_SiLU(a = av1[3, :]),
				nn.Linear(neurons,neurons),

				CUSTOM_SiLU(a = av1[4, :]),
				nn.Linear(neurons,neurons),

				CUSTOM_SiLU(a = av1[5, :]),
				nn.Linear(neurons,neurons),

				CUSTOM_SiLU(a = av1[6, :]),
				nn.Linear(neurons,neurons),

				CUSTOM_SiLU(a = av1[7, :]),
				nn.Linear(neurons,neurons),

				CUSTOM_SiLU(a = av1[8, :]),
				nn.Linear(neurons,neurons),

				CUSTOM_SiLU(a = av1[9, :]),
				nn.Linear(neurons,neurons),

				CUSTOM_SiLU(a = av1[10,:]),
				nn.Linear(neurons,1)
			)

		def forward(self, x):
			output = self.upstream(x)
			return output			

	# ****** Z-VELOCITY COMPONENT: w

	class W_VEL_NN(CUSTOM_SiLU):
		def __init__(self):
			super().__init__(a)
			self.upstream = nn.Sequential(
				nn.Linear(input_VAR,neurons),

				CUSTOM_SiLU(a = aw1[0, :]),
				nn.Linear(neurons,neurons),

				CUSTOM_SiLU(a = aw1[1, :]),
				nn.Linear(neurons,neurons),

				CUSTOM_SiLU(a = aw1[2, :]),
				nn.Linear(neurons,neurons),

				CUSTOM_SiLU(a = aw1[3, :]),
				nn.Linear(neurons,neurons),

				CUSTOM_SiLU(a = aw1[4, :]),
				nn.Linear(neurons,neurons),

				CUSTOM_SiLU(a = aw1[5, :]),
				nn.Linear(neurons,neurons),

				CUSTOM_SiLU(a = aw1[6, :]),
				nn.Linear(neurons,neurons),

				CUSTOM_SiLU(a = aw1[7, :]),
				nn.Linear(neurons,neurons),

				CUSTOM_SiLU(a = aw1[8, :]),
				nn.Linear(neurons,neurons),

				CUSTOM_SiLU(a = aw1[9, :]),
				nn.Linear(neurons,neurons),

				CUSTOM_SiLU(a = aw1[10,:]),
				nn.Linear(neurons,1)
			)

		def forward(self, x):
			output = self.upstream(x)
			return output		

	# ****** PRESSURE: p
	
	class PRESS_NN(CUSTOM_SiLU):
		def __init__(self):
			super().__init__(a)
			self.upstream = nn.Sequential(
				nn.Linear(input_VAR,neurons),

				CUSTOM_SiLU(a = ap1[0, :]),
				nn.Linear(neurons,neurons),

				CUSTOM_SiLU(a = ap1[1, :]),
				nn.Linear(neurons,neurons),

				CUSTOM_SiLU(a = ap1[2, :]),
				nn.Linear(neurons,neurons),

				CUSTOM_SiLU(a = ap1[3, :]),
				nn.Linear(neurons,neurons),

				CUSTOM_SiLU(a = ap1[4, :]),
				nn.Linear(neurons,neurons),

				CUSTOM_SiLU(a = ap1[5, :]),
				nn.Linear(neurons,neurons),

				CUSTOM_SiLU(a = ap1[6, :]),
				nn.Linear(neurons,neurons),

				CUSTOM_SiLU(a = ap1[7, :]),
				nn.Linear(neurons,neurons),

				CUSTOM_SiLU(a = ap1[8, :]),
				nn.Linear(neurons,neurons),

				CUSTOM_SiLU(a = ap1[9, :]),
				nn.Linear(neurons,neurons),

				CUSTOM_SiLU(a = ap1[10, :]),
				nn.Linear(neurons,1)
			)

		def forward(self, x):
			output = self.upstream(x) 
			return output

	A_NN = CUSTOM_SiLU(a).to(processor)

	UBC = UBC_NN().to(processor)
	VBC = VBC_NN().to(processor)
	WBC = WBC_NN().to(processor)
	PBC = PBC_NN().to(processor)

	# NEURAL NETWORK
	U_NN = U_VEL_NN().to(processor)
	V_NN = V_VEL_NN().to(processor)
	W_NN = W_VEL_NN().to(processor)
	P_NN = PRESS_NN().to(processor)

	# ****** INITIALISATION OF THE NEURAL NETWORK

	def init_normal(m):
		if isinstance(m, nn.Linear):
			nn.init.kaiming_normal_(m.weight)

	# NEURAL NETWORK
	UBC.apply(init_normal)
	VBC.apply(init_normal)
	WBC.apply(init_normal)
	PBC.apply(init_normal)

	U_NN.apply(init_normal)
	V_NN.apply(init_normal)
	W_NN.apply(init_normal)
	P_NN.apply(init_normal)

	# ADAPTIVE ACTIVATION FUNCTION
	optim_BCAAF = optim.Adam([auBC, avBC, awBC, apBC], lr = BClearn_rate_a, betas = (0.9,0.99),eps = 10**-15)

	optim_UAAF = optim.Adam([au1], lr = learn_rate_a, betas = (0.9,0.99),eps = 10**-15)
	optim_VAAF = optim.Adam([av1], lr = learn_rate_a, betas = (0.9,0.99),eps = 10**-15)
	optim_WAAF = optim.Adam([aw1], lr = learn_rate_a, betas = (0.9,0.99),eps = 10**-15)
	optim_PAAF = optim.Adam([ap1], lr = learn_rate_a, betas = (0.9,0.99),eps = 10**-15)

	# LAMBDA FUNCTION
	optim_Lambda  = optim.Adam([W_NSE, W_BC, W_DATA, W_CONT, W_INLET], lr = BClearn_rate_a, maximize=True, betas = (0.9,0.99),eps = 10**-15)

	# NEURAL NETWORKS
	optim_UBCNN = optim.Adam(UBC.parameters(),  lr=BClearning_rate, betas = (0.9,0.99),eps = 10**-15)
	optim_VBCNN = optim.Adam(VBC.parameters(),  lr=BClearning_rate, betas = (0.9,0.99),eps = 10**-15)
	optim_WBCNN = optim.Adam(WBC.parameters(),  lr=BClearning_rate, betas = (0.9,0.99),eps = 10**-15)
	optim_PBCNN = optim.Adam(PBC.parameters(),  lr=BClearning_rate, betas = (0.9,0.99),eps = 10**-15)

	optim_UNN = optim.Adam(U_NN.parameters(),  lr=learning_rate, betas = (0.9,0.99),eps = 10**-15)
	optim_VNN = optim.Adam(V_NN.parameters(),  lr=learning_rate, betas = (0.9,0.99),eps = 10**-15)
	optim_WNN = optim.Adam(W_NN.parameters(),  lr=learning_rate, betas = (0.9,0.99),eps = 10**-15)
	optim_PNN = optim.Adam(P_NN.parameters(),  lr=learning_rate, betas = (0.9,0.99),eps = 10**-15)

	if (Flag_Dyn_LR):
		# ADAPTIVE ACTIVATION FUNCTION
		scheduler_BCAAF = torch.optim.lr_scheduler.StepLR(optim_BCAAF,  step_size=BCstep_eph_a, gamma=BCdecay_rate)

		scheduler_UAAF = torch.optim.lr_scheduler.StepLR(optim_UAAF,  step_size=step_eph_a, gamma=decay_rate)
		scheduler_VAAF = torch.optim.lr_scheduler.StepLR(optim_VAAF,  step_size=step_eph_a, gamma=decay_rate)
		scheduler_WAAF = torch.optim.lr_scheduler.StepLR(optim_WAAF,  step_size=step_eph_a, gamma=decay_rate)
		scheduler_PAAF = torch.optim.lr_scheduler.StepLR(optim_PAAF,  step_size=step_eph_a, gamma=decay_rate)

		scheduler_UBCNN = torch.optim.lr_scheduler.StepLR(optim_UBCNN,  step_size=BCstep_epoch, gamma=BCdecay_rate)
		scheduler_VBCNN = torch.optim.lr_scheduler.StepLR(optim_VBCNN,  step_size=BCstep_epoch, gamma=BCdecay_rate)
		scheduler_WBCNN = torch.optim.lr_scheduler.StepLR(optim_WBCNN,  step_size=BCstep_epoch, gamma=BCdecay_rate)
		scheduler_PBCNN = torch.optim.lr_scheduler.StepLR(optim_PBCNN,  step_size=BCstep_epoch, gamma=BCdecay_rate)

		# LAMBDA FUNCTION
		scheduler_Lambda  = torch.optim.lr_scheduler.StepLR(optim_Lambda, step_size=BCstep_eph_a, gamma=BCdecay_rate)	

		# NEURAL NETWORKS
		scheduler_UNN = torch.optim.lr_scheduler.StepLR(optim_UNN,  step_size=step_epoch, gamma=decay_rate)
		scheduler_VNN = torch.optim.lr_scheduler.StepLR(optim_VNN,  step_size=step_epoch, gamma=decay_rate)
		scheduler_WNN = torch.optim.lr_scheduler.StepLR(optim_WNN,  step_size=step_epoch, gamma=decay_rate)
		scheduler_PNN = torch.optim.lr_scheduler.StepLR(optim_PNN,  step_size=step_epoch, gamma=decay_rate)	

	# ****************************************** COMPUTATION OF LOSSES *********************************************** #

	def NSE_LOSS(x, y, z, model1, model2, model3, model4):  				# NAVIER-STOKES EQUATION + CONTINUITY EQUN 

		x.requires_grad = True
		y.requires_grad = True
		z.requires_grad = True

		NSE_in = torch.cat((x, y, z),1)

		# GATE FUNCTION ALONG X DIRECTION
		u = model1(NSE_in)
		u = u.view(len(u),-1)

		v = model2(NSE_in)
		v = v.view(len(v),-1)

		w = model3(NSE_in)
		w = w.view(len(w),-1)

		P = model4(NSE_in)
		P = P.view(len(P),-1)
		
		# COMPUTING DERIVATIVES        
		du_dx  = torch.autograd.grad(u, x,     grad_outputs=torch.ones_like(x), create_graph = True, only_inputs=True)[0]
		du_dxx = torch.autograd.grad(du_dx, x, grad_outputs=torch.ones_like(x), create_graph = True, only_inputs=True)[0]

		du_dy  = torch.autograd.grad(u, y,     grad_outputs=torch.ones_like(y), create_graph = True, only_inputs=True)[0]
		du_dyy = torch.autograd.grad(du_dy, y, grad_outputs=torch.ones_like(y), create_graph = True, only_inputs=True)[0]

		du_dz  = torch.autograd.grad(u, z,     grad_outputs=torch.ones_like(z), create_graph = True, only_inputs=True)[0]
		du_dzz = torch.autograd.grad(du_dz, z, grad_outputs=torch.ones_like(z), create_graph = True, only_inputs=True)[0]

		#-------------------
		
		dv_dx  = torch.autograd.grad(v, x,     grad_outputs=torch.ones_like(x), create_graph = True, only_inputs=True)[0]
		dv_dxx = torch.autograd.grad(dv_dx, x, grad_outputs=torch.ones_like(x), create_graph = True, only_inputs=True)[0]
		
		dv_dy  = torch.autograd.grad(v, y,     grad_outputs=torch.ones_like(y), create_graph = True, only_inputs=True)[0]
		dv_dyy = torch.autograd.grad(dv_dy, y, grad_outputs=torch.ones_like(y), create_graph = True, only_inputs=True)[0]

		dv_dz  = torch.autograd.grad(v, z,     grad_outputs=torch.ones_like(z), create_graph = True, only_inputs=True)[0]
		dv_dzz = torch.autograd.grad(dv_dz, z, grad_outputs=torch.ones_like(z), create_graph = True, only_inputs=True)[0]

		#-------------------
		
		dw_dx  = torch.autograd.grad(w, x,     grad_outputs=torch.ones_like(x), create_graph = True, only_inputs=True)[0]
		dw_dxx = torch.autograd.grad(dw_dx, x, grad_outputs=torch.ones_like(x), create_graph = True, only_inputs=True)[0]
		
		dw_dy  = torch.autograd.grad(w, y,     grad_outputs=torch.ones_like(y), create_graph = True, only_inputs=True)[0]
		dw_dyy = torch.autograd.grad(dw_dy, y, grad_outputs=torch.ones_like(y), create_graph = True, only_inputs=True)[0]

		dw_dz  = torch.autograd.grad(w, z,     grad_outputs=torch.ones_like(z), create_graph = True, only_inputs=True)[0]
		dw_dzz = torch.autograd.grad(dw_dz, z, grad_outputs=torch.ones_like(z), create_graph = True, only_inputs=True)[0]

		#-------------------

		dP_dx  = torch.autograd.grad(P, x,     grad_outputs=torch.ones_like(x), create_graph = True, only_inputs=True)[0]
		dP_dy  = torch.autograd.grad(P, y,     grad_outputs=torch.ones_like(y), create_graph = True, only_inputs=True)[0]
		dP_dz  = torch.autograd.grad(P, z,     grad_outputs=torch.ones_like(z), create_graph = True, only_inputs=True)[0]

	
		XX_scale = vel_scale * (x_scale**2)
		YY_scale = vel_scale * (y_scale**2)
		ZZ_scale = vel_scale * (z_scale**2)
		UU_scale = vel_scale **2

	
		# X MOMENTUM EQUATION LOSS
		loss_1 = u*du_dx / x_scale + v*du_dy / y_scale + w*du_dz / z_scale - diffusion*( du_dxx/XX_scale  + du_dyy /YY_scale + du_dzz/ZZ_scale)+ 1/density*(dP_dx*0.50*(P_max - P_min) / (x_scale*UU_scale))  
		
		# Y MOMENTUM EQUATION LOSS
		loss_2 = u*dv_dx / x_scale + v*dv_dy / y_scale + w*dv_dz / z_scale - diffusion*( dv_dxx/ XX_scale + dv_dyy /YY_scale + dv_dzz/ZZ_scale)+ 1/density*(dP_dy*0.50*(P_max - P_min) / (y_scale*UU_scale)) 
		
		# Z MOMENTUM EQUATION LOSS
		loss_3 = u*dw_dx / x_scale + v*dw_dy / y_scale + w*dw_dz / z_scale - diffusion*( dw_dxx/ XX_scale + dw_dyy /YY_scale + dw_dzz/ZZ_scale)+ 1/density*(dP_dz*0.50*(P_max - P_min) / (z_scale*UU_scale))
		
		# CONTINUITY EQUATION LOSS
		loss_4 = (du_dx / x_scale + dv_dy / y_scale + dw_dz / z_scale) 


		loss_f    = nn.MSELoss()
		loss_NSE  = loss_f(loss_1,torch.zeros_like(loss_1))+  loss_f(loss_2,torch.zeros_like(loss_2))+  loss_f(loss_3,torch.zeros_like(loss_3)) 
		loss_CONT = loss_f(loss_4,torch.zeros_like(loss_4))  

		return loss_NSE, loss_CONT

		
	def BC_LOSS(BC_in, model1, model2, model3) :            					# BOUNDARY CONDITION LOSS

		# NO-SLIP WALL 
		out1_u = model1 (BC_in)
		out1_v = model2 (BC_in)
		out1_w = model3 (BC_in)
	
		out1_u = out1_u.view(len(out1_u), -1)
		out1_v = out1_v.view(len(out1_v), -1)  
		out1_w = out1_w.view(len(out1_w), -1)

		loss_f = nn.MSELoss()
		loss_noslip = loss_f(out1_u, torch.zeros_like(out1_u)) + loss_f(out1_v, torch.zeros_like(out1_v)) + loss_f(out1_w, torch.zeros_like(out1_w))

		return loss_noslip


	def DATA_LOSS(x_data, y_data, z_data, P_data, model):						# DATA LOSS AT THE PROBED LOCATIONS

		x_data.requires_grad = True
		y_data.requires_grad = True
		z_data.requires_grad = True

		DATA_in = torch.cat((x_data, y_data, z_data), 1)

		P_out   = model(DATA_in)
		P_out   = P_out.view(len(P_out), -1)
	
		loss_f    = nn.MSELoss()
		loss_data = loss_f (P_out, P_data)

		return loss_data

	def BC_INLET_LOSS (x, y, z):							   				# INLET PROFILE

		x.requires_grad = True
		y.requires_grad = True
		z.requires_grad = True

		INLET_in = torch.cat((x, y, z), 1)

		U = UBC(INLET_in)
		U = U.view(len(U), -1)

		V = VBC(INLET_in)
		V = V.view(len(V), -1)

		W = WBC(INLET_in)
		W = W.view(len(W), -1)

		Vm = torch.sqrt(U**2 + V**2 + W**2)

		Up = torch.ones_like(U)
		
		dVm_dx  = torch.autograd.grad(Vm, x,     grad_outputs=torch.ones_like(x), create_graph = True, only_inputs=True)[0]		
		dVm_dy  = torch.autograd.grad(Vm, y,     grad_outputs=torch.ones_like(y), create_graph = True, only_inputs=True)[0]
		dVm_dz  = torch.autograd.grad(Vm, z,     grad_outputs=torch.ones_like(z), create_graph = True, only_inputs=True)[0]

		loss_f = nn.MSELoss()
		loss1  = loss_f (dVm_dx, torch.zeros_like(dVm_dx)) + loss_f (dVm_dy, torch.zeros_like(dVm_dy)) + loss_f (dVm_dz, torch.zeros_like(dVm_dz))
		loss2  = loss_f (U, Up)

		return loss1 + loss2

	def INLET_LOSS(x_inlet, y_inlet, z_inlet):							   # INLET PROFILE

		INLET_in = torch.cat((x_inlet, y_inlet, z_inlet), 1)

		U = U_NN(INLET_in)
		U = U.view(len(U), -1)
		
		V = V_NN(INLET_in)
		V = V.view(len(V), -1)
		
		W = W_NN(INLET_in)
		W = W.view(len(W), -1)

		P = P_NN(INLET_in)
		P = P.view(len(P), -1)

		with torch.no_grad():
			UB = UBC(INLET_in).detach()
			UB = UB.view(len(UB), -1)
		
			VB = VBC(INLET_in).detach()
			VB = VB.view(len(VB), -1)
		
			WB = WBC(INLET_in).detach()
			WB = WB.view(len(WB), -1)

			PB = PBC(INLET_in).detach()
			PB = PB.view(len(PB), -1)

		loss_f     = nn.MSELoss()
		loss_inlet = loss_f (UB, U) + loss_f (VB, V) + loss_f (WB, W) + loss_f (PB, P) 

		return loss_inlet

	# ************************************* NEURAL NETWORK COMPUTATION *********************************************** #


	# ********************************** BOUNDARY CONDITION DOMAIN TRAINING  ***************************************** #

	if (Flag_Resume):
		print('READING THE TRAINED DATA OF NET I ..... \n')
		U_NN.load_state_dict(torch.load(path + "U_velocity.pt"))
		V_NN.load_state_dict(torch.load(path + "V_Velocity.pt"))
		W_NN.load_state_dict(torch.load(path + "W_Velocity.pt"))
		P_NN.load_state_dict(torch.load(path + "Pressure.pt"))

	# LOSS FUNCTIONS
	Loss_NSE   = torch.empty(size=(epoch_offset, 1))
	Loss_CONT  = torch.empty(size=(epoch_offset, 1))
	Loss_BC    = torch.empty(size=(epoch_offset, 1))
	Loss_Data  = torch.empty(size=(epoch_offset, 1))
	Loss_Inlet  = torch.empty(size=(epoch_offset, 1))

	for epoch in range (epoch_offset):
		loss_nse   = 0.
		loss_bc    = 0.
		loss_data  = 0.
		loss_cont  = 0.
		loss_inlet = 0.

		for batch_idx, (XX, YY, ZZ) in enumerate(train_load_up):

			batch_nse, batch_cont  = NSE_LOSS   (XX, YY, ZZ, UBC, VBC, WBC, PBC)
			batch_bc               = BC_LOSS    (BC_in_up, UBC, VBC, WBC)
			batch_data             = DATA_LOSS  (x_data_up, y_data_up, z_data_up, P_data_up, PBC)
			batch_inlet            = BC_INLET_LOSS (xbc_inlet, ybc_inlet, zbc_inlet)			  

			loss = loss = W_NSE*batch_nse + W_CONT*batch_cont + W_BC*batch_bc + W_DATA*batch_data + W_INLET*batch_inlet
			
			# ADAPTIVE ACTIVATION FUNCTION
			optim_BCAAF.zero_grad()

			# NEURAL NETWORK
			optim_UBCNN.zero_grad()
			optim_VBCNN.zero_grad()
			optim_WBCNN.zero_grad()
			optim_PBCNN.zero_grad()

			# LAMBDA FUNCTION
			optim_Lambda.zero_grad()

			loss.backward()
			
			with torch.no_grad():
				# ADAPTIVE ACTIVATION FUNCTION
				optim_BCAAF.step()

				# NEURAL NETWORK
				optim_UBCNN.step()
				optim_VBCNN.step()
				optim_WBCNN.step()
				optim_PBCNN.step()

				#LAMBDA FUNCTION
				optim_Lambda.step()

				loss_nse   += batch_nse
				loss_cont  += batch_cont
				loss_bc    += batch_bc
				loss_data  += batch_data
				loss_inlet += batch_inlet

		N = batch_idx + 1
		
		Loss_NSE   [epoch] = loss_nse/N
		Loss_CONT  [epoch] = loss_cont/N
		Loss_BC    [epoch] = loss_bc/N
		Loss_Data  [epoch] = loss_data/N
		Loss_Inlet [epoch] = loss_inlet/N
	

		print('TOTAL AVERAGE LOSS OF NN PER EPOCH, [EPOCH =', epoch,']: \nNSE LOSS     :', Loss_NSE[epoch].item(), '\nCONT LOSS    :', Loss_CONT[epoch].item(), 
			"\nBC LOSS      :", Loss_BC[epoch].item(), "\nDATA LOSS    :", Loss_Data[epoch].item(), "\nINLET LOSS   :", Loss_Inlet[epoch].item())
		print("LAMBDA PARAMETERS:")
		print("NSE  =", f"{W_NSE.item():10.6}", "BC   =", f"{W_BC.item():10.6}", "\nDATA =", f"{W_DATA.item():10.6}", "CONT =", f"{W_CONT.item():10.6}")
		print('LEARNING RATE:', optim_Lambda.param_groups[0]['lr'])
		print ("*"*85)
		
		# SAVE THE NETWORK DATA AND LOSS DATA FOR EVERY 100 EPOCHS
		if epoch % 100 == 0:
			# NETWORK DATA
			torch.save(PBC.state_dict(),   path + "Pressure_BC.pt"  )
			torch.save(UBC.state_dict(),   path + "UBC_velocity.pt")
			torch.save(VBC.state_dict(),   path + "VBC_Velocity.pt")
			torch.save(WBC.state_dict(),   path + "WBC_Velocity.pt")

			# LOSS DATA
			torch.save(Loss_NSE  [0:epoch],  path + "BCLoss_NSE.pt"  )
			torch.save(Loss_CONT [0:epoch],  path + "BCLoss_CONT.pt" )
			torch.save(Loss_BC   [0:epoch],  path + "BCLoss_BC.pt"   )
			torch.save(Loss_Data [0:epoch],  path + "BCLoss_Data.pt" )
			torch.save(Loss_Inlet [0:epoch], path + "BCLoss_Inlet.pt")
			
			# ADAPTIVE ACTIVATION FUNCTION
			torch.save(auBC,   path + "BCAAF_AU.pt")
			torch.save(avBC,   path + "BCAAF_AV.pt")
			torch.save(awBC,   path + "BCAAF_AW.pt")
			torch.save(apBC,   path + "BCAAF_AP.pt")					

			print ("\n DATA SAVED.....\n ")
			print ("*"*85)

		# IMPOSE DYNAMIC LEARNING RATE
		scheduler_BCAAF.step()

		scheduler_UBCNN.step()
		scheduler_VBCNN.step()
		scheduler_WBCNN.step()
		scheduler_PBCNN.step()
	
		# LAMBDA FUNCTIONS
		scheduler_Lambda.step()

	# *******************************************  DOMAIN TRAINING  ************************************************** #

	W_NSE   = Parameter(torch.tensor(5.0))					# NAVIER STOKES EQUATION
	W_CONT  = Parameter(torch.tensor(2.0))					# CONTINUITY EQUATION
	W_BC    = Parameter(torch.tensor(6.0))					# NOSLIP BOUNDARY CONDITION
	W_DATA  = Parameter(torch.tensor(5.0))					# SENSOR DATA
	W_INLET = Parameter(torch.tensor(2.0))					# INLET VELOCITY PROFILE	
	

	# LAMBDA FUNCTION
	optim_Lambda  = optim.Adam([W_NSE, W_BC, W_DATA, W_CONT, W_INLET], lr = learn_rate_a,  maximize=True, betas = (0.9,0.99),eps = 10**-15)

	if (Flag_Dyn_LR):
		# LAMBDA FUNCTION
		scheduler_Lambda  = torch.optim.lr_scheduler.StepLR(optim_Lambda, step_size=step_eph_a, gamma=decay_rate)

	# LOSS FUNCTIONS
	Loss_NSE   = torch.empty(size=(epochs, 1))
	Loss_CONT  = torch.empty(size=(epochs, 1))
	Loss_BC    = torch.empty(size=(epochs, 1))
	Loss_Data  = torch.empty(size=(epochs, 1))	
	Loss_Inlet = torch.empty(size=(epochs, 1))	

	for epoch in range (epochs):
		loss_nse   = 0.
		loss_bc    = 0.
		loss_data  = 0.
		loss_cont  = 0.		
		loss_inlet = 0.		

		for batch_idx, (XX, YY, ZZ) in enumerate(train_load):

			batch_nse, batch_cont  = NSE_LOSS   (XX, YY, ZZ, U_NN, V_NN, W_NN, P_NN)
			batch_bc               = BC_LOSS    (BC_in, U_NN, V_NN, W_NN)
			batch_data             = DATA_LOSS  (x_data, y_data, z_data, P_data, P_NN)
			batch_inlet            = INLET_LOSS (x_inlet, y_inlet, z_inlet)

			loss = W_NSE*batch_nse + W_CONT*batch_cont + W_BC*batch_bc + W_INLET*batch_inlet + W_DATA*batch_data
	
			# ADAPTIVE ACTIVATION FUNCTION
			optim_UAAF.zero_grad()
			optim_VAAF.zero_grad()
			optim_WAAF.zero_grad()
			optim_PAAF.zero_grad()

			# NEURAL NETWORK
			optim_UNN.zero_grad()
			optim_VNN.zero_grad()
			optim_WNN.zero_grad()
			optim_PNN.zero_grad()

			# LAMBDA FUNCTION
			optim_Lambda.zero_grad()

			loss.backward()
			with torch.no_grad():
				# ADAPTIVE ACTIVATION FUNCTION
				optim_UAAF.step()
				optim_VAAF.step()
				optim_WAAF.step()
				optim_PAAF.step()

				# NEURAL NETWORK
				optim_UNN.step()
				optim_VNN.step()
				optim_WNN.step()
				optim_PNN.step()

				#LAMBDA FUNCTION
				optim_Lambda.step()

				loss_nse   += batch_nse
				loss_cont  += batch_cont
				loss_bc    += batch_bc
				loss_data  += batch_data
				loss_inlet += batch_inlet

		N = batch_idx + 1
		
		Loss_NSE   [epoch] = loss_nse/N
		Loss_CONT  [epoch] = loss_cont/N
		Loss_BC    [epoch] = loss_bc/N
		Loss_Data  [epoch] = loss_data/N
		Loss_Inlet [epoch] = loss_inlet/N	
	

		print('TOTAL AVERAGE LOSS OF NN PER EPOCH, [EPOCH =', epoch,']: \nNSE LOSS     :', Loss_NSE[epoch].item(), '\nCONT LOSS    :', Loss_CONT[epoch].item(), 
			"\nBC LOSS      :", Loss_BC[epoch].item(), "\nDATA LOSS    :", Loss_Data[epoch].item(), "\nINLET LOSS   :", Loss_Inlet[epoch].item())
		print("LAMBDA PARAMETERS:")
		print("NSE  =", f"{W_NSE.item():10.6}", "BC    =", f"{W_BC.item():10.6}", "\nDATA =", f"{W_DATA.item():10.6}", "INLET =", f"{W_INLET.item():10.6}",
			"CONT =", f"{W_CONT.item():10.6}")
		print('LEARNING RATE:', optim_UNN.param_groups[0]['lr'])
		print ("*"*85)
		
		# SAVE THE NETWORK DATA AND LOSS DATA FOR EVERY 100 EPOCHS
		if epoch % 100 == 0:
			# NETWORK DATA
			torch.save(P_NN.state_dict(),   path + "Pressure.pt"  )
			torch.save(U_NN.state_dict(),   path + "U_velocity.pt")
			torch.save(V_NN.state_dict(),   path + "V_Velocity.pt")
			torch.save(W_NN.state_dict(),   path + "W_Velocity.pt")

			# LOSS DATA
			torch.save(Loss_NSE  [0:epoch], path + "Loss_NSE.pt"  )
			torch.save(Loss_CONT [0:epoch], path + "Loss_CONT.pt" )
			torch.save(Loss_BC   [0:epoch], path + "Loss_BC.pt"   )
			torch.save(Loss_Data [0:epoch], path + "Loss_Data.pt" )
			torch.save(Loss_Inlet[0:epoch], path + "Loss_Inlet.pt")

			# ADAPTIVE ACTIVATION FUNCTION
			torch.save(au1,   path + "AAF_AU1.pt")
			torch.save(av1,   path + "AAF_AV1.pt")
			torch.save(aw1,   path + "AAF_AW1.pt")
			torch.save(ap1,   path + "AAF_AP1.pt")					

			print ("\n DATA SAVED.....\n ")
			print ("*"*85)

		# IMPOSE DYNAMIC LEARNING RATE
		scheduler_UAAF.step()
		scheduler_VAAF.step()
		scheduler_WAAF.step()
		scheduler_PAAF.step()

		scheduler_UNN.step()
		scheduler_VNN.step()
		scheduler_WNN.step()
		scheduler_PNN.step()
	
		# LAMBDA FUNCTIONS
		scheduler_Lambda.step()

	return


# ******************************************************************************************************************** #

# ********************************************* MAIN PROGRAM STARTS HERE ********************************************* #

# ******************************************************************************************************************** #

print ("PINN PROGRAM HAS BEEN STARTED SUCCESSFULLY ....... \n")

# ***** CHOOSE A PROCESSING UNIT FOR COMPUTATION: CPU or GPU

processor = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("AVAILABLE PROCESSOR:", processor, '\n')

# ***** HYPER PARAMETERS FOR THE NEURAL NETWORK
batchsize    = 12288                                 	# No. of data points in a whole dataset
epoch_offset = 501										# No. of Iterations fror UPSTREAM
epochs       = 20000                    				# No. of Iterations
input_VAR    = 3                                       	# No. of Flow variables
neurons      = 128                                     	# No. of Neurons in a layer

# ***** ADAPTIVE ACTIVATION FUNCTION
n 		   = 1.0  										# Scaling factor

# NEURAL NETWORK
auBC = Parameter(torch.ones(11, neurons), requires_grad= False).to(processor)	
avBC = Parameter(torch.ones(11, neurons), requires_grad= False).to(processor)		
awBC = Parameter(torch.ones(11, neurons), requires_grad= False).to(processor)											
apBC = Parameter(torch.ones(11, neurons), requires_grad= False).to(processor)

au1  = Parameter(torch.ones(11, neurons), requires_grad= False).to(processor)	
av1  = Parameter(torch.ones(11, neurons), requires_grad= False).to(processor)		
aw1  = Parameter(torch.ones(11, neurons), requires_grad= False).to(processor)											
ap1  = Parameter(torch.ones(11, neurons), requires_grad= False).to(processor)				

# ***** FILENAMES TO READ & WRITE THE DATA
mesh 		= "Symm_Stenosis.vtk"
mesh_up		= "Symm_Stenosis_up.vtk"
inlet_file  = "Inlet.vtk"
bc_wall     = "Wall_Symm_stenosis.vtk"
bc_wall_up  = "Wall_Symm_stenosis_up.vtk"
sensor_file	= "Symm_Stenosis.vtk"
sensor_ups	= "Symm_Stenosis_up.vtk"

# ***** LOCATION TO READ AND WRITE THE DATA
directory 	= os.getcwd()  								# GET THE CURRENT WORKING DIRECTORY  
path        = directory + '/'
mesh      	= path + mesh
mesh_up   	= path + mesh_up
inlet_file  = path + inlet_file
bc_wall     = path + bc_wall
bc_wall_up  = path + bc_wall_up 
sensor_file	= path + sensor_file
sensor_ups	= path + sensor_ups

# ***** NORMALISATION OF FLOW VARIABLES
x_scale		= 3.00
y_scale 	= 0.30
z_scale		= 0.30
P_max 		= 825.956113122666
P_min  		= -459.6687777001301
vel_scale   = 18.90947
x_offset    = 1.0

# ***** FLUID PROPERTIES
density     = 1.06
diffusion   = 0.0377358

# ***** UPSTREAM DOMAIN
xsensor      = 0.20
Nslice_up    = 8

# ***** FLAGS TO IMPROVE THE USER-INTERFACE
Flag_Batch	= True  									# True  : ENABLES THE BATCH-WISE COMPUTATION
Flag_Resume = False  									# False : STARTS FROM EPOCH = 0
Flag_Dyn_LR = True 										# True  : DYNAMIC LEARNING RATE

if (Flag_Dyn_LR):
	learning_rate = 1e-3
	learn_rate_a  = 1e-3
	step_epoch    = 2000
	step_eph_a    = 2000
	decay_rate    = 0.50

	# BC - UPSTREAM DOMAIN
	BClearning_rate = 1e-3
	BClearn_rate_a  = 1e-3
	BCstep_epoch    = 150
	BCstep_eph_a    = 150
	BCdecay_rate    = 0.50

# ****** ADAPTIVE WEIGHTS FOR THE LOSS FUNCTIONS
W_NSE   = Parameter(torch.tensor(5.0))					# NAVIER STOKES EQUATION
W_CONT  = Parameter(torch.tensor(2.0))					# CONTINUITY EQUATION
W_BC    = Parameter(torch.tensor(5.0))					# NOSLIP BOUNDARY CONDITION
W_DATA  = Parameter(torch.tensor(2.0))					# SENSOR DATA
W_INLET = Parameter(torch.tensor(2.0))					# INLET VELOCITY PROFILE

# ***** SENSORS LOCATION
Nslice = 128

# ***** READING THE FILES
''' *********************************************** MESH FILE ****************************************************** '''
print ("*"*85)
print ('READING THE MESH FILE: ', mesh[len(directory)+1:])
reader = vtk.vtkUnstructuredGridReader()
reader.SetFileName(mesh)
reader.Update()
data = reader.GetOutput()
n_points = data.GetNumberOfPoints()
print ('NO. OF GRID POINTS IN THE MESH:', n_points)
x_vtk_mesh = np.zeros((n_points,1))
y_vtk_mesh = np.zeros((n_points,1))
z_vtk_mesh = np.zeros((n_points,1))
VTKpoints = vtk.vtkPoints()
for i in range(n_points):
	pt_iso  =  data.GetPoint(i)
	x_vtk_mesh[i] = pt_iso[0]	
	y_vtk_mesh[i] = pt_iso[1]
	z_vtk_mesh[i] = pt_iso[2]

x  = np.reshape(x_vtk_mesh , (np.size(x_vtk_mesh [:]),1)) 
y  = np.reshape(y_vtk_mesh , (np.size(y_vtk_mesh [:]),1))
z  = np.reshape(z_vtk_mesh , (np.size(z_vtk_mesh [:]),1))

print ("SHAPE OF X:", 	x.shape)
print ("SHAPE OF Y:", 	y.shape)
print ("SHAPE OF Z:", 	z.shape)

print ("*"*85)

print ('READING THE UPSTREAM MESH FILE: ', mesh_up[len(directory)+1:])
reader = vtk.vtkUnstructuredGridReader()
reader.SetFileName(mesh_up)
reader.Update()
data = reader.GetOutput()
n_points = data.GetNumberOfPoints()
print ('NO. OF GRID POINTS IN THE MESH:', n_points)
x_vtk_mesh = np.zeros((n_points,1))
y_vtk_mesh = np.zeros((n_points,1))
z_vtk_mesh = np.zeros((n_points,1))
VTKpoints = vtk.vtkPoints()
for i in range(n_points):
	pt_iso  =  data.GetPoint(i)
	x_vtk_mesh[i] = pt_iso[0]	
	y_vtk_mesh[i] = pt_iso[1]
	z_vtk_mesh[i] = pt_iso[2]

xup  = np.reshape(x_vtk_mesh , (np.size(x_vtk_mesh [:]),1)) 
yup  = np.reshape(y_vtk_mesh , (np.size(y_vtk_mesh [:]),1))
zup  = np.reshape(z_vtk_mesh , (np.size(z_vtk_mesh [:]),1))

print ("SHAPE OF X:", 	xup.shape)
print ("SHAPE OF Y:", 	yup.shape)
print ("SHAPE OF Z:", 	zup.shape)

print ("*"*85)

''' ************************************** WALL BOUNDARY POINTS FILE *********************************************** '''

print ('READING THE WALL BOUNDARY FILE:', bc_wall[len(directory)+1:])
reader =  vtk.vtkPolyDataReader()
reader.SetFileName(bc_wall)
reader.Update()
data_vtk = reader.GetOutput()
n_pointsw = data_vtk.GetNumberOfPoints()
print ('NO. OF GRID POINTS AT THE WALL:' ,n_pointsw)
x_vtk_mesh = np.zeros((n_pointsw,1))
y_vtk_mesh = np.zeros((n_pointsw,1))
z_vtk_mesh = np.zeros((n_pointsw,1))
VTKpoints = vtk.vtkPoints()
for i in range(n_pointsw):
	pt_iso  =  data_vtk.GetPoint(i)
	x_vtk_mesh[i] = pt_iso[0]	
	y_vtk_mesh[i] = pt_iso[1]
	z_vtk_mesh[i] = pt_iso[2]
	VTKpoints.InsertPoint(i, pt_iso[0], pt_iso[1], pt_iso[2])
point_data = vtk.vtkUnstructuredGrid()
point_data.SetPoints(VTKpoints)
xbc_wall  = np.reshape(x_vtk_mesh , (np.size(x_vtk_mesh [:]),1)) 
ybc_wall  = np.reshape(y_vtk_mesh , (np.size(y_vtk_mesh [:]),1))
zbc_wall  = np.reshape(z_vtk_mesh , (np.size(z_vtk_mesh [:]),1))

print ("*"*85)


print ('READING THE UPSTREAM WALL BOUNDARY FILE:', bc_wall_up[len(directory)+1:])
reader =  vtk.vtkPolyDataReader()
reader.SetFileName(bc_wall_up)
reader.Update()
data_vtk = reader.GetOutput()
n_pointsw = data_vtk.GetNumberOfPoints()
print ('NO. OF GRID POINTS AT THE WALL:' ,n_pointsw)
x_vtk_mesh = np.zeros((n_pointsw,1))
y_vtk_mesh = np.zeros((n_pointsw,1))
z_vtk_mesh = np.zeros((n_pointsw,1))
VTKpoints = vtk.vtkPoints()
for i in range(n_pointsw):
	pt_iso  =  data_vtk.GetPoint(i)
	x_vtk_mesh[i] = pt_iso[0]	
	y_vtk_mesh[i] = pt_iso[1]
	z_vtk_mesh[i] = pt_iso[2]
	VTKpoints.InsertPoint(i, pt_iso[0], pt_iso[1], pt_iso[2])
point_data = vtk.vtkUnstructuredGrid()
point_data.SetPoints(VTKpoints)
xbc_wall_up  = np.reshape(x_vtk_mesh , (np.size(x_vtk_mesh [:]),1)) 
ybc_wall_up  = np.reshape(y_vtk_mesh , (np.size(y_vtk_mesh [:]),1))
zbc_wall_up  = np.reshape(z_vtk_mesh , (np.size(z_vtk_mesh [:]),1))

print ("*"*85)

''' ****************************************** SENSOR DATA FILE **************************************************** '''
fieldname  = 'pressure' 														# FIELD NAME FOR VTK FILES

print ("READING THE SENSOR DATA FILE:", sensor_file[len(directory)+1:] )
reader = vtk.vtkUnstructuredGridReader()
reader.SetFileName(sensor_file)
reader.Update()

x_locations = torch.linspace(0.001, x_scale-0.01, steps = Nslice)		# RANDOMLY PICK X LOCATIONS FOR SLICING

x_data = np.zeros (Nslice)
y_data = np.zeros (Nslice)
z_data = np.zeros (Nslice)

for i, xl in enumerate (x_locations):

	#****** PLANE CREATION
	plane = vtk.vtkPlane()
	plane.SetOrigin(xl, 0.0, 0.0)								# LOCATION OF SLICE ALONG X DIRECTION
	plane.SetNormal(1, 0, 0)									# SLICE IN YZ PLANE 
	
	#****** SLICE THE MESH AT THE CHOSEN PLANE
	cutter = vtk.vtkCutter()
	cutter.SetCutFunction(plane)
	cutter.SetInputConnection(reader.GetOutputPort())
	cutter.Update()

	#****** INTEGRATE THE VARIABLES TO GET CENTROID
	integrate = vtk.vtkIntegrateAttributes()
	integrate.SetInputConnection(cutter.GetOutputPort())
	integrate.Update()
	x_data [i] = integrate.GetOutput().GetBounds()[0]
	y_data [i] = integrate.GetOutput().GetBounds()[2]
	z_data [i] = integrate.GetOutput().GetBounds()[4]

data_vtk = reader.GetOutput()
n_point  = data_vtk.GetNumberOfPoints()
print ('NO. OF DATA POINTS IN THE SENSOR FILE:', n_point)

VTKpoints = vtk.vtkPoints()
for i in range(len(x_data)): 
	VTKpoints.InsertPoint(i, x_data[i] , y_data[i]  , z_data[i])

point_data = vtk.vtkUnstructuredGrid()
point_data.SetPoints(VTKpoints)
probe = vtk.vtkProbeFilter()
probe.SetInputData(point_data)
probe.SetSourceData(data_vtk)
probe.Update()
array  = probe.GetOutput().GetPointData().GetArray(fieldname)
P_data = VN.vtk_to_numpy(array)

print ("*"*85)

del data, data_vtk, reader, point_data, x_vtk_mesh, y_vtk_mesh, z_vtk_mesh, VTKpoints

print ("READING THE SENSOR DATA FILE:", sensor_ups[len(directory)+1:] )
reader = vtk.vtkUnstructuredGridReader()
reader.SetFileName(sensor_ups)
reader.Update()

x_locations = torch.linspace(0.001, xsensor-0.0001, steps = Nslice_up)	# RANDOMLY PICK X LOCATIONS FOR SLICING

x_data_up = np.zeros (Nslice_up)
y_data_up = np.zeros (Nslice_up)
z_data_up = np.zeros (Nslice_up)

for i, xl in enumerate (x_locations):

	#****** PLANE CREATION
	plane = vtk.vtkPlane()
	plane.SetOrigin(xl, 0.0, 0.0)								# LOCATION OF SLICE ALONG X DIRECTION
	plane.SetNormal(1, 0, 0)									# SLICE IN YZ PLANE 
	
	#****** SLICE THE MESH AT THE CHOSEN PLANE
	cutter = vtk.vtkCutter()
	cutter.SetCutFunction(plane)
	cutter.SetInputConnection(reader.GetOutputPort())
	cutter.Update()

	#****** INTEGRATE THE VARIABLES TO GET CENTROID
	integrate = vtk.vtkIntegrateAttributes()
	integrate.SetInputConnection(cutter.GetOutputPort())
	integrate.Update()
	x_data_up [i] = integrate.GetOutput().GetBounds()[0]
	y_data_up [i] = integrate.GetOutput().GetBounds()[2]
	z_data_up [i] = integrate.GetOutput().GetBounds()[4]

data_vtk = reader.GetOutput()
n_point  = data_vtk.GetNumberOfPoints()
print ('NO. OF DATA POINTS IN THE SENSOR FILE:', n_point)

VTKpoints = vtk.vtkPoints()
for i in range(len(x_data_up)): 
	VTKpoints.InsertPoint(i, x_data_up[i] , y_data_up[i]  , z_data_up[i])

point_data = vtk.vtkUnstructuredGrid()
point_data.SetPoints(VTKpoints)
probe = vtk.vtkProbeFilter()
probe.SetInputData(point_data)
probe.SetSourceData(data_vtk)
probe.Update()
array  = probe.GetOutput().GetPointData().GetArray(fieldname)
P_data_up = VN.vtk_to_numpy(array)

print ("*"*85)

del data_vtk, reader, point_data, VTKpoints

''' ************************************** INLET WALL BOUNDARY FILE ************************************************ '''

print ('READING THE INLET WALL FILE:', inlet_file[len(directory)+1:])
reader =  vtk.vtkPolyDataReader()
reader.SetFileName(inlet_file)
reader.Update()
data_vtk = reader.GetOutput()
n_pointsw = data_vtk.GetNumberOfPoints()
print ('NO. OF GRID POINTS AT THE INLET:' ,n_pointsw)
x_vtk_mesh = np.zeros((n_pointsw,1))
y_vtk_mesh = np.zeros((n_pointsw,1))
z_vtk_mesh = np.zeros((n_pointsw,1))
VTKpoints = vtk.vtkPoints()
for i in range(n_pointsw):
	pt_iso  =  data_vtk.GetPoint(i)
	x_vtk_mesh[i] = pt_iso[0]	
	y_vtk_mesh[i] = pt_iso[1]
	z_vtk_mesh[i] = pt_iso[2]
	VTKpoints.InsertPoint(i, pt_iso[0], pt_iso[1], pt_iso[2])
point_data = vtk.vtkUnstructuredGrid()
point_data.SetPoints(VTKpoints)
x_inlet  = np.reshape(x_vtk_mesh , (np.size(x_vtk_mesh [:]),1)) 
y_inlet  = np.reshape(y_vtk_mesh , (np.size(y_vtk_mesh [:]),1))
z_inlet  = np.reshape(z_vtk_mesh , (np.size(z_vtk_mesh [:]),1))

print ("*"*85)

''' ******************************** RESHAPE THE ARRAYS TO GET 2D-ARRAY ******************************************** '''
# WALL BOUNDARY
xbc_wall = xbc_wall.reshape(-1, 1)
ybc_wall = ybc_wall.reshape(-1, 1)
zbc_wall = zbc_wall.reshape(-1, 1)

xbc_wall_up = xbc_wall_up.reshape(-1, 1)
ybc_wall_up = ybc_wall_up.reshape(-1, 1)
zbc_wall_up = zbc_wall_up.reshape(-1, 1)

# INLET BOUNDARY
x_inlet = x_inlet.reshape(-1, 1)
y_inlet = y_inlet.reshape(-1, 1)
z_inlet = z_inlet.reshape(-1, 1)

# SENSOR DATA
x_data = x_data.reshape(-1, 1) 
y_data = y_data.reshape(-1, 1)
z_data = z_data.reshape(-1, 1) 
P_data = P_data.reshape(-1, 1)

x_data_up = x_data_up.reshape(-1, 1) 
y_data_up = y_data_up.reshape(-1, 1)
z_data_up = z_data_up.reshape(-1, 1) 
P_data_up = P_data_up.reshape(-1, 1) 

# WALL BOUNDARY
print("SHAPE OF WALL  BC X:", xbc_wall.shape)
print("SHAPE OF WALL  BC Y:", ybc_wall.shape)
print("SHAPE OF WALL  BC Z:", zbc_wall.shape)

# SENSOR DATA
print("SHAPE OF SENSOR X  :", x_data.shape)
print("SHAPE OF SENSOR Y  :", y_data.shape)
print("SHAPE OF SENSOR Z  :", z_data.shape)
print("SHAPE OF SENSOR Pr :", P_data.shape)

print ("*"*85)

''' ************************************* NORMALISATION OF VARIABLES *********************************************** '''
# MESH POINTS
x 	= x / x_scale
y 	= y / y_scale
z 	= z / z_scale

xup = xup / x_scale
yup = yup / y_scale
zup = zup / z_scale

# WALL BOUNDARY POINTS
xbc_wall = xbc_wall / x_scale
ybc_wall = ybc_wall / y_scale
zbc_wall = zbc_wall / z_scale

xbc_wall_up = xbc_wall_up / x_scale
ybc_wall_up = ybc_wall_up / y_scale
zbc_wall_up = zbc_wall_up / z_scale

# INLET BOUNDARY POINTS
x_inlet = x_inlet / x_scale
y_inlet = y_inlet / y_scale
z_inlet = z_inlet / z_scale

# SENSOR DATA POINTS
x_data	 = x_data / x_scale
y_data   = y_data / y_scale
z_data   = z_data / z_scale
P_data   = (P_data - P_min)/(P_max - P_min) + (P_data - P_max)/(P_max - P_min)

x_data_up = x_data_up / x_scale
y_data_up = y_data_up / y_scale
z_data_up = z_data_up / z_scale
P_data_up = (P_data_up - P_min)/(P_max - P_min) + (P_data_up - P_max)/(P_max - P_min)

PINN(processor, x, y, z, xup, yup, zup, xbc_wall, ybc_wall, zbc_wall, xbc_wall_up, ybc_wall_up, zbc_wall_up, x_data, y_data, 
	z_data, x_data_up, y_data_up, z_data_up, P_data, P_data_up, x_inlet, y_inlet, z_inlet, x_scale, y_scale, z_scale, P_max, 
	P_min, vel_scale, density, diffusion, Nslice, learning_rate, learn_rate_a, step_epoch, step_eph_a, decay_rate, BClearning_rate, 
	BClearn_rate_a, BCstep_epoch, BCstep_eph_a, BCdecay_rate, batchsize, epochs, epoch_offset, input_VAR, neurons, auBC, avBC, 
	awBC, apBC, au1, av1, aw1, ap1, n, path, Flag_Batch, Flag_Resume, Flag_Dyn_LR, W_NSE, W_CONT, W_BC, W_DATA, W_INLET)

