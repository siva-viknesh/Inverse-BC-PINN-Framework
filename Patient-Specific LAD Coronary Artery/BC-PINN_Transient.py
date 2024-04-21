# ********************** PHYSICS INFORMED NEURAL NETWORK - SOLVING & MODELING 3D PDE'S ******************************* #
# Author  : SIVA VIKNESH
# Email   : siva.viknesh@sci.utah.edu / sivaviknesh14@gmail.com
# Address : SCI INSTITUTE, UNIVERSITY OF UTAH, SALT LAKE CITY, UTAH, USA
# ******************************************************************************************************************** #

# ****** IMPORTING THE NECESSARY LIBRARIES
import os
import torch
import copy
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader, TensorDataset
import vtk
from vtk.util import numpy_support as VN
import Supplement_PINN

''' *********************** JUMP TO THE MAIN PROGRAM TO CONTROL THE PROGRAM PARAMETERS ***************************** '''

# ************************************** WRITING THE CUSTOM FUNCTIONS ************************************************ #
''' ****************************************** SENSOR DATA FILE **************************************************** '''

def SENSOR_DATA (fieldname, file_name, xmin, xmax, ymin, ymax, zmin, zmax, xplane, yplane, zplane, Nslice):

	x_data = np.zeros (Nslice)
	y_data = np.zeros (Nslice)
	z_data = np.zeros (Nslice)
	P_data = np.zeros (Nslice)

	xl = torch.linspace(xmin, xmax, steps = Nslice)
	yl = torch.linspace(ymin, ymax, steps = Nslice)
	zl = torch.linspace(zmin, zmax, steps = Nslice)

	print ("READING THE SENSOR DATA FILE:", file_name[len(directory)+1:] )
	reader = vtk.vtkUnstructuredGridReader()
	reader.SetFileName(file_name)
	reader.Update()

	for i in range(Nslice):

		#****** PLANE CREATION
		plane = vtk.vtkPlane()
		plane.SetOrigin(xl [i], yl [i], zl [i])					# LOCATION OF SLICE ALONG X DIRECTION
		plane.SetNormal(xplane, yplane, zplane)					# SLICE IN YZ PLANE 
	
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

	VTKpoints = vtk.vtkPoints()
	for k in range(Nslice): 
		VTKpoints.InsertPoint(k, x_data[k] , y_data[k]  , z_data[k])

	point_data = vtk.vtkUnstructuredGrid()
	point_data.SetPoints(VTKpoints)
	probe = vtk.vtkProbeFilter()
	probe.SetInputData(point_data)
	probe.SetSourceData(data_vtk)
	probe.Update()
	array  = probe.GetOutput().GetPointData().GetArray(fieldname)
	P_data = VN.vtk_to_numpy(array)
	del data_vtk, point_data, reader, VTKpoints, array, probe

	# -------------------------------------------------------------------------------------------------------------------#
	fieldname  = 'velocity' 														# FIELD NAME FOR VTK FILES
	reader = vtk.vtkUnstructuredGridReader()
	reader.SetFileName(file_name)
	reader.Update()

	#****** PLANE CREATION
	plane = vtk.vtkPlane()
	plane.SetOrigin(xmin+0.0001, ymin+0.0001, zmin+0.0001)					# LOCATION OF SLICE ALONG X DIRECTION
	plane.SetNormal(xplane, yplane, zplane)					# SLICE IN YZ PLANE 

	#****** SLICE THE MESH AT THE CHOSEN PLANE
	cutter = vtk.vtkCutter()
	cutter.SetCutFunction(plane)
	cutter.SetInputConnection(reader.GetOutputPort())
	cutter.Update()

	#****** INTEGRATE THE VARIABLES TO GET CENTROID
	integrate = vtk.vtkIntegrateAttributes()
	integrate.SetInputConnection(cutter.GetOutputPort())
	integrate.Update()
	xd = integrate.GetOutput().GetBounds()[0]
	yd = integrate.GetOutput().GetBounds()[2]
	zd = integrate.GetOutput().GetBounds()[4]

	data_vtk = reader.GetOutput()
	n_point  = data_vtk.GetNumberOfPoints()

	VTKpoints = vtk.vtkPoints()
	for k in range(1): 
		VTKpoints.InsertPoint(k, xd, yd, zd)

	point_data = vtk.vtkUnstructuredGrid()
	point_data.SetPoints(VTKpoints)
	probe = vtk.vtkProbeFilter()
	probe.SetInputData(point_data)
	probe.SetSourceData(data_vtk)
	probe.Update()
	array  = probe.GetOutput().GetPointData().GetArray('velocity')
	vel = VN.vtk_to_numpy(array)
	del data_vtk, point_data

	return np.array(x_data), np.array(y_data), np.array(z_data), np.array(P_data), vel

def PINN(processor, x, y, z, xbc_wall, ybc_wall, zbc_wall, x_scale,	y_scale, z_scale, xmin, xmax, ymin, ymax, zmin, zmax, 
	xplane, yplane, zplane, sensor_file, fieldname, Nfile, density, diffusion, batchsize, epochs, input_VAR, neurons, 
	n, Flag_Batch, Flag_Resume, Flag_Dyn_LR, path):

	# MESH SPATIAL DATA
	x = torch.Tensor(x).to(processor)
	y = torch.Tensor(y).to(processor)
	z = torch.Tensor(z).to(processor)
	
	# INLET PROFILE
	x_inlet  = np.array ([0.693579 /x_scale]).reshape (-1, 1)
	y_inlet  = np.array ([1.387520 /y_scale]).reshape (-1, 1)
	z_inlet  = np.array ([1.554370 /z_scale]).reshape (-1, 1)

	x_inlet  = torch.Tensor(x_inlet).to(processor)
	y_inlet  = torch.Tensor(y_inlet).to(processor)
	z_inlet  = torch.Tensor(z_inlet).to(processor)

	# WALL BOUNDARY
	xbc_wall = torch.Tensor(xbc_wall)
	ybc_wall = torch.Tensor(ybc_wall)
	zbc_wall = torch.Tensor(zbc_wall)
	BC_in    = torch.cat((xbc_wall, ybc_wall, zbc_wall), 1).to(processor)
	a  		 = Parameter(torch.ones(1))

	train_load = DataLoader(TensorDataset(x, y, z), batch_size=batchsize, shuffle=True,  drop_last = False)
	del xbc_wall, ybc_wall, zbc_wall, x, y, z

	# NEURAL NETWORKS 
	au1 = Parameter(torch.ones(11, neurons), requires_grad= False).to(processor)	
	av1 = Parameter(torch.ones(11, neurons), requires_grad= False).to(processor)		
	aw1 = Parameter(torch.ones(11, neurons), requires_grad= False).to(processor)											
	ap1 = Parameter(torch.ones(11, neurons), requires_grad= False).to(processor)

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

	class U_VEL_NN(CUSTOM_SiLU):

		def __init__(self):
			super().__init__(a)
			self.main = nn.Sequential(
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
			output = self.main(x)
			return output		

	# ****** Y-VELOCITY COMPONENT: 

	class V_VEL_NN(CUSTOM_SiLU):
		def __init__(self):
			super().__init__(a)
			self.main = nn.Sequential(
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
			
			output = self.main(x)
			
			return output			

	# ****** Z-VELOCITY COMPONENT: w

	class W_VEL_NN(CUSTOM_SiLU):
		def __init__(self):
			super().__init__(a)
			self.main = nn.Sequential(
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
			output = self.main(x)
			
			return output	

	# ****** PRESSURE: p
	
	class PRESS_NN(CUSTOM_SiLU):
		def __init__(self):
			super().__init__(a)
			self.main = nn.Sequential(
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

				CUSTOM_SiLU(a = ap1[10,:]),
				nn.Linear(neurons,1)
			)

		def forward(self, x):
			output = self.main(x)
			
			return output

	A_NN = CUSTOM_SiLU(a).to(processor)

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
	U_NN.apply(init_normal)
	V_NN.apply(init_normal)
	W_NN.apply(init_normal)
	P_NN.apply(init_normal)	

	# *************************************** FUNCTIONS FOR PINN TRIANING ******************************************** #

	def NSE_LOSS (x, y, z, vel_scale, P_max):   					# NAVIER-STOKES EQUATION + CONTINUITY EQUN 

		x.requires_grad = True
		y.requires_grad = True
		z.requires_grad = True

		NSE_in = torch.cat((x, y, z),1)

		u = U_NN(NSE_in)
		v = V_NN(NSE_in)
		w = W_NN(NSE_in)
		P = P_NN(NSE_in)

		u = u.view(len(u),-1)
		v = v.view(len(v),-1)
		w = w.view(len(w),-1)
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
		loss_1 = u*du_dx / x_scale + v*du_dy / y_scale + w*du_dz / z_scale - diffusion*( du_dxx/XX_scale  + du_dyy /YY_scale + du_dzz/ZZ_scale)+ 1/density*(dP_dx*P_max / (x_scale*UU_scale))  
		
		# Y MOMENTUM EQUATION LOSS
		loss_2 = u*dv_dx / x_scale + v*dv_dy / y_scale + w*dv_dz / z_scale - diffusion*( dv_dxx/ XX_scale + dv_dyy /YY_scale + dv_dzz/ZZ_scale)+ 1/density*(dP_dy*P_max / (y_scale*UU_scale)) 
		
		# Z MOMENTUM EQUATION LOSS
		loss_3 = u*dw_dx / x_scale + v*dw_dy / y_scale + w*dw_dz / z_scale - diffusion*( dw_dxx/ XX_scale + dw_dyy /YY_scale + dw_dzz/ZZ_scale)+ 1/density*(dP_dz*P_max / (z_scale*UU_scale))
		
		# CONTINUITY EQUATION LOSS
		loss_4 = (du_dx / x_scale + dv_dy / y_scale + dw_dz / z_scale) 


		loss_f    = nn.MSELoss()
		loss_NSE  = loss_f(loss_1,torch.zeros_like(loss_1))+  loss_f(loss_2,torch.zeros_like(loss_2))+  loss_f(loss_3,torch.zeros_like(loss_3)) 
		loss_CONT = loss_f(loss_4,torch.zeros_like(loss_4))  

		return loss_NSE, loss_CONT

		
	def BC_LOSS (BC_in):            							# BOUNDARY CONDITION LOSS

		# NO-SLIP WALL 
		u = U_NN(BC_in)
		v = V_NN(BC_in)
		w = W_NN(BC_in)

		u = u.view(len(u),-1)
		v = v.view(len(v),-1)
		w = w.view(len(w),-1)

		loss_f = nn.MSELoss()
		loss_noslip = loss_f(u, torch.zeros_like(u)) + loss_f(v, torch.zeros_like(v)) + loss_f(w, torch.zeros_like(w))

		return loss_noslip


	def DATA_LOSS (x_data, y_data, z_data, P_data):								# DATA LOSS AT THE PROBED LOCATIONS

		x_data.requires_grad = True
		y_data.requires_grad = True
		z_data.requires_grad = True

		DATA_in = torch.cat((x_data, y_data, z_data), 1)
		
		P_out = P_NN(DATA_in)
		P_out = P_out.view(len(P_out),-1)

		loss_f    = nn.MSELoss()
		loss_data = loss_f (P_out, P_data) 

		return loss_data

	def INLET_LOSS (x, y, z, Uscale, Vscale, Wscale):						# INLET PROFILE

		x.requires_grad = True
		y.requires_grad = True
		z.requires_grad = True

		INLET_in = torch.cat((x, y, z), 1)

		U = U_NN(INLET_in)
		V = V_NN(INLET_in)
		W = W_NN(INLET_in)

		U = U.view(len(U),-1)
		V = V.view(len(V),-1)
		W = W.view(len(W),-1)

		Vm = torch.sqrt(U**2 + V**2 + W**2)

		Up = torch.ones_like(U)*Uscale
		Vp = torch.ones_like(V)*Vscale
		Wp = torch.ones_like(W)*Vscale

		dVm_dx  = torch.autograd.grad(Vm, x,     grad_outputs=torch.ones_like(x), create_graph = True, only_inputs=True)[0]		
		dVm_dy  = torch.autograd.grad(Vm, y,     grad_outputs=torch.ones_like(y), create_graph = True, only_inputs=True)[0]
		dVm_dz  = torch.autograd.grad(Vm, z,     grad_outputs=torch.ones_like(z), create_graph = True, only_inputs=True)[0]

		loss_f = nn.MSELoss()
		loss1  = loss_f (dVm_dx, torch.zeros_like(dVm_dx)) + loss_f (dVm_dy, torch.zeros_like(dVm_dy)) + loss_f (dVm_dz, torch.zeros_like(dVm_dz))
		loss2  = loss_f (U, Up) + loss_f (V, Vp) + loss_f (W, Wp) 

		return loss1 + loss2

# **************************************** NEURAL NETWORK COMPUTATION *********************************************** #

	if (Flag_Resume):
		print('READING THE TRAINED DATA OF NET I ..... \n')
		U_NN.load_state_dict(torch.load(path + "U_velocity.pt"))
		V_NN.load_state_dict(torch.load(path + "V_Velocity.pt"))
		W_NN.load_state_dict(torch.load(path + "W_Velocity.pt"))
		P_NN.load_state_dict(torch.load(path + "Pressure.pt"))

#--------------------------------- MARCHED SOLUTION OF NEURAL NETWORK COMPUTATION -------------------------------------#
	for mod in range (start_file, end_file+1):

		# ****** ADAPTIVE WEIGHTS FOR THE LOSS FUNCTIONS
		W_NSE   = Parameter(torch.tensor(5.0))					# NAVIER STOKES EQUATION
		W_CONT  = Parameter(torch.tensor(2.0))					# CONTINUITY EQUATION
		W_BC    = Parameter(torch.tensor(3.0))					# NOSLIP BOUNDARY CONDITION
		W_DATA  = Parameter(torch.tensor(5.0))					# SENSOR DATA
		W_INLET = Parameter(torch.tensor(2.0))					# INLET DATA

		# SENSOR LOCATION
		file_name = sensor_file + str(mod) + ".vtk" 
		x_data, y_data, z_data, P_data, velocity = SENSOR_DATA (fieldname, file_name, xmin, xmax, ymin, ymax, zmin, zmax, xplane, yplane, zplane, Nslice)

		x_data = x_data.reshape(-1, 1)
		y_data = y_data.reshape(-1, 1)
		z_data = z_data.reshape(-1, 1)
		P_data = P_data.reshape(-1, 1)

		x_data    = torch.Tensor(x_data/x_scale).to(processor)
		y_data    = torch.Tensor(y_data/y_scale).to(processor)
		z_data    = torch.Tensor(z_data/z_scale).to(processor)
		P_data    = torch.Tensor(P_data).to(processor)
		velocity  = torch.squeeze(torch.Tensor(velocity)).to(processor)

		P_max     = torch.max(P_data)
		P_data    = P_data / P_max
		vel_scale = torch.sqrt(velocity[0]**2 + velocity[1]**2 + velocity[2]**2 )
		Uscale    = (velocity [0]) / vel_scale
		Vscale    = (velocity [1]) / vel_scale
		Wscale    = (velocity [2]) / vel_scale

		print ("VELOCITY SCALE =", vel_scale.item())
		print ("MAX PRESSURE   =", P_max.item())
		print ("*"*85)
		
		epochs += 1 
		# LOSS FUNCTIONS
		Loss_NSE     = torch.zeros(epochs)
		Loss_CONT    = torch.zeros(epochs)
		Loss_BC      = torch.zeros(epochs)
		Loss_Data    = torch.zeros(epochs)
		Loss_Inlet   = torch.zeros(epochs)

		if (Flag_Dyn_LR):
			learning_rate = 1e-3
			learn_rate_a  = 1e-3
			step_epoch    = 75
			step_eph_a    = 75
			decay_rate    = 0.50

		# ADAPTIVE ACTIVATION FUNCTION
		optim_AAF = optim.Adam([au1, av1, aw1, ap1], lr = learn_rate_a, betas = (0.9,0.99),eps = 10**-15)

		# LAMBDA FUNCTION
		optim_Lambda  = optim.Adam([W_NSE, W_BC, W_DATA, W_CONT, W_INLET], lr = learn_rate_a, maximize=True, betas = (0.9,0.99),eps = 10**-15)

		# NEURAL NETWORKS
		optim_UNN  = optim.Adam(U_NN.parameters(),  lr=learning_rate, betas = (0.9,0.99),eps = 10**-15)
		optim_VNN  = optim.Adam(V_NN.parameters(),  lr=learning_rate, betas = (0.9,0.99),eps = 10**-15)
		optim_WNN  = optim.Adam(W_NN.parameters(),  lr=learning_rate, betas = (0.9,0.99),eps = 10**-15)
		optim_PNN  = optim.Adam(P_NN.parameters(),  lr=learning_rate, betas = (0.9,0.99),eps = 10**-15)

		if (Flag_Dyn_LR):			
			# ADAPTIVE ACTIVATION FUNCTION
			scheduler_AAF = torch.optim.lr_scheduler.StepLR(optim_AAF, step_size=step_eph_a, gamma=decay_rate)

			# LAMBDA FUNCTION
			scheduler_Lambda  = torch.optim.lr_scheduler.StepLR(optim_Lambda, step_size=step_eph_a, gamma=decay_rate)
		
			# NEURAL NETWORK
			scheduler_UNN = torch.optim.lr_scheduler.StepLR (optim_UNN, step_size=step_epoch, gamma=decay_rate)
			scheduler_VNN = torch.optim.lr_scheduler.StepLR (optim_VNN, step_size=step_epoch, gamma=decay_rate)
			scheduler_WNN = torch.optim.lr_scheduler.StepLR (optim_WNN, step_size=step_epoch, gamma=decay_rate)
			scheduler_PNN = torch.optim.lr_scheduler.StepLR (optim_PNN, step_size=step_epoch, gamma=decay_rate)

		epoch      = 0
		total_loss = 1.0
		while (epoch < epochs) and (total_loss > 2e-4):
			loss_nse   = 0.
			loss_bc    = 0.
			loss_data  = 0.
			loss_cont  = 0.
			loss_inlet = 0.

			for batch_idx, (X, Y, Z) in enumerate(train_load):

				batch_nse, batch_cont  = NSE_LOSS   (X, Y, Z, vel_scale, P_max)
				batch_bc               = BC_LOSS    (BC_in)
				batch_data             = DATA_LOSS  (x_data, y_data, z_data, P_data)
				batch_inlet            = INLET_LOSS (x_inlet, y_inlet, z_inlet, Uscale, Vscale, Wscale)
			
				loss = W_NSE*batch_nse + W_CONT*batch_cont + W_BC*batch_bc + W_DATA*batch_data + W_INLET*batch_inlet
			
				# ADAPTIVE ACTIVATION FUNCTION
				optim_AAF.zero_grad()

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
					optim_AAF.step()

					# NEURAL NETWORK
					optim_UNN.step()
					optim_VNN.step()
					optim_WNN.step()
					optim_PNN.step()

					#LAMBDA FUNCTION
					optim_Lambda.step()

				loss_nse   += batch_nse.detach()
				loss_cont  += batch_cont.detach()
				loss_bc    += batch_bc.detach()
				loss_data  += batch_data.detach()			
				loss_inlet += batch_inlet.detach()		

			N = batch_idx + 1
			Loss_NSE   [epoch] = loss_nse/N
			Loss_CONT  [epoch] = loss_cont/N
			Loss_BC    [epoch] = loss_bc/N
			Loss_Data  [epoch] = loss_data/N
			Loss_Inlet [epoch] = loss_inlet/N

			total_loss = Loss_NSE   [epoch] + Loss_CONT  [epoch] + Loss_BC    [epoch] + Loss_Data  [epoch] + Loss_Inlet [epoch]
	
			print('TOTAL AVERAGE LOSS OF NN - MODEL', mod ,' [EPOCH =', epoch,']: \nNSE LOSS     :', (loss_nse/N).item(), '\nCONT LOSS    :', (loss_cont/N).item(), 
			"\nBC LOSS      :", (loss_bc/N).item(), "\nDATA LOSS    :", (loss_data/N).item(), "\nINLET LOSS   :", (loss_inlet/N).item(), "\nTOTAL LOSS   :", total_loss.item())
			print("LAMBDA PARAMETERS:")
			print("NSE  =", f"{W_NSE.item():10.6}", "BC    =", f"{W_BC.item():10.6}", "\nDATA =", f"{W_DATA.item():10.6}", "CONT =", f"{W_CONT.item():10.6}", "INLET =", f"{W_INLET.item():10.6}")
			print('LEARNING RATE:', optim_UNN.param_groups[0]['lr'])
			print ("*"*85)

			# SAVE THE NETWORK DATA AND LOSS DATA FOR EVERY 100 EPOCHS
			# NETWORK DATA
			torch.save(P_NN.state_dict(), path + "Pressure"   + "_M_" + str(mod) + ".pt")
			torch.save(U_NN.state_dict(), path + "U_velocity" + "_M_" + str(mod) + ".pt")
			torch.save(V_NN.state_dict(), path + "V_Velocity" + "_M_" + str(mod) + ".pt")
			torch.save(W_NN.state_dict(), path + "W_Velocity" + "_M_" + str(mod) + ".pt")

			# LOSS DATA
			torch.save(Loss_NSE   [0 : epoch], path + "Loss_NSE"   + "_M_" + str(mod) + ".pt")
			torch.save(Loss_CONT  [0 : epoch], path + "Loss_CONT"  + "_M_" + str(mod) + ".pt")
			torch.save(Loss_BC    [0 : epoch], path + "Loss_BC"    + "_M_" + str(mod) + ".pt")
			torch.save(Loss_Data  [0 : epoch], path + "Loss_Data"  + "_M_" + str(mod) + ".pt")
			torch.save(Loss_Inlet [0 : epoch], path + "Loss_Inlet" + "_M_" + str(mod) + ".pt")
			
			# ADAPTIVE ACTIVATION FUNCTION
			torch.save(au1, path + "AAF_AUE" + "_M_"+ str(mod) + ".pt")
			torch.save(av1, path + "AAF_AVE" + "_M_"+ str(mod) + ".pt")
			torch.save(aw1, path + "AAF_AWE" + "_M_"+ str(mod) + ".pt")
			torch.save(ap1, path + "AAF_APE" + "_M_"+ str(mod) + ".pt")				

			# IMPOSE DYNAMIC LEARNING RATE

			if (Flag_Dyn_LR):
				# ADAPTIVE ACTIVATION FUNCTION
				scheduler_AAF.step()
	
				# LAMBDA FUNCTIONS
				scheduler_Lambda.step()
			
				# NEURAL NETWORKS
				scheduler_UNN.step()
				scheduler_VNN.step()
				scheduler_WNN.step()
				scheduler_PNN.step()
			epoch += 1

	return


# ******************************************************************************************************************** #

# ********************************************* MAIN PROGRAM STARTS HERE ********************************************* #

# ******************************************************************************************************************** #

print ("PINN PROGRAM HAS BEEN STARTED SUCCESSFULLY ....... \n")

# ***** CHOOSE A PROCESSING UNIT FOR COMPUTATION: CPU or GPU

processor = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("AVAILABLE PROCESSOR:", processor, '\n')

# ***** HYPER PARAMETERS FOR THE NEURAL NETWORK
batchsize  = 8192                                 	    # No. of data points in a whole dataset
epochs     = 750                                   	# No. of Iterations
input_VAR  = 3                                       	# No. of Flow variables
neurons    = 128                                     	# No. of Neurons in a layer

# ***** ADAPTIVE ACTIVATION FUNCTION
n 		   = 1.0  										# Scaling factor			

# ***** FILENAMES TO READ & WRITE THE DATA
mesh 		= "LDA_upstream_data_0.vtk"
sensor_file	= "LDA_upstream_data_"
bc_wall     = "Wall_LDA_Stenosis.vtk"
start_file  = 0
end_file    = 400
step_file   = 1
Nfile       = (end_file - start_file) + 1
fieldname  = 'pressure' 
Nslice      = 5

# ***** LOCATION TO READ AND WRITE THE DATA
directory 	= os.getcwd()  								# GET THE CURRENT WORKING DIRECTORY  
path        = directory + '/'
mesh      	= path + mesh
sensor_file = path + sensor_file
bc_wall     = path + bc_wall

# ***** NORMALISATION OF FLOW VARIABLES
x_scale		= 0.208662
y_scale 	= 0.326006
z_scale		= 0.317616

# ***** SENSOR MESH DATA
xmin, xmax  = 0.066163, 0.143461
ymin, ymax  = 0.144274, 0.178394
zmin, zmax  = 0.156519, 0.168698
xplane, yplane, zplane = 0.905481, 0.399689, 0.142667

# ***** FLUID PROPERTIES
density     = 1.06
diffusion   = 0.0377358

# ***** FLAGS TO IMPROVE THE USER-INTERFACE
Flag_Batch	= True  									# True  : ENABLES THE BATCH-WISE COMPUTATION
Flag_Resume = False  									# False : STARTS FROM EPOCH = 0
Flag_Dyn_LR = True 										# True  : DYNAMIC LEARNING RATE

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

''' ******************************** RESHAPE THE ARRAYS TO GET 2D-ARRAY ******************************************** '''
# WALL BOUNDARY
xbc_wall = xbc_wall.reshape(-1, 1)
ybc_wall = ybc_wall.reshape(-1, 1)
zbc_wall = zbc_wall.reshape(-1, 1)

# WALL BOUNDARY
print("SHAPE OF WALL  BC X:", xbc_wall.shape)
print("SHAPE OF WALL  BC Y:", ybc_wall.shape)
print("SHAPE OF WALL  BC Z:", zbc_wall.shape)

print ("*"*85)

''' ************************************* NORMALISATION OF VARIABLES *********************************************** '''
# MESH POINTS
x 		 = x / x_scale
y 		 = y / y_scale
z 		 = z / z_scale

# WALL BOUNDARY POINTS
xbc_wall = xbc_wall / x_scale
ybc_wall = ybc_wall / y_scale
zbc_wall = zbc_wall / z_scale

PINN(processor, x, y, z, xbc_wall, ybc_wall, zbc_wall, x_scale,	y_scale, z_scale, xmin, xmax, ymin, ymax, zmin, zmax, 
	xplane, yplane, zplane, sensor_file, fieldname, Nfile, density, diffusion, batchsize, epochs, input_VAR, neurons, 
	n, Flag_Batch, Flag_Resume, Flag_Dyn_LR, path)
