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
	
def PINN(processor, x1, y1, z1, x2, y2, z2, x3, y3, z3, xup, yup, zup, xbc_wall1, ybc_wall1, zbc_wall1, xbc_wall2, ybc_wall2, 
	zbc_wall2, xbc_wall3, ybc_wall3, zbc_wall3, xbc_wall_up, ybc_wall_up, zbc_wall_up, x_data1, y_data1, z_data1, x_data2, 
	y_data2, z_data2, x_data3, y_data3, z_data3, x_data_up, y_data_up, z_data_up, P_data1, P_data2, P_data3, P_data_up, 
	x_inlet, y_inlet, z_inlet, x_inter1, y_inter1, z_inter1, x_inter2, y_inter2, z_inter2, x_scale, y_scale, z_scale, 
	P_max, P_min, vel_scale, density, diffusion, Nslice, learning_rate, step_epoch, decay_rate, batchsize,	epochs, 
	BClearning_rate, BCstep_epoch, BCdecay_rate, epoch_offset, input_VAR, neurons, auBC, avBC, awBC, apBC, au1, av1, 
	aw1, ap1, n, path, Flag_Batch, Flag_Resume, Flag_Dyn_LR, W_NSE, W_CONT, W_BC, W_DATA, W_INLET) :

	# MESH SPATIAL DATA
	x1 = torch.Tensor(x1).to(processor)
	y1 = torch.Tensor(y1).to(processor)
	z1 = torch.Tensor(z1).to(processor)

	x2 = torch.Tensor(x2).to(processor)
	y2 = torch.Tensor(y2).to(processor)
	z2 = torch.Tensor(z2).to(processor)

	x3 = torch.Tensor(x3).to(processor)
	y3 = torch.Tensor(y3).to(processor)
	z3 = torch.Tensor(z3).to(processor)

	xup = torch.Tensor(xup).to(processor)
	yup = torch.Tensor(yup).to(processor)
	zup = torch.Tensor(zup).to(processor)

	# INLET WALL BOUNDARY	
	x_inlet = torch.Tensor(x_inlet).to(processor)
	y_inlet = torch.Tensor(y_inlet).to(processor)
	z_inlet = torch.Tensor(z_inlet).to(processor)

	x_inter1 = torch.Tensor(x_inter1).to(processor)
	y_inter1 = torch.Tensor(y_inter1).to(processor)
	z_inter1 = torch.Tensor(z_inter1).to(processor)

	x_inter2 = torch.Tensor(x_inter2).to(processor)
	y_inter2 = torch.Tensor(y_inter2).to(processor)
	z_inter2 = torch.Tensor(z_inter2).to(processor)

	# BC INLET PROFILE
	xbc_inlet  = np.array ([0.0655386/x_scale]).reshape (-1, 1)
	ybc_inlet  = np.array ([1.56817  /y_scale]).reshape (-1, 1)
	zbc_inlet  = np.array ([0.562097 /z_scale]).reshape (-1, 1)

	xbc_inlet  = torch.Tensor(xbc_inlet).to(processor)
	ybc_inlet  = torch.Tensor(ybc_inlet).to(processor)
	zbc_inlet  = torch.Tensor(zbc_inlet).to(processor)

	# WALL BOUNDARY
	xbc_wall1 = torch.Tensor(xbc_wall1).to(processor)
	ybc_wall1 = torch.Tensor(ybc_wall1).to(processor)
	zbc_wall1 = torch.Tensor(zbc_wall1).to(processor)

	BC_in1 = torch.cat((xbc_wall1, ybc_wall1, zbc_wall1), 1).to(processor)

	xbc_wall2 = torch.Tensor(xbc_wall2).to(processor)
	ybc_wall2 = torch.Tensor(ybc_wall2).to(processor)
	zbc_wall2 = torch.Tensor(zbc_wall2).to(processor)

	BC_in2 = torch.cat((xbc_wall2, ybc_wall2, zbc_wall2), 1).to(processor)

	xbc_wall3 = torch.Tensor(xbc_wall3).to(processor)
	ybc_wall3 = torch.Tensor(ybc_wall3).to(processor)
	zbc_wall3 = torch.Tensor(zbc_wall3).to(processor)

	BC_in3 = torch.cat((xbc_wall3, ybc_wall3, zbc_wall3), 1).to(processor)

	xbc_wall_up = torch.Tensor(xbc_wall_up).to(processor)
	ybc_wall_up = torch.Tensor(ybc_wall_up).to(processor)
	zbc_wall_up = torch.Tensor(zbc_wall_up).to(processor)

	BC_in_up = torch.cat((xbc_wall_up, ybc_wall_up, zbc_wall_up), 1).to(processor)

	# SENSOR LOCATION
	x_data1   = torch.Tensor(x_data1).to(processor)
	y_data1   = torch.Tensor(y_data1).to(processor)
	z_data1   = torch.Tensor(z_data1).to(processor)
	P_data1   = torch.Tensor(P_data1).to(processor)

	x_data2   = torch.Tensor(x_data2).to(processor)
	y_data2   = torch.Tensor(y_data2).to(processor)
	z_data2   = torch.Tensor(z_data2).to(processor)
	P_data2   = torch.Tensor(P_data2).to(processor)

	x_data3   = torch.Tensor(x_data3).to(processor)
	y_data3   = torch.Tensor(y_data3).to(processor)
	z_data3   = torch.Tensor(z_data3).to(processor)
	P_data3   = torch.Tensor(P_data3).to(processor)

	x_data_up   = torch.Tensor(x_data_up).to(processor)
	y_data_up   = torch.Tensor(y_data_up).to(processor)
	z_data_up   = torch.Tensor(z_data_up).to(processor)
	P_data_up   = torch.Tensor(P_data_up).to(processor)

	a  		 = Parameter(torch.ones(1))

	train_load_up  = DataLoader(TensorDataset(xup, yup, zup), batch_size=batchsize, shuffle=True,  drop_last = False)
	train_load1    = DataLoader(TensorDataset(x1, y1, z1),    batch_size=batchsize, shuffle=True,  drop_last = False)
	train_load2    = DataLoader(TensorDataset(x2, y2, z2),    batch_size=batchsize, shuffle=True,  drop_last = False)
	train_load3    = DataLoader(TensorDataset(x3, y3, z3),    batch_size=batchsize, shuffle=True,  drop_last = False)

	# ******************************************** NEURAL NETWORK **************************************************** #

	# GATE NETWORKS FUNCTION
	
	def GATE_1(x):
		return 0.50*(torch.erf(20.0*(x + 0.20)) + 1.0) - 0.50*(torch.erf(20.0*(x -1.50)) + 1.0)

	def GATE_2(x):
		return 0.50*(torch.erf(20.0*(x - 1.50)) + 1.0)

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

	class U_VEL_NN1(CUSTOM_SiLU):

		def __init__(self):
			super().__init__(a)
			self.domain1 = nn.Sequential(
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
			output = self.domain1(x)
			return output	

	# ****** Y-VELOCITY COMPONENT: 

	class V_VEL_NN1(CUSTOM_SiLU):
		def __init__(self):
			super().__init__(a)
			self.domain1 = nn.Sequential(
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
			output = self.domain1(x)
			return output		

	# ****** Z-VELOCITY COMPONENT: w

	class W_VEL_NN1(CUSTOM_SiLU):
		def __init__(self):
			super().__init__(a)
			self.domain1 = nn.Sequential(
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
			output = self.domain1(x)
			return output		

	# ****** PRESSURE: p
	
	class PRESS_NN1(CUSTOM_SiLU):
		def __init__(self):
			super().__init__(a)
			self.domain1 = nn.Sequential(
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
			output = self.domain1(x)
			return output


	class U_VEL_NN2(CUSTOM_SiLU):

		def __init__(self):
			super().__init__(a)
			self.domain1 = nn.Sequential(
				nn.Linear(input_VAR,neurons),

				CUSTOM_SiLU(a = au2[0, : ]),
				nn.Linear(neurons,neurons),

				CUSTOM_SiLU(a = au2[1, :]),
				nn.Linear(neurons,neurons),

				CUSTOM_SiLU(a = au2[2, :]),
				nn.Linear(neurons,neurons),

				CUSTOM_SiLU(a = au2[3, :]),
				nn.Linear(neurons,neurons),

				CUSTOM_SiLU(a = au2[4, :]),
				nn.Linear(neurons,neurons),

				CUSTOM_SiLU(a = au2[5, :]),
				nn.Linear(neurons,neurons),

				CUSTOM_SiLU(a = au2[6, :]),
				nn.Linear(neurons,neurons),

				CUSTOM_SiLU(a = au2[7, :]),
				nn.Linear(neurons,neurons),

				CUSTOM_SiLU(a = au2[8, :]),
				nn.Linear(neurons,neurons),

				CUSTOM_SiLU(a = au2[9, :]),
				nn.Linear(neurons,neurons),

				CUSTOM_SiLU(a = au2[10, :]),
				nn.Linear(neurons,1),
			)

		def forward(self, x):
			output = self.domain1(x)
			return output	

	# ****** Y-VELOCITY COMPONENT: 

	class V_VEL_NN2(CUSTOM_SiLU):
		def __init__(self):
			super().__init__(a)
			self.domain1 = nn.Sequential(
				nn.Linear(input_VAR,neurons),

				CUSTOM_SiLU(a = av2[0, :]),
				nn.Linear(neurons,neurons),

				CUSTOM_SiLU(a = av2[1, :]),
				nn.Linear(neurons,neurons),

				CUSTOM_SiLU(a = av2[2, :]),
				nn.Linear(neurons,neurons),

				CUSTOM_SiLU(a = av2[3, :]),
				nn.Linear(neurons,neurons),

				CUSTOM_SiLU(a = av2[4, :]),
				nn.Linear(neurons,neurons),

				CUSTOM_SiLU(a = av2[5, :]),
				nn.Linear(neurons,neurons),

				CUSTOM_SiLU(a = av2[6, :]),
				nn.Linear(neurons,neurons),

				CUSTOM_SiLU(a = av2[7, :]),
				nn.Linear(neurons,neurons),

				CUSTOM_SiLU(a = av2[8, :]),
				nn.Linear(neurons,neurons),

				CUSTOM_SiLU(a = av2[9, :]),
				nn.Linear(neurons,neurons),

				CUSTOM_SiLU(a = av2[10,:]),
				nn.Linear(neurons,1)
			)

		def forward(self, x):
			output = self.domain1(x)
			return output		

	# ****** Z-VELOCITY COMPONENT: w

	class W_VEL_NN2(CUSTOM_SiLU):
		def __init__(self):
			super().__init__(a)
			self.domain1 = nn.Sequential(
				nn.Linear(input_VAR,neurons),

				CUSTOM_SiLU(a = aw2[0, :]),
				nn.Linear(neurons,neurons),

				CUSTOM_SiLU(a = aw2[1, :]),
				nn.Linear(neurons,neurons),

				CUSTOM_SiLU(a = aw2[2, :]),
				nn.Linear(neurons,neurons),

				CUSTOM_SiLU(a = aw2[3, :]),
				nn.Linear(neurons,neurons),

				CUSTOM_SiLU(a = aw2[4, :]),
				nn.Linear(neurons,neurons),

				CUSTOM_SiLU(a = aw2[5, :]),
				nn.Linear(neurons,neurons),

				CUSTOM_SiLU(a = aw2[6, :]),
				nn.Linear(neurons,neurons),

				CUSTOM_SiLU(a = aw2[7, :]),
				nn.Linear(neurons,neurons),

				CUSTOM_SiLU(a = aw2[8, :]),
				nn.Linear(neurons,neurons),

				CUSTOM_SiLU(a = aw2[9, :]),
				nn.Linear(neurons,neurons),

				CUSTOM_SiLU(a = aw2[10,:]),
				nn.Linear(neurons,1)
			)
		def forward(self, x):						
			output = self.domain1(x)
			return output		

	# ****** PRESSURE: p
	
	class PRESS_NN2(CUSTOM_SiLU):
		def __init__(self):
			super().__init__(a)
			self.domain1 = nn.Sequential(
				nn.Linear(input_VAR,neurons),

				CUSTOM_SiLU(a = ap2[0, :]),
				nn.Linear(neurons,neurons),

				CUSTOM_SiLU(a = ap2[1, :]),
				nn.Linear(neurons,neurons),

				CUSTOM_SiLU(a = ap2[2, :]),
				nn.Linear(neurons,neurons),

				CUSTOM_SiLU(a = ap2[3, :]),
				nn.Linear(neurons,neurons),

				CUSTOM_SiLU(a = ap2[4, :]),
				nn.Linear(neurons,neurons),

				CUSTOM_SiLU(a = ap2[5, :]),
				nn.Linear(neurons,neurons),

				CUSTOM_SiLU(a = ap2[6, :]),
				nn.Linear(neurons,neurons),

				CUSTOM_SiLU(a = ap2[7, :]),
				nn.Linear(neurons,neurons),

				CUSTOM_SiLU(a = ap2[8, :]),
				nn.Linear(neurons,neurons),

				CUSTOM_SiLU(a = ap2[9, :]),
				nn.Linear(neurons,neurons),

				CUSTOM_SiLU(a = ap2[10, :]),
				nn.Linear(neurons,1)
			)

		def forward(self, x):
			output = self.domain1(x)
			return output

	class U_VEL_NN3(CUSTOM_SiLU):

		def __init__(self):
			super().__init__(a)
			self.domain1 = nn.Sequential(
				nn.Linear(input_VAR,neurons),

				CUSTOM_SiLU(a = au3[0, : ]),
				nn.Linear(neurons,neurons),

				CUSTOM_SiLU(a = au3[1, :]),
				nn.Linear(neurons,neurons),

				CUSTOM_SiLU(a = au3[2, :]),
				nn.Linear(neurons,neurons),

				CUSTOM_SiLU(a = au3[3, :]),
				nn.Linear(neurons,neurons),

				CUSTOM_SiLU(a = au3[4, :]),
				nn.Linear(neurons,neurons),

				CUSTOM_SiLU(a = au3[5, :]),
				nn.Linear(neurons,neurons),

				CUSTOM_SiLU(a = au3[6, :]),
				nn.Linear(neurons,neurons),

				CUSTOM_SiLU(a = au3[7, :]),
				nn.Linear(neurons,neurons),

				CUSTOM_SiLU(a = au3[8, :]),
				nn.Linear(neurons,neurons),

				CUSTOM_SiLU(a = au3[9, :]),
				nn.Linear(neurons,neurons),

				CUSTOM_SiLU(a = au3[10, :]),
				nn.Linear(neurons,1),
			)

		def forward(self, x):
			output = self.domain1(x)
			return output	

	# ****** Y-VELOCITY COMPONENT: 

	class V_VEL_NN3(CUSTOM_SiLU):
		def __init__(self):
			super().__init__(a)
			self.domain1 = nn.Sequential(
				nn.Linear(input_VAR,neurons),

				CUSTOM_SiLU(a = av3[0, :]),
				nn.Linear(neurons,neurons),

				CUSTOM_SiLU(a = av3[1, :]),
				nn.Linear(neurons,neurons),

				CUSTOM_SiLU(a = av3[2, :]),
				nn.Linear(neurons,neurons),

				CUSTOM_SiLU(a = av3[3, :]),
				nn.Linear(neurons,neurons),

				CUSTOM_SiLU(a = av3[4, :]),
				nn.Linear(neurons,neurons),

				CUSTOM_SiLU(a = av3[5, :]),
				nn.Linear(neurons,neurons),

				CUSTOM_SiLU(a = av3[6, :]),
				nn.Linear(neurons,neurons),

				CUSTOM_SiLU(a = av3[7, :]),
				nn.Linear(neurons,neurons),

				CUSTOM_SiLU(a = av3[8, :]),
				nn.Linear(neurons,neurons),

				CUSTOM_SiLU(a = av3[9, :]),
				nn.Linear(neurons,neurons),

				CUSTOM_SiLU(a = av3[10,:]),
				nn.Linear(neurons,1)
			)

		def forward(self, x):
			output = self.domain1(x)
			return output		

	# ****** Z-VELOCITY COMPONENT: w

	class W_VEL_NN3(CUSTOM_SiLU):
		def __init__(self):
			super().__init__(a)
			self.domain1 = nn.Sequential(
				nn.Linear(input_VAR,neurons),

				CUSTOM_SiLU(a = aw3[0, :]),
				nn.Linear(neurons,neurons),

				CUSTOM_SiLU(a = aw3[1, :]),
				nn.Linear(neurons,neurons),

				CUSTOM_SiLU(a = aw3[2, :]),
				nn.Linear(neurons,neurons),

				CUSTOM_SiLU(a = aw3[3, :]),
				nn.Linear(neurons,neurons),

				CUSTOM_SiLU(a = aw3[4, :]),
				nn.Linear(neurons,neurons),

				CUSTOM_SiLU(a = aw3[5, :]),
				nn.Linear(neurons,neurons),

				CUSTOM_SiLU(a = aw3[6, :]),
				nn.Linear(neurons,neurons),

				CUSTOM_SiLU(a = aw3[7, :]),
				nn.Linear(neurons,neurons),

				CUSTOM_SiLU(a = aw3[8, :]),
				nn.Linear(neurons,neurons),

				CUSTOM_SiLU(a = aw3[9, :]),
				nn.Linear(neurons,neurons),

				CUSTOM_SiLU(a = aw3[10,:]),
				nn.Linear(neurons,1)
			)
		def forward(self, x):						
			output = self.domain1(x)
			return output		

	# ****** PRESSURE: p
	
	class PRESS_NN3(CUSTOM_SiLU):
		def __init__(self):
			super().__init__(a)
			self.domain1 = nn.Sequential(
				nn.Linear(input_VAR,neurons),

				CUSTOM_SiLU(a = ap3[0, :]),
				nn.Linear(neurons,neurons),

				CUSTOM_SiLU(a = ap3[1, :]),
				nn.Linear(neurons,neurons),

				CUSTOM_SiLU(a = ap3[2, :]),
				nn.Linear(neurons,neurons),

				CUSTOM_SiLU(a = ap3[3, :]),
				nn.Linear(neurons,neurons),

				CUSTOM_SiLU(a = ap3[4, :]),
				nn.Linear(neurons,neurons),

				CUSTOM_SiLU(a = ap3[5, :]),
				nn.Linear(neurons,neurons),

				CUSTOM_SiLU(a = ap3[6, :]),
				nn.Linear(neurons,neurons),

				CUSTOM_SiLU(a = ap3[7, :]),
				nn.Linear(neurons,neurons),

				CUSTOM_SiLU(a = ap3[8, :]),
				nn.Linear(neurons,neurons),

				CUSTOM_SiLU(a = ap3[9, :]),
				nn.Linear(neurons,neurons),

				CUSTOM_SiLU(a = ap3[10, :]),
				nn.Linear(neurons,1)
			)

		def forward(self, x):
			output = self.domain1(x)
			return output

	A_NN = CUSTOM_SiLU(a).to(processor)

	UBC = UBC_NN().to(processor)
	VBC = VBC_NN().to(processor)
	WBC = WBC_NN().to(processor)
	PBC = PBC_NN().to(processor)

	# NEURAL NETWORK
	U1_NN = U_VEL_NN1().to(processor)
	V1_NN = V_VEL_NN1().to(processor)
	W1_NN = W_VEL_NN1().to(processor)
	P1_NN = PRESS_NN1().to(processor)

	U2_NN = U_VEL_NN2().to(processor)
	V2_NN = V_VEL_NN2().to(processor)
	W2_NN = W_VEL_NN2().to(processor)
	P2_NN = PRESS_NN2().to(processor)

	U3_NN = U_VEL_NN3().to(processor)
	V3_NN = V_VEL_NN3().to(processor)
	W3_NN = W_VEL_NN3().to(processor)
	P3_NN = PRESS_NN3().to(processor)

	# ****** INITIALISATION OF THE NEURAL NETWORK

	def init_normal(m):
		if isinstance(m, nn.Linear):
			nn.init.kaiming_normal_(m.weight)

	# NEURAL NETWORK
	UBC.apply(init_normal)
	VBC.apply(init_normal)
	WBC.apply(init_normal)
	PBC.apply(init_normal)

	U1_NN.apply(init_normal)
	V1_NN.apply(init_normal)
	W1_NN.apply(init_normal)
	P1_NN.apply(init_normal)

	U2_NN.apply(init_normal)
	V2_NN.apply(init_normal)
	W2_NN.apply(init_normal)
	P2_NN.apply(init_normal)

	U3_NN.apply(init_normal)
	V3_NN.apply(init_normal)
	W3_NN.apply(init_normal)
	P3_NN.apply(init_normal)


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

		
	def BC_LOSS(BC_in, model1, model2, model3) :            				# BOUNDARY CONDITION LOSS

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


	def DATA_LOSS(x_data, y_data, z_data, P_data, model):					# DATA LOSS AT THE PROBED LOCATIONS

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

		Up = torch.ones_like(U)*33.3588/vel_scale
		Vp = torch.ones_like(V)*14.7374/vel_scale
		Wp = torch.ones_like(V)*5.25452/vel_scale

		dVm_dx  = torch.autograd.grad(Vm, x,     grad_outputs=torch.ones_like(x), create_graph = True, only_inputs=True)[0]		
		dVm_dy  = torch.autograd.grad(Vm, y,     grad_outputs=torch.ones_like(y), create_graph = True, only_inputs=True)[0]
		dVm_dz  = torch.autograd.grad(Vm, z,     grad_outputs=torch.ones_like(z), create_graph = True, only_inputs=True)[0]

		loss_f = nn.MSELoss()
		loss1  = loss_f (dVm_dx, torch.zeros_like(dVm_dx)) + loss_f (dVm_dy, torch.zeros_like(dVm_dy)) + loss_f (dVm_dz, torch.zeros_like(dVm_dz))
		loss2  = loss_f (U, Up) + loss_f (V, Vp) + loss_f (W, Wp)

		return loss1 + loss2

	def INLET_LOSS(x_inlet, y_inlet, z_inlet, UNN, VNN, WNN, PNN, uNN, vNN, wNN, pNN):					    # INLET PROFILE

		INLET_in = torch.cat((x_inlet, y_inlet, z_inlet), 1)

		U = UNN(INLET_in)
		U = U.view(len(U), -1)
		
		V = VNN(INLET_in)
		V = V.view(len(V), -1)
		
		W = WNN(INLET_in)
		W = W.view(len(W), -1)

		P = PNN(INLET_in)
		P = P.view(len(P), -1)

		with torch.no_grad():
			UB = uNN(INLET_in)
			UB = UB.view(len(UB), -1)
		
			VB = vNN(INLET_in)
			VB = VB.view(len(VB), -1)
		
			WB = wNN(INLET_in)
			WB = WB.view(len(WB), -1)

			PB = pNN(INLET_in)
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

	if (INLET_TRAINING):

		# ADAPTIVE ACTIVATION FUNCTION
		optim_BCAAF = optim.Adam([auBC, avBC, awBC, apBC], lr = BClearning_rate, betas = (0.9,0.99),eps = 10**-15)

		# LAMBDA FUNCTION
		optim_Lambda  = optim.Adam([W_NSE, W_BC, W_DATA, W_CONT, W_INLET], lr = BClearning_rate, maximize=True, betas = (0.9,0.99),eps = 10**-15)

		# NEURAL NETWORKS
		optim_UBCNN = optim.Adam(UBC.parameters(),  lr = BClearning_rate, betas = (0.9,0.99),eps = 10**-15)
		optim_VBCNN = optim.Adam(VBC.parameters(),  lr = BClearning_rate, betas = (0.9,0.99),eps = 10**-15)
		optim_WBCNN = optim.Adam(WBC.parameters(),  lr = BClearning_rate, betas = (0.9,0.99),eps = 10**-15)
		optim_PBCNN = optim.Adam(PBC.parameters(),  lr = BClearning_rate, betas = (0.9,0.99),eps = 10**-15)


		# ADAPTIVE ACTIVATION FUNCTION
		scheduler_BCAAF = torch.optim.lr_scheduler.StepLR(optim_BCAAF,  step_size=BCstep_epoch, gamma=BCdecay_rate)

		scheduler_UBC = torch.optim.lr_scheduler.StepLR(optim_UBCNN,  step_size=BCstep_epoch, gamma=BCdecay_rate)
		scheduler_VBC = torch.optim.lr_scheduler.StepLR(optim_VBCNN,  step_size=BCstep_epoch, gamma=BCdecay_rate)
		scheduler_WBC = torch.optim.lr_scheduler.StepLR(optim_WBCNN,  step_size=BCstep_epoch, gamma=BCdecay_rate)
		scheduler_PBC = torch.optim.lr_scheduler.StepLR(optim_PBCNN,  step_size=BCstep_epoch, gamma=BCdecay_rate)

		scheduler_Lambda  = torch.optim.lr_scheduler.StepLR(optim_Lambda, step_size=BCstep_epoch, gamma=BCdecay_rate)

		# LOSS FUNCTIONS
		Loss_NSE    = torch.empty(size=(epoch_offset, 1))
		Loss_CONT   = torch.empty(size=(epoch_offset, 1))
		Loss_BC     = torch.empty(size=(epoch_offset, 1))
		Loss_Data   = torch.empty(size=(epoch_offset, 1))
		Loss_Inlet  = torch.empty(size=(epoch_offset, 1))

		for epoch in range (epoch_offset):
			loss_nse   = 0.
			loss_bc    = 0.
			loss_data  = 0.
			loss_cont  = 0.
			loss_inlet = 0.

			for batch_idx, (XX, YY, ZZ) in enumerate(train_load_up):

				batch_nse, batch_cont  = NSE_LOSS      (XX, YY, ZZ, UBC, VBC, WBC, PBC)
				batch_bc               = BC_LOSS       (BC_in_up, UBC, VBC, WBC)
				batch_data             = DATA_LOSS     (x_data_up, y_data_up, z_data_up, P_data_up, PBC)
				batch_inlet            = BC_INLET_LOSS (xbc_inlet, ybc_inlet, zbc_inlet)			            

				loss = W_NSE*batch_nse + W_CONT*batch_cont + W_BC*batch_bc + W_DATA*batch_data + W_INLET*batch_inlet
			
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
	

			print('TOTAL AVERAGE LOSS OF NN PER EPOCH, [EPOCH =', epoch,']: \nNSE LOSS     :', Loss_NSE[epoch].item(), '\nCONT LOSS    :', Loss_CONT[epoch].item(), 
				"\nBC LOSS      :", Loss_BC[epoch].item(), "\nDATA LOSS    :", Loss_Data[epoch].item(), "\nINLET LOSS   :", Loss_Inlet[epoch].item())
			print("LAMBDA PARAMETERS:")
			print("NSE  =", f"{W_NSE.item():10.6}", "BC    =", f"{W_BC.item():10.6}", "\nDATA =", f"{W_DATA.item():10.6}", "CONT =", f"{W_CONT.item():10.6}", "INLET =", f"{W_INLET.item():10.6}")
			print('LEARNING RATE:', optim_UBCNN.param_groups[0]['lr'])
			print ("*"*85)
		
			# SAVE THE NETWORK DATA AND LOSS DATA FOR EVERY 100 EPOCHS
			if epoch % 75 == 0:
				# NETWORK DATA
				torch.save(PBC.state_dict(),   path + "Pressure_BC.pt"  )
				torch.save(UBC.state_dict(),   path + "UBC_velocity.pt")
				torch.save(VBC.state_dict(),   path + "VBC_Velocity.pt")
				torch.save(WBC.state_dict(),   path + "WBC_Velocity.pt")

				# LOSS DATA
				torch.save(Loss_NSE   [0:epoch], path + "BCLoss_NSE.pt"  )
				torch.save(Loss_CONT  [0:epoch], path + "BCLoss_CONT.pt" )
				torch.save(Loss_BC    [0:epoch], path + "BCLoss_BC.pt"   )
				torch.save(Loss_Data  [0:epoch], path + "BCLoss_Data.pt" )
				torch.save(Loss_Inlet [0:epoch], path + "BCLoss_Inlet.pt")
			
				# ADAPTIVE ACTIVATION FUNCTION
				torch.save(auBC,   path + "BCAAF_AU.pt")
				torch.save(avBC,   path + "BCAAF_AV.pt")
				torch.save(awBC,   path + "BCAAF_AW.pt")
				torch.save(apBC,   path + "BCAAF_AP.pt")					

				print ("\n DATA SAVED.....\n ")
				print ("*"*85)

			# ADAPTIVE ACTIVATION FUNCTION
			scheduler_BCAAF.step()
	
			# LAMBDA FUNCTIONS
			scheduler_Lambda.step()
			
			# NEURAL NETWORKS
			scheduler_UBC.step()
			scheduler_VBC.step()
			scheduler_WBC.step()
			scheduler_PBC.step()

	# *******************************************  DOMAIN TRAINING  ************************************************** #

	if (DOMAIN_I):

		learning_rate = 1e-3
		step_epoch    = 500
		decay_rate    = 0.5

		optim_NNAAF = optim.Adam([au1, av1, aw1, ap1], lr = learning_rate, betas = (0.9,0.99),eps = 10**-15)

		optim_UNN = optim.Adam(U1_NN.parameters(),  lr=learning_rate, betas = (0.9,0.99),eps = 10**-15)
		optim_VNN = optim.Adam(V1_NN.parameters(),  lr=learning_rate, betas = (0.9,0.99),eps = 10**-15)
		optim_WNN = optim.Adam(W1_NN.parameters(),  lr=learning_rate, betas = (0.9,0.99),eps = 10**-15)
		optim_PNN = optim.Adam(P1_NN.parameters(),  lr=learning_rate, betas = (0.9,0.99),eps = 10**-15)


		scheduler_UNN = torch.optim.lr_scheduler.StepLR(optim_UNN,  step_size=step_epoch, gamma=decay_rate)
		scheduler_VNN = torch.optim.lr_scheduler.StepLR(optim_VNN,  step_size=step_epoch, gamma=decay_rate)
		scheduler_WNN = torch.optim.lr_scheduler.StepLR(optim_WNN,  step_size=step_epoch, gamma=decay_rate)
		scheduler_PNN = torch.optim.lr_scheduler.StepLR(optim_PNN,  step_size=step_epoch, gamma=decay_rate)	

		# NEURAL NETWORKS
		scheduler_NNAAF = torch.optim.lr_scheduler.StepLR(optim_NNAAF,  step_size=step_epoch, gamma=decay_rate)

		W_NSE   = Parameter(torch.tensor(5.0))					# NAVIER STOKES EQUATION
		W_CONT  = Parameter(torch.tensor(2.0))					# CONTINUITY EQUATION
		W_BC    = Parameter(torch.tensor(2.0))					# NOSLIP BOUNDARY CONDITION
		W_DATA  = Parameter(torch.tensor(5.0))					# SENSOR DATA
		W_INLET = Parameter(torch.tensor(2.0))					# INLET VELOCITY PROFILE	
	

		# LAMBDA FUNCTION
		optim_Lambda  = optim.Adam([W_NSE, W_BC, W_DATA, W_CONT, W_INLET], lr = learning_rate,  maximize=True, betas = (0.9,0.99),eps = 10**-15)

		scheduler_Lambda  = torch.optim.lr_scheduler.StepLR(optim_Lambda, step_size=step_epoch, gamma=decay_rate)

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

			for batch_idx, (XX, YY, ZZ) in enumerate(train_load1):

				batch_nse, batch_cont  = NSE_LOSS   (XX, YY, ZZ, U1_NN, V1_NN, W1_NN, P1_NN)
				batch_bc               = BC_LOSS    (BC_in1, U1_NN, V1_NN, W1_NN)
				batch_data             = DATA_LOSS  (x_data1, y_data1, z_data1, P_data1, P1_NN)
				batch_inlet            = INLET_LOSS (x_inlet, y_inlet, z_inlet, U1_NN, V1_NN, W1_NN, P1_NN, UBC, VBC, WBC, PBC)

				loss = W_NSE*batch_nse + W_CONT*batch_cont + W_BC*batch_bc + W_INLET*batch_inlet + W_DATA*batch_data
	
				# ADAPTIVE ACTIVATION FUNCTION
				optim_NNAAF.zero_grad()

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
					optim_NNAAF.step()

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
				torch.save(P1_NN.state_dict(),   path + "Pressure1.pt"  )
				torch.save(U1_NN.state_dict(),   path + "U1_velocity.pt")
				torch.save(V1_NN.state_dict(),   path + "V1_Velocity.pt")
				torch.save(W1_NN.state_dict(),   path + "W1_Velocity.pt")

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
			scheduler_NNAAF.step()

			scheduler_UNN.step()
			scheduler_VNN.step()
			scheduler_WNN.step()
			scheduler_PNN.step()
	
			# LAMBDA FUNCTIONS
			scheduler_Lambda.step()

	# *******************************************  DOMAIN TRAINING  ************************************************** #
	
	if (DOMAIN_II):

		learning_rate = 1e-3
		step_epoch    = 500
		decay_rate    = 0.5

		optim_NNAAF = optim.Adam([au2, av2, aw2, ap2], lr = learning_rate, betas = (0.9,0.99),eps = 10**-15)

		optim_UNN = optim.Adam(U2_NN.parameters(),  lr=learning_rate, betas = (0.9,0.99),eps = 10**-15)
		optim_VNN = optim.Adam(V2_NN.parameters(),  lr=learning_rate, betas = (0.9,0.99),eps = 10**-15)
		optim_WNN = optim.Adam(W2_NN.parameters(),  lr=learning_rate, betas = (0.9,0.99),eps = 10**-15)
		optim_PNN = optim.Adam(P2_NN.parameters(),  lr=learning_rate, betas = (0.9,0.99),eps = 10**-15)


		scheduler_UNN = torch.optim.lr_scheduler.StepLR(optim_UNN,  step_size=step_epoch, gamma=decay_rate)
		scheduler_VNN = torch.optim.lr_scheduler.StepLR(optim_VNN,  step_size=step_epoch, gamma=decay_rate)
		scheduler_WNN = torch.optim.lr_scheduler.StepLR(optim_WNN,  step_size=step_epoch, gamma=decay_rate)
		scheduler_PNN = torch.optim.lr_scheduler.StepLR(optim_PNN,  step_size=step_epoch, gamma=decay_rate)	

		# NEURAL NETWORKS
		scheduler_NNAAF = torch.optim.lr_scheduler.StepLR(optim_NNAAF,  step_size=step_epoch, gamma=decay_rate)

		W_NSE   = Parameter(torch.tensor(5.0))					# NAVIER STOKES EQUATION
		W_CONT  = Parameter(torch.tensor(2.0))					# CONTINUITY EQUATION
		W_BC    = Parameter(torch.tensor(2.0))					# NOSLIP BOUNDARY CONDITION
		W_DATA  = Parameter(torch.tensor(5.0))					# SENSOR DATA
		W_INLET = Parameter(torch.tensor(2.0))					# INLET VELOCITY PROFILE	
	

		# LAMBDA FUNCTION
		optim_Lambda  = optim.Adam([W_NSE, W_BC, W_DATA, W_CONT, W_INLET], lr = learning_rate,  maximize=True, betas = (0.9,0.99),eps = 10**-15)

		scheduler_Lambda  = torch.optim.lr_scheduler.StepLR(optim_Lambda, step_size=step_epoch, gamma=decay_rate)

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

			for batch_idx, (XX, YY, ZZ) in enumerate(train_load2):

				batch_nse, batch_cont  = NSE_LOSS   (XX, YY, ZZ, U2_NN, V2_NN, W2_NN, P2_NN)
				batch_bc               = BC_LOSS    (BC_in2, U2_NN, V2_NN, W2_NN)
				batch_data             = DATA_LOSS  (x_data2, y_data2, z_data2, P_data2, P2_NN)
				batch_inlet            = INLET_LOSS (x_inter1, y_inter1, z_inter1, U2_NN, V2_NN, W2_NN, P2_NN, U1_NN, V1_NN, W1_NN, P1_NN)

				loss = W_NSE*batch_nse + W_CONT*batch_cont + W_BC*batch_bc + W_INLET*batch_inlet + W_DATA*batch_data
	
				# ADAPTIVE ACTIVATION FUNCTION
				optim_NNAAF.zero_grad()

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
					optim_NNAAF.step()

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
				torch.save(P2_NN.state_dict(),   path + "Pressure2.pt"  )
				torch.save(U2_NN.state_dict(),   path + "U2_velocity.pt")
				torch.save(V2_NN.state_dict(),   path + "V2_Velocity.pt")
				torch.save(W2_NN.state_dict(),   path + "W2_Velocity.pt")

				# LOSS DATA
				torch.save(Loss_NSE  [0:epoch], path + "Loss_NSE.pt"  )
				torch.save(Loss_CONT [0:epoch], path + "Loss_CONT.pt" )
				torch.save(Loss_BC   [0:epoch], path + "Loss_BC.pt"   )
				torch.save(Loss_Data [0:epoch], path + "Loss_Data.pt" )
				torch.save(Loss_Inlet[0:epoch], path + "Loss_Inlet.pt")

				# ADAPTIVE ACTIVATION FUNCTION
				torch.save(au2,   path + "AAF_AU2.pt")
				torch.save(av2,   path + "AAF_AV2.pt")
				torch.save(aw2,   path + "AAF_AW2.pt")
				torch.save(ap2,   path + "AAF_AP2.pt")					

				print ("\n DATA SAVED.....\n ")
				print ("*"*85)

			# IMPOSE DYNAMIC LEARNING RATE
			scheduler_NNAAF.step()

			scheduler_UNN.step()
			scheduler_VNN.step()
			scheduler_WNN.step()
			scheduler_PNN.step()
	
			# LAMBDA FUNCTIONS
			scheduler_Lambda.step()


	if (DOMAIN_III):

		learning_rate = 1e-3
		step_epoch    = 500
		decay_rate    = 0.5

		optim_NNAAF = optim.Adam([au3, av3, aw3, ap3], lr = learning_rate, betas = (0.9,0.99),eps = 10**-15)

		optim_UNN = optim.Adam(U3_NN.parameters(),  lr=learning_rate, betas = (0.9,0.99),eps = 10**-15)
		optim_VNN = optim.Adam(V3_NN.parameters(),  lr=learning_rate, betas = (0.9,0.99),eps = 10**-15)
		optim_WNN = optim.Adam(W3_NN.parameters(),  lr=learning_rate, betas = (0.9,0.99),eps = 10**-15)
		optim_PNN = optim.Adam(P3_NN.parameters(),  lr=learning_rate, betas = (0.9,0.99),eps = 10**-15)


		scheduler_UNN = torch.optim.lr_scheduler.StepLR(optim_UNN,  step_size=step_epoch, gamma=decay_rate)
		scheduler_VNN = torch.optim.lr_scheduler.StepLR(optim_VNN,  step_size=step_epoch, gamma=decay_rate)
		scheduler_WNN = torch.optim.lr_scheduler.StepLR(optim_WNN,  step_size=step_epoch, gamma=decay_rate)
		scheduler_PNN = torch.optim.lr_scheduler.StepLR(optim_PNN,  step_size=step_epoch, gamma=decay_rate)	
		scheduler_NNAAF = torch.optim.lr_scheduler.StepLR(optim_NNAAF,  step_size=step_epoch, gamma=decay_rate)

		W_NSE   = Parameter(torch.tensor(5.0))					# NAVIER STOKES EQUATION
		W_CONT  = Parameter(torch.tensor(2.0))					# CONTINUITY EQUATION
		W_BC    = Parameter(torch.tensor(2.0))					# NOSLIP BOUNDARY CONDITION
		W_DATA  = Parameter(torch.tensor(5.0))					# SENSOR DATA
		W_INLET = Parameter(torch.tensor(2.0))					# INLET VELOCITY PROFILE	
	

		# LAMBDA FUNCTION
		optim_Lambda  = optim.Adam([W_NSE, W_BC, W_DATA, W_CONT, W_INLET], lr = learning_rate,  maximize=True, betas = (0.9,0.99),eps = 10**-15)

		scheduler_Lambda  = torch.optim.lr_scheduler.StepLR(optim_Lambda, step_size=step_epoch, gamma=decay_rate)

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

			for batch_idx, (XX, YY, ZZ) in enumerate(train_load3):

				batch_nse, batch_cont  = NSE_LOSS   (XX, YY, ZZ, U3_NN, V3_NN, W3_NN, P3_NN)
				batch_bc               = BC_LOSS    (BC_in3, U3_NN, V3_NN, W3_NN)
				batch_data             = DATA_LOSS  (x_data3, y_data3, z_data3, P_data3, P3_NN)
				batch_inlet            = INLET_LOSS (x_inter2, y_inter2, z_inter2, U3_NN, V3_NN, W3_NN, P3_NN, U2_NN, V2_NN, W2_NN, P2_NN)

				loss = W_NSE*batch_nse + W_CONT*batch_cont + W_BC*batch_bc + W_INLET*batch_inlet + W_DATA*batch_data
	
				# ADAPTIVE ACTIVATION FUNCTION
				optim_NNAAF.zero_grad()

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
					optim_NNAAF.step()

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
				torch.save(P3_NN.state_dict(),   path + "Pressure3.pt"  )
				torch.save(U3_NN.state_dict(),   path + "U3_velocity.pt")
				torch.save(V3_NN.state_dict(),   path + "V3_Velocity.pt")
				torch.save(W3_NN.state_dict(),   path + "W3_Velocity.pt")

				# LOSS DATA
				torch.save(Loss_NSE  [0:epoch], path + "Loss_NSE.pt"  )
				torch.save(Loss_CONT [0:epoch], path + "Loss_CONT.pt" )
				torch.save(Loss_BC   [0:epoch], path + "Loss_BC.pt"   )
				torch.save(Loss_Data [0:epoch], path + "Loss_Data.pt" )
				torch.save(Loss_Inlet[0:epoch], path + "Loss_Inlet.pt")

				# ADAPTIVE ACTIVATION FUNCTION
				torch.save(au3,   path + "AAF_AU3.pt")
				torch.save(av3,   path + "AAF_AV3.pt")
				torch.save(aw3,   path + "AAF_AW3.pt")
				torch.save(ap3,   path + "AAF_AP3.pt")					

				print ("\n DATA SAVED.....\n ")
				print ("*"*85)

			# IMPOSE DYNAMIC LEARNING RATE
			scheduler_NNAAF.step()

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
batchsize    = 12288	                               	# No. of data points in a whole dataset
epoch_offset = 751									# No. of Iterations fror UPSTREAM
epochs       = 10000                    				# No. of Iterations
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

au2  = Parameter(torch.ones(11, neurons), requires_grad= False).to(processor)	
av2  = Parameter(torch.ones(11, neurons), requires_grad= False).to(processor)		
aw2  = Parameter(torch.ones(11, neurons), requires_grad= False).to(processor)											
ap2  = Parameter(torch.ones(11, neurons), requires_grad= False).to(processor)

au3  = Parameter(torch.ones(11, neurons), requires_grad= False).to(processor)	
av3  = Parameter(torch.ones(11, neurons), requires_grad= False).to(processor)		
aw3  = Parameter(torch.ones(11, neurons), requires_grad= False).to(processor)											
ap3  = Parameter(torch.ones(11, neurons), requires_grad= False).to(processor)
				

# ***** FILENAMES TO READ & WRITE THE DATA
mesh_up		= "LDA_Stenosis_up.vtk"
mesh_1 		= "LDA_Stenosis_Domain_1.vtk"
mesh_2 		= "LDA_Stenosis_Domain_2.vtk"
mesh_3 		= "LDA_Stenosis_Domain_3.vtk"
inlet_file  = "Inlet_LDA_Stenosis_FULL.vtk"
interface1  = "Interface_1.vtk"
interface2  = "Interface_2.vtk"
bc_wall_up  = "Wall_LDA_Stenosis_up.vtk"
bc_wall_1   = "Wall_LDA_Stenosis_Domain_1.vtk"
bc_wall_2   = "Wall_LDA_Stenosis_Domain_2.vtk"
bc_wall_3   = "Wall_LDA_Stenosis_Domain_3.vtk"
sensor_file	= "LDA_Stenosis_FULL.vtk"

# ***** LOCATION TO READ AND WRITE THE DATA
directory 	= os.getcwd()  								# GET THE CURRENT WORKING DIRECTORY  
path        = directory + '/'
mesh_1      = path + mesh_1
mesh_2      = path + mesh_2
mesh_3      = path + mesh_3
mesh_up   	= path + mesh_up
inlet_file  = path + inlet_file
interface1  = path + interface1
interface2  = path + interface2
bc_wall_1   = path + bc_wall_1
bc_wall_2   = path + bc_wall_2
bc_wall_3   = path + bc_wall_3
bc_wall_up  = path + bc_wall_up 
sensor_file	= path + sensor_file

# ***** NORMALISATION OF FLOW VARIABLES
x_scale		= 2.3776
y_scale 	= 1.82699
z_scale		= 1.00065
P_max 		= 2345.359
P_min  		= -76.4930
vel_scale   = 36.5487
x_offset    = 1.0

# ***** FLUID PROPERTIES
density     = 1.06
diffusion   = 0.0377358

# ***** UPSTREAM DOMAIN
Nslice_up    = 12

# ***** FLAGS TO IMPROVE THE USER-INTERFACE
Flag_Batch	= True  									# True  : ENABLES THE BATCH-WISE COMPUTATION
Flag_Resume = False  									# False : STARTS FROM EPOCH = 0
Flag_Dyn_LR = True 										# True  : DYNAMIC LEARNING RATE

INLET_TRAINING = True
DOMAIN_I       = True
DOMAIN_II      = True
DOMAIN_III     = True

if (Flag_Dyn_LR):
	learning_rate = 1e-3
	step_epoch    = 500
	decay_rate    = 0.5

	BClearning_rate = 1e-3
	BCstep_epoch    = 75
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
print ('READING THE MESH FILE: ', mesh_1[len(directory)+1:])
reader = vtk.vtkUnstructuredGridReader()
reader.SetFileName(mesh_1)
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

x1  = np.reshape(x_vtk_mesh , (np.size(x_vtk_mesh [:]),1)) 
y1  = np.reshape(y_vtk_mesh , (np.size(y_vtk_mesh [:]),1))
z1  = np.reshape(z_vtk_mesh , (np.size(z_vtk_mesh [:]),1))

print ("SHAPE OF X:", 	x1.shape)
print ("SHAPE OF Y:", 	y1.shape)
print ("SHAPE OF Z:", 	z1.shape)

print ("*"*85)
print ('READING THE MESH FILE: ', mesh_2[len(directory)+1:])
reader = vtk.vtkUnstructuredGridReader()
reader.SetFileName(mesh_2)
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

x2  = np.reshape(x_vtk_mesh , (np.size(x_vtk_mesh [:]),1)) 
y2  = np.reshape(y_vtk_mesh , (np.size(y_vtk_mesh [:]),1))
z2  = np.reshape(z_vtk_mesh , (np.size(z_vtk_mesh [:]),1))

print ("SHAPE OF X:", 	x2.shape)
print ("SHAPE OF Y:", 	y2.shape)
print ("SHAPE OF Z:", 	z2.shape)

print ('READING THE MESH FILE: ', mesh_3[len(directory)+1:])
reader = vtk.vtkUnstructuredGridReader()
reader.SetFileName(mesh_3)
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

x3  = np.reshape(x_vtk_mesh , (np.size(x_vtk_mesh [:]),1)) 
y3  = np.reshape(y_vtk_mesh , (np.size(y_vtk_mesh [:]),1))
z3  = np.reshape(z_vtk_mesh , (np.size(z_vtk_mesh [:]),1))

print ("SHAPE OF X:", 	x3.shape)
print ("SHAPE OF Y:", 	y3.shape)
print ("SHAPE OF Z:", 	z3.shape)

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

print ('READING THE WALL BOUNDARY FILE:', bc_wall_1[len(directory)+1:])
reader =  vtk.vtkPolyDataReader()
reader.SetFileName(bc_wall_1)
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
xbc_wall1  = np.reshape(x_vtk_mesh , (np.size(x_vtk_mesh [:]),1)) 
ybc_wall1  = np.reshape(y_vtk_mesh , (np.size(y_vtk_mesh [:]),1))
zbc_wall1  = np.reshape(z_vtk_mesh , (np.size(z_vtk_mesh [:]),1))

print ("*"*85)

print ('READING THE WALL BOUNDARY FILE:', bc_wall_2[len(directory)+1:])
reader =  vtk.vtkPolyDataReader()
reader.SetFileName(bc_wall_2)
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
xbc_wall2  = np.reshape(x_vtk_mesh , (np.size(x_vtk_mesh [:]),1)) 
ybc_wall2  = np.reshape(y_vtk_mesh , (np.size(y_vtk_mesh [:]),1))
zbc_wall2  = np.reshape(z_vtk_mesh , (np.size(z_vtk_mesh [:]),1))

print ("*"*85)

print ('READING THE WALL BOUNDARY FILE:', bc_wall_3[len(directory)+1:])
reader =  vtk.vtkPolyDataReader()
reader.SetFileName(bc_wall_3)
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
xbc_wall3  = np.reshape(x_vtk_mesh , (np.size(x_vtk_mesh [:]),1)) 
ybc_wall3  = np.reshape(y_vtk_mesh , (np.size(y_vtk_mesh [:]),1))
zbc_wall3  = np.reshape(z_vtk_mesh , (np.size(z_vtk_mesh [:]),1))

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

xl = torch.linspace(0.083, 1.00, steps = Nslice)

x_data1 = np.zeros (Nslice)
y_data1 = np.zeros (Nslice)
z_data1 = np.zeros (Nslice)

for i in range(Nslice):

	#****** PLANE CREATION
	plane = vtk.vtkPlane()
	plane.SetOrigin(xl [i], 0.9135, 0.50032)								# LOCATION OF SLICE ALONG X DIRECTION
	plane.SetNormal(1, 0, 0)							# SLICE IN YZ PLANE 
	
	#****** SLICE THE MESH AT THE CHOSEN PLANE
	cutter = vtk.vtkCutter()
	cutter.SetCutFunction(plane)
	cutter.SetInputConnection(reader.GetOutputPort())
	cutter.Update()

	#****** INTEGRATE THE VARIABLES TO GET CENTROID
	integrate = vtk.vtkIntegrateAttributes()
	integrate.SetInputConnection(cutter.GetOutputPort())
	integrate.Update()
	x_data1 [i] = integrate.GetOutput().GetBounds()[0]
	y_data1 [i] = integrate.GetOutput().GetBounds()[2]
	z_data1 [i] = integrate.GetOutput().GetBounds()[4]

data_vtk = reader.GetOutput()
n_point  = data_vtk.GetNumberOfPoints()
print ('NO. OF DATA POINTS IN THE SENSOR FILE:', n_point)

VTKpoints = vtk.vtkPoints()
for i in range(len(x_data1)): 
	VTKpoints.InsertPoint(i, x_data1[i] , y_data1[i]  , z_data1[i])

point_data = vtk.vtkUnstructuredGrid()
point_data.SetPoints(VTKpoints)
probe = vtk.vtkProbeFilter()
probe.SetInputData(point_data)
probe.SetSourceData(data_vtk)
probe.Update()
array  = probe.GetOutput().GetPointData().GetArray(fieldname)
P_data1 = VN.vtk_to_numpy(array)

print ("*"*85)

print ("READING THE SENSOR DATA FILE:", sensor_file[len(directory)+1:] )
reader = vtk.vtkUnstructuredGridReader()
reader.SetFileName(sensor_file)
reader.Update()

xl = torch.linspace(1.00, 1.4801, steps = Nslice)

x_data2 = np.zeros (Nslice)
y_data2 = np.zeros (Nslice)
z_data2 = np.zeros (Nslice)

for i in range(Nslice):

	#****** PLANE CREATION
	plane = vtk.vtkPlane()
	plane.SetOrigin(xl [i], 0.9135, 0.50032)								# LOCATION OF SLICE ALONG X DIRECTION
	plane.SetNormal(1, 0, 0)							# SLICE IN YZ PLANE 
	
	#****** SLICE THE MESH AT THE CHOSEN PLANE
	cutter = vtk.vtkCutter()
	cutter.SetCutFunction(plane)
	cutter.SetInputConnection(reader.GetOutputPort())
	cutter.Update()

	#****** INTEGRATE THE VARIABLES TO GET CENTROID
	integrate = vtk.vtkIntegrateAttributes()
	integrate.SetInputConnection(cutter.GetOutputPort())
	integrate.Update()
	x_data2 [i] = integrate.GetOutput().GetBounds()[0]
	y_data2 [i] = integrate.GetOutput().GetBounds()[2]
	z_data2 [i] = integrate.GetOutput().GetBounds()[4]

data_vtk = reader.GetOutput()
n_point  = data_vtk.GetNumberOfPoints()
print ('NO. OF DATA POINTS IN THE SENSOR FILE:', n_point)

VTKpoints = vtk.vtkPoints()
for i in range(len(x_data2)): 
	VTKpoints.InsertPoint(i, x_data2[i] , y_data2[i]  , z_data2[i])

point_data = vtk.vtkUnstructuredGrid()
point_data.SetPoints(VTKpoints)
probe = vtk.vtkProbeFilter()
probe.SetInputData(point_data)
probe.SetSourceData(data_vtk)
probe.Update()
array  = probe.GetOutput().GetPointData().GetArray(fieldname)
P_data2 = VN.vtk_to_numpy(array)

print ("*"*85)


print ("READING THE SENSOR DATA FILE:", sensor_file[len(directory)+1:] )
reader = vtk.vtkUnstructuredGridReader()
reader.SetFileName(sensor_file)
reader.Update()

xl = torch.linspace(1.4801, 2.250, steps = Nslice)

x_data3 = np.zeros (Nslice)
y_data3 = np.zeros (Nslice)
z_data3 = np.zeros (Nslice)

for i in range(Nslice):

	#****** PLANE CREATION
	plane = vtk.vtkPlane()
	plane.SetOrigin(xl [i], 0.9135, 0.50032)								# LOCATION OF SLICE ALONG X DIRECTION
	plane.SetNormal(1, 0, 0)							# SLICE IN YZ PLANE 
	
	#****** SLICE THE MESH AT THE CHOSEN PLANE
	cutter = vtk.vtkCutter()
	cutter.SetCutFunction(plane)
	cutter.SetInputConnection(reader.GetOutputPort())
	cutter.Update()

	#****** INTEGRATE THE VARIABLES TO GET CENTROID
	integrate = vtk.vtkIntegrateAttributes()
	integrate.SetInputConnection(cutter.GetOutputPort())
	integrate.Update()
	x_data3 [i] = integrate.GetOutput().GetBounds()[0]
	y_data3 [i] = integrate.GetOutput().GetBounds()[2]
	z_data3 [i] = integrate.GetOutput().GetBounds()[4]

data_vtk = reader.GetOutput()
n_point  = data_vtk.GetNumberOfPoints()
print ('NO. OF DATA POINTS IN THE SENSOR FILE:', n_point)

VTKpoints = vtk.vtkPoints()
for i in range(len(x_data3)): 
	VTKpoints.InsertPoint(i, x_data3[i] , y_data3[i]  , z_data3[i])

point_data = vtk.vtkUnstructuredGrid()
point_data.SetPoints(VTKpoints)
probe = vtk.vtkProbeFilter()
probe.SetInputData(point_data)
probe.SetSourceData(data_vtk)
probe.Update()
array  = probe.GetOutput().GetPointData().GetArray(fieldname)
P_data3 = VN.vtk_to_numpy(array)

print ("*"*85)

del data, data_vtk, reader, point_data, x_vtk_mesh, y_vtk_mesh, z_vtk_mesh, VTKpoints

print ("READING THE SENSOR DATA FILE:", sensor_file[len(directory)+1:] )
reader = vtk.vtkUnstructuredGridReader()
reader.SetFileName(sensor_file)
reader.Update()

xl = torch.linspace(0.33757, 0.38438, steps = Nslice_up)
yl = torch.linspace(1.02029, 1.04192, steps = Nslice_up)
zl = torch.linspace(0.49873, 0.50702, steps = Nslice_up)

x_data_up = np.zeros (Nslice_up)
y_data_up = np.zeros (Nslice_up)
z_data_up = np.zeros (Nslice_up)

for i in range(Nslice_up):

	#****** PLANE CREATION
	plane = vtk.vtkPlane()
	plane.SetOrigin(xl[i], yl[i], zl[i])								# LOCATION OF SLICE ALONG X DIRECTION
	plane.SetNormal(-0.89626, -0.414236, -0.15879)									# SLICE IN YZ PLANE 
	
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

print ('READING THE INLET WALL FILE:', interface1[len(directory)+1:])
reader =  vtk.vtkPolyDataReader()
reader.SetFileName(interface1)
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
x_inter1  = np.reshape(x_vtk_mesh , (np.size(x_vtk_mesh [:]),1)) 
y_inter1  = np.reshape(y_vtk_mesh , (np.size(y_vtk_mesh [:]),1))
z_inter1  = np.reshape(z_vtk_mesh , (np.size(z_vtk_mesh [:]),1))

print ("*"*85)


print ('READING THE INLET WALL FILE:', interface2[len(directory)+1:])
reader =  vtk.vtkPolyDataReader()
reader.SetFileName(interface2)
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
x_inter2  = np.reshape(x_vtk_mesh , (np.size(x_vtk_mesh [:]),1)) 
y_inter2  = np.reshape(y_vtk_mesh , (np.size(y_vtk_mesh [:]),1))
z_inter2  = np.reshape(z_vtk_mesh , (np.size(z_vtk_mesh [:]),1))

print ("*"*85)
''' ******************************** RESHAPE THE ARRAYS TO GET 2D-ARRAY ******************************************** '''
# WALL BOUNDARY
xbc_wall1 = xbc_wall1.reshape(-1, 1)
ybc_wall1 = ybc_wall1.reshape(-1, 1)
zbc_wall1 = zbc_wall1.reshape(-1, 1)

xbc_wall2 = xbc_wall2.reshape(-1, 1)
ybc_wall2 = ybc_wall2.reshape(-1, 1)
zbc_wall2 = zbc_wall2.reshape(-1, 1)

xbc_wall3 = xbc_wall3.reshape(-1, 1)
ybc_wall3 = ybc_wall3.reshape(-1, 1)
zbc_wall3 = zbc_wall3.reshape(-1, 1)

xbc_wall_up = xbc_wall_up.reshape(-1, 1)
ybc_wall_up = ybc_wall_up.reshape(-1, 1)
zbc_wall_up = zbc_wall_up.reshape(-1, 1)

# INLET BOUNDARY
x_inlet = x_inlet.reshape(-1, 1)
y_inlet = y_inlet.reshape(-1, 1)
z_inlet = z_inlet.reshape(-1, 1)

x_inter1 = x_inter1.reshape(-1, 1)
y_inter1 = y_inter1.reshape(-1, 1)
z_inter1 = z_inter1.reshape(-1, 1)

x_inter2 = x_inter2.reshape(-1, 1)
y_inter2 = y_inter2.reshape(-1, 1)
z_inter2 = z_inter2.reshape(-1, 1)

# SENSOR DATA
x_data1 = x_data1.reshape(-1, 1) 
y_data1 = y_data1.reshape(-1, 1)
z_data1 = z_data1.reshape(-1, 1) 
P_data1 = P_data1.reshape(-1, 1)

x_data2 = x_data2.reshape(-1, 1) 
y_data2 = y_data2.reshape(-1, 1)
z_data2 = z_data2.reshape(-1, 1) 
P_data2 = P_data2.reshape(-1, 1)

x_data3 = x_data3.reshape(-1, 1) 
y_data3 = y_data3.reshape(-1, 1)
z_data3 = z_data3.reshape(-1, 1) 
P_data3 = P_data3.reshape(-1, 1)

x_data_up = x_data_up.reshape(-1, 1) 
y_data_up = y_data_up.reshape(-1, 1)
z_data_up = z_data_up.reshape(-1, 1) 
P_data_up = P_data_up.reshape(-1, 1) 

# WALL BOUNDARY
print("SHAPE OF WALL  BC X:", xbc_wall1.shape)
print("SHAPE OF WALL  BC Y:", ybc_wall1.shape)
print("SHAPE OF WALL  BC Z:", zbc_wall1.shape)

print("SHAPE OF WALL  BC X:", xbc_wall2.shape)
print("SHAPE OF WALL  BC Y:", ybc_wall2.shape)
print("SHAPE OF WALL  BC Z:", zbc_wall2.shape)

print("SHAPE OF WALL  BC X:", xbc_wall3.shape)
print("SHAPE OF WALL  BC Y:", ybc_wall3.shape)
print("SHAPE OF WALL  BC Z:", zbc_wall3.shape)

print ("*"*85)

''' ************************************* NORMALISATION OF VARIABLES *********************************************** '''
# MESH POINTS
x1 	= x1 / x_scale
y1 	= y1 / y_scale
z1 	= z1 / z_scale

x2 	= x2 / x_scale
y2 	= y2 / y_scale
z2 	= z2 / z_scale

x3 	= x3 / x_scale
y3 	= y3 / y_scale
z3 	= z3 / z_scale

xup = xup / x_scale
yup = yup / y_scale
zup = zup / z_scale

# WALL BOUNDARY POINTS
xbc_wall1 = xbc_wall1 / x_scale
ybc_wall1 = ybc_wall1 / y_scale
zbc_wall1 = zbc_wall1 / z_scale

xbc_wall2 = xbc_wall2 / x_scale
ybc_wall2 = ybc_wall2 / y_scale
zbc_wall2 = zbc_wall2 / z_scale

xbc_wall3 = xbc_wall3 / x_scale
ybc_wall3 = ybc_wall3 / y_scale
zbc_wall3 = zbc_wall3 / z_scale

xbc_wall_up = xbc_wall_up / x_scale
ybc_wall_up = ybc_wall_up / y_scale
zbc_wall_up = zbc_wall_up / z_scale

# INLET BOUNDARY POINTS
x_inlet = x_inlet / x_scale
y_inlet = y_inlet / y_scale
z_inlet = z_inlet / z_scale

x_inter1 = x_inter1 /x_scale
y_inter1 = y_inter1 /y_scale
z_inter1 = z_inter1 /z_scale

x_inter2 = x_inter2 /x_scale
y_inter2 = y_inter2 /y_scale
z_inter2 = z_inter2 /z_scale

# SENSOR DATA POINTS
x_data1	 = x_data1 / x_scale
y_data1  = y_data1 / y_scale
z_data1  = z_data1 / z_scale
P_data1  = (P_data1 - P_min)/(P_max - P_min) + (P_data1 - P_max)/(P_max - P_min)

x_data2	 = x_data2 / x_scale
y_data2  = y_data2 / y_scale
z_data2  = z_data2 / z_scale
P_data2  = (P_data2 - P_min)/(P_max - P_min) + (P_data2 - P_max)/(P_max - P_min)

x_data3	 = x_data2 / x_scale
y_data3  = y_data2 / y_scale
z_data3  = z_data2 / z_scale
P_data3  = (P_data3 - P_min)/(P_max - P_min) + (P_data3 - P_max)/(P_max - P_min)

x_data_up = x_data_up / x_scale
y_data_up = y_data_up / y_scale
z_data_up = z_data_up / z_scale
P_data_up = (P_data_up - P_min)/(P_max - P_min) + (P_data_up - P_max)/(P_max - P_min)


PINN(processor, x1, y1, z1, x2, y2, z2, x3, y3, z3, xup, yup, zup, xbc_wall1, ybc_wall1, zbc_wall1, xbc_wall2, ybc_wall2, 
	zbc_wall2, xbc_wall3, ybc_wall3, zbc_wall3, xbc_wall_up, ybc_wall_up, zbc_wall_up, x_data1, y_data1, z_data1, x_data2, 
	y_data2, z_data2, x_data3, y_data3, z_data3, x_data_up, y_data_up, z_data_up, P_data1, P_data2, P_data3, P_data_up, 
	x_inlet, y_inlet, z_inlet, x_inter1, y_inter1, z_inter1, x_inter2, y_inter2, z_inter2, x_scale, y_scale, z_scale, 
	P_max, P_min, vel_scale, density, diffusion, Nslice, learning_rate, step_epoch, decay_rate, batchsize,	epochs, 
	BClearning_rate, BCstep_epoch, BCdecay_rate, epoch_offset, input_VAR, neurons, auBC, avBC, awBC, apBC, au1, av1, 
	aw1, ap1, n, path, Flag_Batch, Flag_Resume, Flag_Dyn_LR, W_NSE, W_CONT, W_BC, W_DATA, W_INLET)

