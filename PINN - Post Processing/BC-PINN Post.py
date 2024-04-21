# ********************** PHYSICS INFORMED NEURAL NETWORK WRITE PDE'S TRAINED MODEL *********************************** #
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
import matplotlib.pyplot as plt

''' *********************** JUMP TO THE MAIN PROGRAM TO CONTROL THE PROGRAM PARAMETERS ******************************* '''
def WRITE_OUTPUT_DATA(x, y, z, xup, yup, zup, x_scale, y_scale, z_scale, x_offset, vel_scale, P_max, P_min, density, 
	viscosity, input_VAR, neurons, output_file, path, output_ups):

	# MESH SPATIAL DATA
	x = torch.Tensor(x).to(processor)
	y = torch.Tensor(y).to(processor)
	z = torch.Tensor(z).to(processor)

	xup = torch.Tensor(xup).to(processor)
	yup = torch.Tensor(yup).to(processor)
	zup = torch.Tensor(zup).to(processor)


	# LAYER-WISE ADAPTIVE ACTIVATION FUNCTION

	# NETWORK I (DOMAIN - I)
	au1 = torch.load(path + "AAF_AU1.pt").to(processor)
	av1 = torch.load(path + "AAF_AV1.pt").to(processor)
	aw1 = torch.load(path + "AAF_AW1.pt").to(processor)
	ap1 = torch.load(path + "AAF_AP1.pt").to(processor)

	auBC = torch.load(path + "BCAAF_AU.pt").to(processor)
	avBC = torch.load(path + "BCAAF_AV.pt").to(processor)
	awBC = torch.load(path + "BCAAF_AW.pt").to(processor)
	apBC = torch.load(path + "BCAAF_AP.pt").to(processor)

	a  = Parameter(torch.ones(1))	
	n  = 1.0
	
# ******************************************** NEURAL NETWORK **************************************************** #

	# GATE NETWORKS FUNCTION
	
	def GATE_X(x):
		return 0.50*(torch.erf(20.0*(x - x_offset/x_scale)) + 1.0)

	def GATE_Y(y):
		return 0.50*(torch.erf(y) + 1.0)


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

	UBC.load_state_dict(torch.load(path + "UBC_velocity.pt"))
	VBC.load_state_dict(torch.load(path + "VBC_Velocity.pt"))
	WBC.load_state_dict(torch.load(path + "WBC_Velocity.pt"))
	PBC.load_state_dict(torch.load(path + "Pressure_BC.pt"))	

	U_NN.load_state_dict(torch.load(path + "U_velocity.pt"))
	V_NN.load_state_dict(torch.load(path + "V_Velocity.pt"))
	W_NN.load_state_dict(torch.load(path + "W_Velocity.pt"))
	P_NN.load_state_dict(torch.load(path + "Pressure.pt"))

	# ADAPTIVE ACTIVATION FUNCTION
	A_NN.eval()
	
	# NEURAL NETWORK
	UBC.eval()
	VBC.eval()
	WBC.eval()
	PBC.eval()

	U_NN.eval()
	V_NN.eval()
	W_NN.eval()
	P_NN.eval()

	# ****** COMPUTING THE VELOCITY FIELDS BY USING TRAINED NETWORK

	print ("WRITING THE DATA HAS BEEN STARTED")
	
	net_in = torch.cat((x.requires_grad_(),y.requires_grad_(), z.requires_grad_()),1)
	output_u = U_NN(net_in).detach()                        
	output_u = output_u.data.numpy() 
	output_v = V_NN(net_in).detach()  
	output_v = output_v.data.numpy()
	output_w = W_NN(net_in).detach()
	output_w = output_w.data.numpy()  

	Velocity = np.zeros((n_points, 3))                                  # VELOCITY VECTOR
	Velocity[:,0] = output_u[:,0] * vel_scale
	Velocity[:,1] = output_v[:,0] * vel_scale
	Velocity[:,2] = output_w[:,0] * vel_scale

	Pressure = P_NN(net_in).detach()                   					# PRESSURE
	Pressure = 0.50*(Pressure.data.numpy() * (P_max - P_min) + P_min + P_max)


	# ****** SAVING THE FIELD DATA IN VTK FORMAT

	reader = vtk.vtkUnstructuredGridReader()                           # GRID POINTS
	reader.SetFileName(mesh)
	reader.ReadAllScalarsOn()
	reader.ReadAllVectorsOn()
	reader.ReadAllTensorsOn()
	reader.Update()
	data_vtk = reader.GetOutput()
	print("MESH DATA IS WRITTEN!")

	theta_vtk = VN.numpy_to_vtk(Velocity)                               # VELOCITY
	theta_vtk.SetName('Velocity_PINN')   
	data_vtk.GetPointData().AddArray(theta_vtk)
	print("VELOCITY DATA IS WRITTEN!")

	theta_vtk = VN.numpy_to_vtk(Pressure)                               # PRESSURE
	theta_vtk.SetName('Pressure_PINN')   
	data_vtk.GetPointData().AddArray(theta_vtk)
	print("PRESSURE DATA IS WRITTEN!")

	grad = vtk.vtkGradientFilter()										# GRADIENT
	grad.SetInputData(data_vtk)
	grad.SetInputScalars(0, "Velocity_PINN")
	grad.SetResultArrayName("gradient")
	grad.Update()
	output_data = grad.GetOutput()

	myoutput = vtk.vtkDataSetWriter()
	myoutput.SetInputData(output_data)
	myoutput.SetFileName(output_file)
	myoutput.Write()

	print ("*"*85)
	print ("WRITING THE UPSTREAM DATA HAS BEEN STARTED")
	
	net_in = torch.cat((xup.requires_grad_(),yup.requires_grad_(), zup.requires_grad_()),1)
	output_u = UBC(net_in).detach()                        
	output_u = output_u.data.numpy() 
	output_v = VBC(net_in).detach()  
	output_v = output_v.data.numpy()
	output_w = WBC(net_in).detach()
	output_w = output_w.data.numpy()  

	Velocity = np.zeros((n_points_UP, 3))                                  # VELOCITY VECTOR
	Velocity[:,0] = output_u[:,0] * vel_scale
	Velocity[:,1] = output_v[:,0] * vel_scale
	Velocity[:,2] = output_w[:,0] * vel_scale

	Pressure = PBC(net_in).detach()                   					# PRESSURE
	Pressure = 0.50*(Pressure.data.numpy() * (P_max - P_min) + P_min + P_max)

	# ****** SAVING THE FIELD DATA IN VTK FORMAT

	reader = vtk.vtkUnstructuredGridReader()                           # GRID POINTS
	reader.SetFileName(mesh_up)
	reader.ReadAllScalarsOn()
	reader.ReadAllVectorsOn()
	reader.ReadAllTensorsOn()
	reader.Update()
	data_vtk = reader.GetOutput()
	print("MESH DATA IS WRITTEN!")

	theta_vtk = VN.numpy_to_vtk(Velocity)                               # VELOCITY
	theta_vtk.SetName('Velocity_PINN')   
	data_vtk.GetPointData().AddArray(theta_vtk)
	print("VELOCITY DATA IS WRITTEN!")

	theta_vtk = VN.numpy_to_vtk(Pressure)                               # PRESSURE
	theta_vtk.SetName('Pressure_PINN')   
	data_vtk.GetPointData().AddArray(theta_vtk)
	print("PRESSURE DATA IS WRITTEN!")

	myoutput = vtk.vtkDataSetWriter()
	myoutput.SetInputData(data_vtk)
	myoutput.SetFileName(output_ups)
	myoutput.Write()

	# WRITING THE LOSS FUNCTION IN A GRAPH
	Loss_NSE   = torch.load(path + "Loss_NSE.pt" ).to(processor).detach().numpy()
	Loss_CONT  = torch.load(path + "Loss_CONT.pt").to(processor).detach().numpy()
	Loss_BC    = torch.load(path + "Loss_BC.pt"  ).to(processor).detach().numpy()
	Loss_Data  = torch.load(path + "Loss_Data.pt").to(processor).detach().numpy()
	Loss_Inlet = torch.load(path + "Loss_Inlet.pt").to(processor).detach().numpy()
	Total_Loss = Loss_NSE + Loss_BC + Loss_Data + Loss_CONT + Loss_Inlet

	epochs   = Loss_NSE.size
	index    = torch.arange(epochs).to(processor).detach().numpy()
	plt.figure(1)
	plt.plot(index, Loss_NSE,   label = "NSE Loss")
	plt.plot(index, Loss_CONT,  label = "CONT Loss")
	plt.plot(index, Loss_BC,    label = "BC Loss" )
	plt.plot(index, Loss_Data,  label = "Data Loss" )
	plt.plot(index, Loss_Inlet, label = "Inlet Loss" )
	plt.plot(index, Total_Loss, label = "Total Loss")
	plt.yscale('log')
	plt.legend()
	plt.savefig('Loss_PINN.pdf')


	# WRITING THE LOSS FUNCTION IN A GRAPH
	Loss_NSE   = torch.load(path + "BCLoss_NSE.pt" ).to(processor).detach().numpy()
	Loss_CONT  = torch.load(path + "BCLoss_CONT.pt").to(processor).detach().numpy()
	Loss_BC    = torch.load(path + "BCLoss_BC.pt"  ).to(processor).detach().numpy()
	Loss_Data  = torch.load(path + "BCLoss_Data.pt").to(processor).detach().numpy()
	Total_Loss = Loss_NSE + Loss_BC + Loss_Data + Loss_CONT 

	epochs   = Loss_NSE.size
	index    = torch.arange(epochs).to(processor).detach().numpy()
	plt.figure(2)
	plt.plot(index, Loss_NSE,   label = "NSE Loss")
	plt.plot(index, Loss_CONT,  label = "CONT Loss")
	plt.plot(index, Loss_BC,    label = "BC Loss" )
	plt.plot(index, Loss_Data,  label = "Data Loss" )
	plt.plot(index, Total_Loss, label = "Total Loss")
	plt.yscale('log')
	plt.legend()
	plt.savefig('BCLoss_PINN.pdf')

	print ("********************************************************************\n")
	print ("CONVERSION FROM .PT TO VTK IS FINSIHED !!! \nAND THE OUTPUT FLOWFIELDS ARE SAVED !!!\n ")
	print ("********************************************************************")


# ******************************************************************************************************************** #

# ********************************************* MAIN PROGRAM STARTS HERE ********************************************* #

# ******************************************************************************************************************** #

print ("PINN POST PROCESS PROGRAM HAS BEEN STARTED SUCCESSFULLY!! \n")

# ***** CHOOSE A CPU FOR DATA WRITING
processor = torch.device('cpu')
print("CHOSEN PROCESSOR:", processor, '\n')

# ***** HYPER PARAMETERS FOR THE NEURAL NETWORK
input_VAR     = 3                                       # No. of Flow variables
neurons       = 128                                     # No. of Neurons in a layer

# ***** FILENAMES TO READ & WRITE THE DATA
mesh 		= "Symm_Stenosis.vtk"
mesh_up		= "Symm_Stenosis_up.vtk"
output_file = "PINN_output_data.vtk" 
output_ups  = "PINN_output_upstream.vtk" 

# ***** LOCATION TO READ AND WRITE THE DATA
directory 	= os.getcwd()  								# GET THE CURRENT WORKING DIRECTORY  
path        = directory + '/'
mesh      	= path + mesh
mesh_up   	= path + mesh_up
output_file = path + output_file
output_ups  = path + output_ups

# ***** NORMALISATION OF FLOW VARIABLES
x_scale		= 3.00
y_scale 	= 0.30
z_scale		= 0.30
P_max 		= 825.956113122666
P_min  		= -459.6687777001301
vel_scale   = 18.90947
x_offset     = 1.0

# ***** FLUID PROPERTIES
density     = 1.06
viscosity   = 0.0377358

''' *********************************************** MESH FILE ****************************************************** '''
print ("*"*85)
print ('READING THE MESH FILE: ', mesh[len(directory)+1:])
reader = vtk.vtkUnstructuredGridReader()
reader.SetFileName(mesh)
reader.Update()
data_vtk = reader.GetOutput()
n_points = data_vtk.GetNumberOfPoints()
print ('NO. OF GRID POINTS IN THE MESH FILE:' ,n_points)
x_vtk_mesh = np.zeros((n_points,1))
y_vtk_mesh = np.zeros((n_points,1))
z_vtk_mesh = np.zeros((n_points,1))
VTKpoints = vtk.vtkPoints()
for i in range(n_points):
	pt_iso  =  data_vtk.GetPoint(i)
	x_vtk_mesh[i] = pt_iso[0]   
	y_vtk_mesh[i] = pt_iso[1]
	z_vtk_mesh[i] = pt_iso[2]
	VTKpoints.InsertPoint(i, pt_iso[0], pt_iso[1], pt_iso[2])

point_data = vtk.vtkUnstructuredGrid()
point_data.SetPoints(VTKpoints)

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
n_points_UP = data.GetNumberOfPoints()
print ('NO. OF GRID POINTS IN THE MESH:', n_points_UP)
x_vtk_mesh = np.zeros((n_points_UP,1))
y_vtk_mesh = np.zeros((n_points_UP,1))
z_vtk_mesh = np.zeros((n_points_UP,1))
VTKpoints = vtk.vtkPoints()
for i in range(n_points_UP):
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

x = x/x_scale
y = y/y_scale
z = z/z_scale

xup = xup / x_scale
yup = yup / y_scale
zup = zup / z_scale


WRITE_OUTPUT_DATA(x, y, z, xup, yup, zup, x_scale, y_scale, z_scale, x_offset, vel_scale, P_max, P_min, density, 
	viscosity, input_VAR, neurons, output_file, path, output_ups)

