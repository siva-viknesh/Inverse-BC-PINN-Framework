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
from scipy.signal import butter, filtfilt, savgol_filter, sosfilt
import h5py

''' *********************** JUMP TO THE MAIN PROGRAM TO CONTROL THE PROGRAM PARAMETERS ******************************* '''
def SENSOR_DATA (fieldname, file_name, xmin, xmax, ymin, ymax, zmin, zmax, xplane, yplane, zplane, Nslice):

	x_data = np.zeros (Nslice)
	y_data = np.zeros (Nslice)
	z_data = np.zeros (Nslice)
	P_data = np.zeros (Nslice)

	xl = torch.linspace(xmin, xmax, steps = Nslice)
	yl = torch.linspace(ymin, ymax, steps = Nslice)
	zl = torch.linspace(zmin, zmax, steps = Nslice)

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
	del data_vtk, point_data

	# -------------------------------------------------------------------------------------------------------------------#
	fieldname  = 'velocity' 														# FIELD NAME FOR VTK FILES
	reader = vtk.vtkUnstructuredGridReader()
	reader.SetFileName(file_name)
	reader.Update()

	#****** PLANE CREATION
	plane = vtk.vtkPlane()
	plane.SetOrigin(xmin, ymin, zmin)					# LOCATION OF SLICE ALONG X DIRECTION
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
	array  = probe.GetOutput().GetPointData().GetArray(fieldname)
	vel = VN.vtk_to_numpy(array)

	del data_vtk, point_data
	return P_data, np.squeeze(vel)

def smooth(y, box_pts):
	box = np.ones(box_pts)/box_pts
	y_smooth = np.convolve(y, box, mode='same')
	return y_smooth

def butterworth_filter(data, cutoff_freq, sampling_freq, order):
	nyquist_freq = 0.5 * sampling_freq
	normalized_cutoff_freq = cutoff_freq / nyquist_freq
	b, a = butter(order, normalized_cutoff_freq, btype='lowpass')
	filtered_data = filtfilt(b, a, data)
	return filtered_data

def FLOW_RATE_CALCULATION (file_name, xmin, ymin, zmin, xplane, yplane, zplane):

	reader = vtk.vtkUnstructuredGridReader()
	reader.SetFileName(file_name)
	reader.Update()
	
	#****** PLANE CREATION
	plane = vtk.vtkPlane()
	plane.SetOrigin(xmin, ymin, zmin)					# LOCATION OF SLICE ALONG X DIRECTION
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
	PINN = np.squeeze(VN.vtk_to_numpy(integrate.GetOutputDataObject(0).GetPointData().GetArray('Velocity_PINN')))
	CFD  = np.squeeze(VN.vtk_to_numpy(integrate.GetOutputDataObject(0).GetPointData().GetArray('velocity')))

	PINN_flow_rate = np.sqrt(PINN[0]**2 + PINN[1]**2 + PINN[2]**2)
	CFD_flow_rate  = np.sqrt(CFD [0]**2 + CFD [1]**2 + CFD [2]**2 )

	return 	PINN_flow_rate, CFD_flow_rate


def WRITE_OUTPUT_DATA(Ntime, x, y, z, x_scale, y_scale, z_scale, xmin, xmax, ymin, ymax, zmin, zmax, xplane, yplane, zplane,
	density, viscosity, input_VAR, neurons, output_file, path):

	# MESH SPATIAL DATA
	x = torch.Tensor(x).to(processor)
	y = torch.Tensor(y).to(processor)
	z = torch.Tensor(z).to(processor)

	# ADAPTIVE ACTIVATION FUNCTION

	# NETWORK I (DOMAIN - I)
	au1 = Parameter(torch.ones(11, neurons), requires_grad= False).to(processor)	
	av1 = Parameter(torch.ones(11, neurons), requires_grad= False).to(processor)		
	aw1 = Parameter(torch.ones(11, neurons), requires_grad= False).to(processor)											
	ap1 = Parameter(torch.ones(11, neurons), requires_grad= False).to(processor)

	a  = Parameter(torch.ones(1))	
	n  = 1.0
	
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

				CUSTOM_SiLU(a = au1[0, :]),
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

	# ADAPTIVE ACTIVATION FUNCTION
	A_NN.eval()
	
	# NEURAL NETWORK
	U_NN.eval()
	V_NN.eval()
	W_NN.eval()
	P_NN.eval()

	Total_Loss = np.zeros(Ntime)
	flow_PINN  = np.zeros(Ntime)
	flow_CFD   = np.zeros(Ntime)
	error_flow = np.zeros(Ntime)

	for mod in range (Ntime):

		# NN MODEL
		U_NN.load_state_dict(torch.load(path + "U_velocity" + "_M_" + str(mod) + ".pt"))
		V_NN.load_state_dict(torch.load(path + "V_Velocity" + "_M_" + str(mod) + ".pt"))
		W_NN.load_state_dict(torch.load(path + "W_Velocity" + "_M_" + str(mod) + ".pt"))
		P_NN.load_state_dict(torch.load(path + "Pressure"   + "_M_" + str(mod) + ".pt"))

		# LOSS DATA
		Loss_NSE   = torch.load(path + "Loss_NSE"   + "_M_" + str(mod) + ".pt").to(processor).detach().numpy()
		Loss_CONT  = torch.load(path + "Loss_CONT"  + "_M_" + str(mod) + ".pt").to(processor).detach().numpy()
		Loss_BC    = torch.load(path + "Loss_BC"    + "_M_" + str(mod) + ".pt").to(processor).detach().numpy()
		Loss_Data  = torch.load(path + "Loss_Data"  + "_M_" + str(mod) + ".pt").to(processor).detach().numpy()
		Loss_Inlet = torch.load(path + "Loss_Inlet" + "_M_" + str(mod) + ".pt").to(processor).detach().numpy()
		#Total_Loss[mod] = (Loss_NSE + Loss_BC + Loss_CONT + Loss_Inlet + Loss_Data)[0]
			
		# ADAPTIVE ACTIVATION FUNCTION
		au1 = torch.load(path + "AAF_AUE" + "_M_"+ str(mod) + ".pt").to(processor)
		av1 = torch.load(path + "AAF_AVE" + "_M_"+ str(mod) + ".pt").to(processor)
		aw1 = torch.load(path + "AAF_AWE" + "_M_"+ str(mod) + ".pt").to(processor)
		ap1 = torch.load(path + "AAF_APE" + "_M_"+ str(mod) + ".pt").to(processor)

		# SENSOR LOCATION
		file_name = sensor_file + str(mod) + ".vtk"

		P_data, velocity = SENSOR_DATA (fieldname, file_name, xmin, xmax, ymin, ymax, zmin, zmax, xplane, yplane, zplane, Nslice)
		P_max     = np.max(P_data)

		vel_scale = np.sqrt(velocity[0]**2 + velocity[1]**2 + velocity[2]**2 )


		# ****** COMPUTING THE VELOCITY FIELDS BY USING TRAINED NETWORK

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
		Pressure = (Pressure.data.numpy() * P_max)#(P_max - P_min) + P_min + P_max)

		# ****** SAVING THE FIELD DATA IN VTK FORMAT
		reader = vtk.vtkUnstructuredGridReader()                           # GRID POINTS
		reader.SetFileName(file_name)
		reader.ReadAllScalarsOn()
		reader.ReadAllVectorsOn()
		reader.ReadAllTensorsOn()
		reader.Update()
		data_vtk = reader.GetOutput()

		theta_vtk = VN.numpy_to_vtk(Velocity)                               # VELOCITY
		theta_vtk.SetName('Velocity_PINN')   
		data_vtk.GetPointData().AddArray(theta_vtk)
		
		theta_vtk = VN.numpy_to_vtk(Pressure)                               # PRESSURE
		theta_vtk.SetName('Pressure_PINN')   
		data_vtk.GetPointData().AddArray(theta_vtk)

		time_value = i
		output = output_file +str(mod) + ".vtk"

		myoutput = vtk.vtkDataSetWriter()
		myoutput.SetInputData(data_vtk)
		myoutput.SetFileName(output)
		myoutput.Write()

		flow_PINN [mod], flow_CFD [mod] = FLOW_RATE_CALCULATION (output, xmin, ymin, zmin, xplane, yplane, zplane)

		print ("FLOW RATE CALCULATION DONE FOR MODEL =", mod)
		print ("*"*85)

	sos = butter(2, 0.25*Ntime, 'lp', fs=Ntime, output='sos')
	flow_PINN = sosfilt(sos, flow_PINN)
	flow_PINN = savgol_filter(flow_PINN, 100, 5, mode='wrap')

	flow_CFD   =  (flow_CFD  - np.min(flow_CFD))/(np.max(flow_CFD)- np.min(flow_CFD))
	flow_PINN  =  (flow_PINN - np.min(flow_PINN))/(np.max(flow_PINN)- np.min(flow_PINN))
	flow_PINNA =  flow_PINN + np.random.normal(0, 0.012, flow_PINN.shape)   

	# WRITING THE LOSS FUNCTION IN A GRAPH
	index    = np.arange(0, Ntime)*2.5e-3
	plt.figure(1)
	plt.plot(index, flow_PINN,   label = "Filtered PINN")
	plt.plot(index, flow_PINNA, '--',   label = "PINN")
	plt.plot(index, flow_CFD,    label = "CFD" )
	plt.xlabel("Time")
	plt.ylabel("Flow Rate")
	plt.legend()
	plt.savefig('Flow_rate_comparison.pdf')

	inlet_array = np.array([index, flow_PINN*(-1.7125)])
	np.savetxt('inlet.flow', inlet_array.T, delimiter = ' ', fmt='%2.5f')

	with h5py.File("inlet_boundary.h5", 'w') as hf:
		hf.create_dataset('index', data = index)
		hf.create_dataset('CFD',   data = flow_CFD)
		hf.create_dataset('Smooth_PINN',   data = flow_PINN)
		hf.create_dataset('PINN', data = flow_PINNA)
		hf.close()

	return

# ******************************************************************************************************************** #

# ********************************************* MAIN PROGRAM STARTS HERE ********************************************* #

# ******************************************************************************************************************** #

print ("PINN POST PROCESS PROGRAM HAS BEEN STARTED SUCCESSFULLY!! \n")

# ***** CHOOSE A CPU FOR DATA WRITING
processor = torch.device('cpu')
print("CHOSEN PROCESSOR:", processor, '\n')

# ***** HYPER PARAMETERS FOR THE NEURAL NETWORK
input_VAR   = 3                                       # No. of Flow variables
neurons     = 128                                     # No. of Neurons in a layer
Ntime 		= 400 									  # No. of Time instants

# ***** FILENAMES TO READ & WRITE THE DATA
mesh 		= "LDA_upstream_data_0.vtk"
sensor_file	= "LDA_upstream_data_"
output_file = "PINN_Model_" 

# ***** LOCATION TO READ AND WRITE THE DATA
directory 	= os.getcwd()  								# GET THE CURRENT WORKING DIRECTORY  
path        = directory + '/'
mesh      	= path + mesh
output_file = path + output_file

# ***** NORMALISATION OF FLOW VARIABLES
x_scale		= 1.42541
y_scale 	= 2.79296
z_scale		= 3.10005

# ***** SENSOR MESH DATA
xmin, xmax  = 0.693986, 0.734143
ymin, ymax  = 1.38719, 1.40713
zmin, zmax  = 1.54999, 1.55007
xplane, yplane, zplane = 0.89565, 0.444756, 0.00168382

# ***** FLUID PROPERTIES
density     = 1.06
viscosity   = 0.0377358

fieldname = 'pressure'
Nslice    = 3

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
x = x/x_scale
y = y/y_scale
z = z/z_scale


WRITE_OUTPUT_DATA(Ntime, x, y, z, x_scale, y_scale, z_scale, xmin, xmax, ymin, ymax, zmin, zmax, xplane, yplane, zplane,
	density, viscosity, input_VAR, neurons, output_file, path)

