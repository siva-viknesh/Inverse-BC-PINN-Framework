## INVERSE BC-PINN ##

Utilizing Physics-informed neural networks (PINN) to compute 3D blood flow velocity fields from sparse iFR/FFR pressure data sampled across stenosed coronary arteries. 
<hr>
**Machine Learning Enhanced Hemodynamics: Constructing 3D Blood Flow Fields of Stenosed Coronary Arteries from Pressure Measurements, Siva Viknesh, Ethan Shoemaker, and Amirhossein Arzani**
<hr>
Pytorch codes are included (along with their geometries) for the different examples presented in the paper: <br />

- Idealized Stenosed Coronary Arteries
  *   Symmetric Stenosis
  *   Asymmetric Stenosis
- Patient-specific LAD Stenosed Coronary Artery
  * Steady Flow 
  * Transient Flow
<hr>
Converting the results to VTK: The torch-to-vtk conversion Python programs can be found in PINN- Post Processing folder for both PINN and BC-PINN methodologies.

<hr>
Installation:
Install Pytorch:
https://pytorch.org/
Install VTK after Pytorch is installed.

An example with pip: <br />
conda activate pytorch <br />
pip install vtk<br />

![Model](https://github.com/siva-viknesh/Inverse-BC-PINN-Framework/blob/main/Patient-Specific%20LAD%20Coronary%20Artery/Figure.jpeg)
