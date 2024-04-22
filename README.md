## INVERSE BC-PINN##
Utilizing Physics-informed neural networks (PINN) to compute 3D blood flow velocity fields from pressure data sampled along the centerline of a stenosed coronary artery with iFR measurements. 
-----
Machine Learning Enhanced Hemodynamics: Constructing 3D Blood Flow Fields of Stenosed Coronary Arteries from Pressure Measurements, Siva Viknesh, Ethan Shoemaker, and Amirhossein Arzani, Journal of Biomechanical Engineering, 2024
-----



Converting the results to VTK: The torch-to-vtk conversion Python programs can be found in PINN- Post Processing folder for both PINN and BC-PINN methodologies. 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Installation:
Install Pytorch:
https://pytorch.org/

Install VTK after Pytorch is installed.
An example with pip:

conda activate pytorch
pip install vtk

![Model](https://github.com/siva-viknesh/Inverse-BC-PINN-Framework/blob/main/Patient-Specific%20LAD%20Coronary%20Artery/Figure.jpeg)
