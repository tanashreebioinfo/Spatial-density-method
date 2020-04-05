#from __future__ import print_function
import mdtraj as md
import sys, math, random, copy, pickle, re
import numpy as np
import numpy.linalg
import matplotlib.pyplot as plt
from scipy import stats
from mayavi import mlab
# Changing the ctf:
from tvtk.util.ctf import ColorTransferFunction
# Changing the otf:
from mpl_toolkits.mplot3d import Axes3D
#using a colourmap
from sklearn.neighbors import KernelDensity
from matplotlib import cm 
import joblib
"""
units of MDTraj
distance - nanometers
"""
class atoms():
	def __init__(self, index, element):
		self.index = index
		self.element = element	

	def add_position_and_colour(self, x, y, z, rgb, radius):
		self.posx = x
		self.posy = y
		self.posz = z
		self.color = rgb
		self.radius = radius
		return

def centre_of_mass(molecule):
#molecule is a list of atoms object
	sumx = 0; sumy = 0; sumz = 0
	
	for i in molecule:
		sumx += i.posx
		sumy += i.posy
		sumz += i.posz

	centrex = sumx/float(len(molecule))
	centrey = sumy/float(len(molecule))
	centrez = sumz/float(len(molecule))

	return (centrex, centrey, centrez)

def get_bonds(psffile):
		f = open(psffile, "r").read()
		strings = re.findall("bonds.*angles", f, re.DOTALL)
		lines = strings[0].split("\n")
		bonds = []
		for l in lines[1:-2]:
			bonds = bonds + l.split()
		return bonds

def plot_plane(normal):
	# create x,y
	xx, yy = np.meshgrid(np.linspace(0,100), np.linspace(0,100))
	# calculate corresponding z
	z = (normal[0] * xx + normal[1] * yy + normal[2])
	# plot the surface
	plt3d = plt.figure().gca(projection='3d')
	plt3d.plot_surface(xx, yy, z, color="red")
	plt.show()

def solve_plane(x, y, z):
	#least squares fit for z = ax + by + c
	X = np.array(x)
	Y = np.array(y)
	Z = np.array(z)
	coeff = np.array([[sum(X*X), sum(Y*X), sum(X)],
			 [sum(X*Y), sum(Y*Y), sum(Y)],
		   	 [sum(X), sum(Y), X.size]])
	ordinates = np.array([sum(X*Z), sum(Y*Z), sum(Z)])
	try:
		result = np.linalg.solve(coeff, ordinates)
	except numpy.linalg.linalg.LinAlgError:
		result = np.linalg.lstsq(coeff, ordinates)[0]	
	return result

def best_fit_plane(model_atom_positions):
	coefficients = solve_plane(model_atom_positions[:,0], model_atom_positions[:,1], model_atom_positions[:,2])
	#print coefficients,

	#calculating angles with x,y and z-axis
	normal_vector = np.array([coefficients[0], coefficients[1], -1.0])
	normal_vector = normal_vector/np.linalg.norm(normal_vector)

	angx = math.acos(normal_vector[0])*180.0/math.pi
	angy = math.acos(normal_vector[1])*180.0/math.pi
	angz = math.acos(normal_vector[2])*180.0/math.pi
	#print angx, angy, angz

def get_residues(topology, sel):
	current_res = topology.atom(sel[0]).residue
	residues = []
	temp = [sel[0]]
	for i in sel[1:]:
		if topology.atom(i).residue == current_res:
			temp.append(i)	
		else:
			current_res = topology.atom(i).residue
			residues.append(copy.deepcopy(temp))
			temp = [i]
	
	residues.append(copy.deepcopy(temp))
	return np.array(residues)

def get_minimum_distance_between_selections(traj, topology, sel1, sel2):
	pairs = []
	for i in sel1:
		for j in sel2:
			pairs.append([i,j])	
	atom_pairs = np.array(pairs)	
	distances = md.compute_distances(traj, atom_pairs, periodic=True, opt=True)
	return np.amin(distances)

def get_selections_within_cutoff(traj, topology,sel1, sel2, cutoff = 0.45):
	"""
	This function returns contact_residues which is a np array of dimensions [residue num][atom index], 
	along with a np array of its minimum distance
	"""
	contact_residues = []
	contact_distance = []
	for i in sel1: #selection one is the central solute molecule here
		for j in sel2:
			min_dist = get_minimum_distance_between_selections(traj, topology, i, j)
			#print min_dist
			if (min_dist <= cutoff):
				contact_residues.append(j)
				contact_distance.append(min_dist)
				#print contact_distance
	return np.array(contact_residues), np.array(contact_distance)	

def cartesian2polar(X,Y,Z):
	rho = np.sqrt(X*X + Y*Y + Z*Z)
	theta = np.arctan2(Y,X)*(180.0/math.pi)
	phi = np.arctan2(np.sqrt(Y*Y + X*X),Z)*(180.0/math.pi)
	return [rho, theta, phi]

def read_dcd_for_urea_distribution(dcd, pdb, model_residue_name, coordinates="cart"):
	traj = md.load(dcd, top=pdb)
	#print traj
	#topology = pdb.topology
	frames = []
	for i in range(len(traj)):
		topology = traj[i].topology

		# for atom in topology.atoms:
		# 	print()
		#  	print(atom.__dict__)
		#  	print(atom.residue.__dict__)
		#  	print()

		model_system_atoms = [atom.index for atom in topology.atoms if (atom.residue.name == model_residue_name and atom.element.symbol != "H")]
		model_atom_positions = np.array([traj[i].xyz[0][ind] for ind in model_system_atoms]) #0 denotes first frame

		urea_atoms = [atom.index for atom in topology.atoms if (atom.residue.name == 'UREA' and atom.element.symbol != "H")]
		#print "urea_atoms = ", urea_atoms

		model_residues = get_residues(topology, model_system_atoms)
		urea_residues = get_residues(topology, urea_atoms)


		residues, distance = get_selections_within_cutoff(traj[i],topology, model_residues,urea_residues)
		#print residues
		#convert all coordinates to polar for the spatial probability distribution
		cutoff_urea_residues = []
		for res in residues:
			temp = {}
			for atom_ind in res:
				if coordinates == "polar":
					temp[topology.atom(atom_ind).name] = (cartesian2polar(traj[i].xyz[0][topology.atom(atom_ind).index][0], traj[i].xyz[0][topology.atom(atom_ind).index][1], traj[i].xyz[0][topology.atom(atom_ind).index][2]))
				else:
					temp[topology.atom(atom_ind).name] = (traj[i].xyz[0][topology.atom(atom_ind).index][0], traj[i].xyz[0][topology.atom(atom_ind).index][1], traj[i].xyz[0][topology.atom(atom_ind).index][2])
				
			cutoff_urea_residues.append(temp)
		#print(cutoff_urea_residues)
		
		#np array with dictionary having [residue num][{"atom":"polar coor"} pairs]					
		cutoff_urea_residues = np.array(cutoff_urea_residues)
		#print(cutoff_urea_residues)
		frames.append(cutoff_urea_residues)	
		
	frames = np.array(frames)
	#print(frames)
  	return frames	  

def read_dcd_for_water_distribution(dcd, pdb, model_residue_name, coordinates="cart"):
	traj = md.load(dcd, top=pdb)
	frames = []
	for i in range(len(traj)):
		topology = traj[i].topology
		model_system_atoms = [atom.index for atom in topology.atoms if (atom.residue.name == model_residue_name and atom.element.symbol != "H")]
		model_atom_positions = np.array([traj[i].xyz[0][ind] for ind in model_system_atoms]) #0 denotes first frame

		water_atoms = [atom.index for atom in topology.atoms if (atom.residue.name == 'HOH' and atom.element.symbol != "H")]

		model_residues = get_residues(topology, model_system_atoms)
		water_residues = get_residues(topology, water_atoms)

		residues, distance = get_selections_within_cutoff(traj[i],topology, model_residues,water_residues)

		#convert all coordinates to polar for the spatial probability distribution
		cutoff_water_residues = []
		for res in residues:
			temp = {}
			for atom_ind in res:
				if coordinates == "polar":
					temp[topology.atom(atom_ind).name] = (cartesian2polar(traj[i].xyz[0][topology.atom(atom_ind).index][0], traj[i].xyz[0][topology.atom(atom_ind).index][1], traj[i].xyz[0][topology.atom(atom_ind).index][2]))
				else:
					temp[topology.atom(atom_ind).name] = (traj[i].xyz[0][topology.atom(atom_ind).index][0], traj[i].xyz[0][topology.atom(atom_ind).index][1], traj[i].xyz[0][topology.atom(atom_ind).index][2])
				
			cutoff_water_residues.append(temp)
		
		#np array with dictionary having [residue num][{"atom":"polar coor"} pairs]					
		cutoff_water_residues = np.array(cutoff_water_residues)
		frames.append(cutoff_water_residues)	
		
	frames = np.array(frames)
  	return frames

def read_dcd_for_model_system(dcd, pdb, model_residue_name, psffile):
	#this psf file is in CHARMM format
	traj = md.load(pdb, top=pdb)
	topology = traj[0].topology
	model_system_atoms = [atom for atom in topology.atoms if (atom.residue.name == model_residue_name)]

	model_atom_positions = np.array([traj[0].xyz[0][atom.index] for atom in model_system_atoms]) #0 denotes first frame

	molecule = [atoms(atom.index, atom.element) for atom in model_system_atoms]


	#dictionary mapping for element and colour
	colors = {'carbon':(0.3,0.3,0.3), 'hydrogen':(1.0,1.0,1.0), 'nitrogen':(0.2,0.2,1.0), 'oxygen':(1.0,0.2,0.2), 'sulfur':(1.0,1.0,0.2)}

	#get bond connectivity information
	table, bonds = topology.to_dataframe()
	#print table
	#filling position and colour values for molecules
	index = 0
	for atom in molecule:
		atom.add_position_and_colour(model_atom_positions[index][0],model_atom_positions[index][1],model_atom_positions[index][2], rgb=colors[atom.element.name], radius=atom.element.radius)
		index += 1

	for atom in molecule:
		mlab.points3d(atom.posx, atom.posy, atom.posz, color=atom.color, scale_factor=0.3*atom.radius, scale_mode="none")

	#getting connectivity from psf file
	bonds = get_bonds(psffile)
	#converting bonds list from str to int, we subtract 1 to make it 1-indexed
	bonds = [int(i) - 1 for i in bonds]

	#plotting the bonds
	i = 0
	while(i < len(bonds) - 1):
		mlab.plot3d([molecule[bonds[i]].posx, molecule[bonds[i+1]].posx], [molecule[bonds[i]].posy, molecule[bonds[i+1]].posy], [molecule[bonds[i]].posz, molecule[bonds[i+1]].posz], color=(1.0,1.0,1.0), tube_radius=0.01)
		i += 2
	return molecule

def write_residues(frames, filename):
	f = open(filename, "w")
	serialized = pickle.dump(frames, f)
	return

def load_residues(filename):
	f = open(filename, "r")
	deserialized_a = pickle.load(f)
	return deserialized_a

def scale_value(val, minx, maxx):
	return (val - minx)/(1.0*(maxx - minx))

def check_if_prob_sum_to_one(scores, x_len, y_len, z_len):
	#print(np.sum(np.exp(scores)), x_len, y_len, z_len)
	sum_prob = np.sum(np.exp(scores))*x_len*y_len*z_len
	#print("sum_prob = ", sum_prob)
	np.testing.assert_almost_equal(sum_prob, 1.0, decimal=2)
	return

def get_spatial_probability(frames, symbol, colormap_arg, style="isosurface", scale="linear",c_arg='YlGnBu', granularity=70, compute_scores=True, urea_pkl_name=None, water_pkl_name=None):
	#print ("frames shape = ", frames.shape)
	#print(frames[0])
	#exit(0)
	distribution = []
	distribution_all = []
	no_of_atoms_of_each_type = {}

	if symbol == "N*":
		for f in frames:
			for residues in f:
				#print residues
				if "N1" not in residues:
					continue
				distribution.append(residues["N1"])

		for f in frames:
			for residues in f:
				#print residues
				if "N3" not in residues:
					continue
				distribution.append(residues["N3"])
	
	else:
		for f in frames:
			for residues in f:
				if symbol not in residues:
					continue
				distribution.append(residues[symbol])


	symbol_list=["N1", "N3", "C2", "O2", "O"]
	for single_symbol in symbol_list:
		no_of_atoms_of_each_type[single_symbol] = 0.0
	pkl_file_names = [urea_pkl_name, water_pkl_name]
	for pkl_file_name in pkl_file_names:
		frames_all = load_residues(pkl_file_name)
		distribution_all = []
		for f in frames_all:
			for residues in f:
				if not (any(i in symbol_list for i in residues)):
				#if symbol not in residues:
					continue
				for single_symbol in symbol_list:
					#if single_symbol[0] == "N":
					if single_symbol in residues:
						#print("adding ", single_symbol)
						#distribution_all.append(residues[single_symbol])
						no_of_atoms_of_each_type[single_symbol] += 1.0
					#else:
						#print("adding "+single_symbol)
					#	distribution_all.append(residues[single_symbol])
		distribution_all = np.array(distribution_all)
	#print("Atom frequency dict = ", no_of_atoms_of_each_type)
	if symbol in ["N*", "C2", "O2"]:
		no_of_atoms_of_each_type["N*"] = no_of_atoms_of_each_type["N1"] + no_of_atoms_of_each_type["N3"]

	distribution = np.array(distribution)


	#exit(0)

	#calculate probability density
	binx = 50
	biny = 50
	binz = 50
	Hist, edges = np.histogramdd(distribution,bins=(binx, biny, binz),normed=False)
	print("edges = ", edges)
	
	#Hist += 1
	#Hist /= frames.shape[0]
	#print Hist.shape, "Hist_shape"
	
	model = KernelDensity(bandwidth=0.1)
	model.fit(distribution)
	
	
	indices = np.unravel_index(Hist.argmax(), Hist.shape)
	#print np.mean(edges[0]), np.mean(edges[1]), np.mean(edges[2])
	#make a d-dimensional mesh for the 3-D contour plot
	x, y, z = np.mgrid[edges[0][0]:edges[0][-1]:complex(0,binx),edges[1][0]:edges[1][-1]:complex(0,biny), edges[2][0]:edges[2][-1]:complex(0,binz)]
	#print x.shape, y.shape, z.shape
	"""
	x = 50 2-d arrays. shape = 50x50x50. Each 2-d array filled with same number.
	y = 50 2-d arrays. shape = 50x50x50. Each row of each 2-d array is filled with same number. all 2-d arrays are same.
	z = 50 2-d arrays. shape = 50x50x50. All 2-d arrays are same. All rows of each 2-d array are same. but Each row is filled with different number.
	"""


	
	if compute_scores:
		scores = model.score_samples(np.vstack((x.flatten(), y.flatten(),z.flatten())).T).reshape(Hist.shape)
		#joblib.dump(scores, 'scores_'+sys.argv[1]+'_'+symbol+'.pkl')
		joblib.dump(scores, 'scores_'+symbol+'.pkl')
		return
	
	else:
		#scores = joblib.load('scores_'+sys.argv[1]+'_'+symbol+'.pkl')
		scores = joblib.load('scores_'+symbol+'.pkl')
		#print("scores shape = ", scores.shape)

		# scores_all = []
		# sum_scores_all = None
		# for single_symbol in ["N*", "C2", "O2"]:
		# 	tmp = joblib.load('scores_'+single_symbol+'.pkl')
		# 	scores_all.append(copy.deepcopy(tmp))
		# 	if sum_scores_all is None:
		# 		sum_scores_all = tmp*tmp
		# 	else:
		# 		sum_scores_all = sum_scores_all + tmp*tmp
		# sum_scores_all = np.sqrt(sum_scores_all)
		# scores = scores / sum_scores_all
		# #scores *= 1000

		# global_min = 10000000000000000000000
		# global_max = -10000000000000000000000
		# for single_score in scores_all:
		# 	single_score = single_score / sum_scores_all
		# 	#single_score *= 1000
		# 	global_min = min(np.amin(single_score), global_min)
		# 	global_max = max(np.amax(single_score), global_max)
	scores = scores*(no_of_atoms_of_each_type[symbol]) / sum(no_of_atoms_of_each_type.values())
	#print("scores=",scores)
	
	#check_if_prob_sum_to_one(scores, x[1][0][0]-x[0][0][0], y[0][1][0]-y[0][0][0], z[0][0][1]-z[0][0][0])
	
	# for i in range(binx):
	# 	for j in range(biny):
	# 		for k in range(binz):
	# 			score = np.exp(scores[i][j][k])
	# 			if (score > 0.05 and i==17):
	# 				print(i, j, k, score)


	if(style == "isosurface"):	

		# x = [i[0] for i in distribution]
		# y = [i[1] for i in distribution]
		# z = [i[2] for i in distribution]
		# print(len(x))
		# pts = mlab.points3d(x, y, z, [1 for i in range(len(x))], colormap="copper", scale_factor=.001)

		pts = mlab.contour3d(x, y, z, scores, transparent=False, color=colormap_arg, contours=3)
		#pts = mlab.contour3d(x, y, z, Hist, transparent=True, color=colormap_arg, contours=3)
	
	else:
		if scale == "linear":
			source = mlab.pipeline.scalar_field(x,y,z,scores)
			probabilities = np.unique(Hist)
			
			contours = np.log(np.linspace(np.exp(np.amin(scores)), np.exp(np.amax(scores)), 100))
			print("contours = ", contours)
			contours = contours[::-1][:granularity]
			print("contours 2 = ", contours)
			probabilities = list(contours[::-1])
			print("probabilities = ", probabilities)

			#print probabilities
			for prob in range(len(probabilities)):
			#for prob in range(5):
				value = probabilities[prob]
				
			        ctr = mlab.pipeline.iso_surface(source, contours=[probabilities[prob], ], vmin=probabilities[0], vmax=probabilities[-1], colormap=c_arg, opacity=0.3)
				#ctr = mlab.pipeline.iso_surface(source, contours=[probabilities[prob], ], vmin=0.197, vmax=0.375, colormap=c_arg, opacity=0.3) ##
				
				#ctr = mlab.pipeline.iso_surface(source, contours=[probabilities[prob], ], vmin=global_min, vmax=global_max, colormap=c_arg, opacity=0.3)
		
		elif scale == "log":
			#Hist_temp = Hist[(Hist > 0)]
			Hist[(Hist == 0)] = 1e-10
			Hist = np.log10(Hist)
		
			source = mlab.pipeline.scalar_field(x,y,z,Hist)
			probabilities = np.unique(Hist)
		
			range_prob[(range_prob == 0)] = 1e-5
			probabilities = np.log10(range_prob)
			
			for prob in range(len(list(probabilities))):
				value = probabilities[prob]
				ctr = mlab.pipeline.iso_surface(source, contours=[probabilities[prob], ], vmin=probabilities[0], vmax=probabilities[-1], opacity=0.2*scale_value(value, probabilities[0], probabilities[-1]))
	
	if scale == "linear":	
		mlab.colorbar(object=ctr, title='Log Probability Density '+str(symbol), orientation='vertical')

	if scale == "log":	
		mlab.colorbar(object=ctr, title='Log Probability', orientation='vertical')

def main():
	dcd = sys.argv[1] #the trajectory
	top = sys.argv[2] #PDB file
	model_residue_name = sys.argv[3] #resname of model system in interest
	print sys.argv

	water_pkl_name = "water_residues.pkl"
	urea_pkl_name = "urea_residues.pkl"

	if sys.argv[4] == "process_dcd":
		frames = read_dcd_for_urea_distribution(dcd, top, model_residue_name, "cart")
		frames2 = read_dcd_for_water_distribution(dcd, top, model_residue_name, "cart")
		if sys.argv[6] == "water":
			write_residues(frames2, water_pkl_name)
		elif sys.argv[6] == "urea":
			write_residues(frames, urea_pkl_name)
		
	
	else:
		print sys.argv
		if sys.argv[6] == "urea":
			frames = load_residues(urea_pkl_name)
			#print(frames)
		elif sys.argv[6] == "water":	
			frames = load_residues(water_pkl_name)

		mlab.figure(bgcolor=(1,1,1), fgcolor=(0.,0.,0.))
		psf_file = sys.argv[5] #this is CHARMM PSF file
		
		model_system = read_dcd_for_model_system(dcd, top, model_residue_name, psf_file)
		####### if isosurface viwe is expeced uncomment the following section#######	
          	#isosurface
		#get_spatial_probability(frames, "N1", (0.2,0.2,0.8), style="isosurface")
		#get_spatial_probability(frames, "C2", (0.4, 0.4, 0.4), style="isosurface")
		#get_spatial_probability(frames, "O2", (0.8,0.2,0.2), style="isosurface")
		compute_flag = False
		if sys.argv[8] == "True":
			compute_flag = True
		
		#density map
		if sys.argv[6] == "urea":
			if sys.argv[7] == "NCO":
				get_spatial_probability(frames, "N*", (0.2,0.2,0.8), style="volume", scale="linear",c_arg="summer", granularity=10, compute_scores=compute_flag, urea_pkl_name=urea_pkl_name, water_pkl_name=water_pkl_name) #granularity range from 0-100
				get_spatial_probability(frames, "C2", (0.4, 0.4, 0.4), style="volume", scale="linear",c_arg="cool", granularity=10, compute_scores=compute_flag, urea_pkl_name=urea_pkl_name, water_pkl_name=water_pkl_name)
				get_spatial_probability(frames, "O2", (0.8,0.2,0.2), style="volume", scale="linear",c_arg="Reds", granularity=10, compute_scores=compute_flag, urea_pkl_name=urea_pkl_name, water_pkl_name=water_pkl_name)
				
				# WATER
				#get_spatial_probability(load_residues(water_pkl_name), "O", (0.8,0.2,0.2), style="volume", scale="linear",c_arg="YlOrRd" ,granularity=4, compute_scores=compute_flag, urea_pkl_name=urea_pkl_name, water_pkl_name=water_pkl_name)

########## For seperate plotting of atoms ###########
##################################################### 
			if sys.argv[7] == "N":
				get_spatial_probability(frames, "N*", (0.2,0.2,0.8), style="isosurface", scale="linear",c_arg="YlGnBu", granularity=20, compute_scores=compute_flag, urea_pkl_name=urea_pkl_name, water_pkl_name=water_pkl_name) #granularity range from 0-100
			elif sys.argv[7] == "C":
				get_spatial_probability(frames, "C2", (0.4, 0.4, 0.4), style="volume", scale="linear",c_arg="binary", granularity=20, compute_scores=compute_flag, urea_pkl_name=urea_pkl_name, water_pkl_name=water_pkl_name)
			elif sys.argv[7] == "O":
				get_spatial_probability(frames, "O2", (0.8,0.2,0.2), style="volume", scale="linear",c_arg="YlOrRd", granularity=20, compute_scores=compute_flag, urea_pkl_name=urea_pkl_name, water_pkl_name=water_pkl_name)


		elif sys.argv[6] == "water":
			if sys.argv[7] == "O":
				get_spatial_probability(load_residues(water_pkl_name), "O", (0.8,0.2,0.2), style="volume", scale="linear",c_arg="YlOrRd", granularity=4, compute_scores=compute_flag, urea_pkl_name=urea_pkl_name, water_pkl_name=water_pkl_name)
		
		print "done"
		mlab.orientation_axes()
		mlab.view(focalpoint=centre_of_mass(model_system)) #this is for focussing the viewpoint of the camera to the centre of geometry of the molecule
	
		mlab.show()

if __name__ == '__main__':
 	main()
