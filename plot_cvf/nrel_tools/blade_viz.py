
# system-level libraries
import sys
import os.path

# IO libraries
import copy
import yaml

# plotting
import numpy as np
import scipy.interpolate as interpol
import matplotlib.pyplot as plt
import seaborn as sns
import plot_cvf
import pprint as pp


def check_input(input):

  # input checking
  assert len(input) > 1, "you need to specify at least one file"
  
  filenames= []
  if len(input) > 2:
    for input_slice in range(len(input) - 1):
      filename= check_input([input[0], input[input_slice+1]])
      filenames += filename
  else:
    filename= input[1]
    assert os.path.isfile(filename), "this code only works on yaml files"
    fname_base, extension= os.path.splitext(filename)
    assert extension in [".yaml", ".yml"], "this code only works on yaml files"
    filenames.append(filename)

  return filenames

def extract_blade_vectors(yaml_data):

  # get the data we will want to perturb
  airfoil_position_data= yaml_data['components']['blade']['outer_shape_bem']['airfoil_position']
  chord_data= yaml_data['components']['blade']['outer_shape_bem']['chord']
  twist_data= yaml_data['components']['blade']['outer_shape_bem']['twist']
  pitch_axis_data= yaml_data['components']['blade']['outer_shape_bem']['pitch_axis']
  reference_axis_data= yaml_data['components']['blade']['outer_shape_bem']['reference_axis']
  
  return airfoil_position_data, chord_data, twist_data, pitch_axis_data, reference_axis_data

def extract_airfoil(yaml_data, airfoil_label):

  airfoil_dict= {}

  # make sure the airfoil we want is there, get its index
  assert airfoil_label in [af['name'] for af in yaml_data['airfoils']]
  idx_af= [af['name'] for af in yaml_data['airfoils']].index(airfoil_label)

  # extract the yamldata
  airfoil_dict[airfoil_label]= yaml_data['airfoils'][idx_af]

  return airfoil_dict

def get_splined_section(yaml_data, zq):
  # get the data
  _, chord_data, twist_data, pitch_axis_data, reference_axis_data= \
      extract_blade_vectors(yaml_data)
  
  # chord first, interpolate at a given query non-dim. span
  zp_chord= chord_data['grid']
  cp_chord= chord_data['values']
  cq= interpol.PchipInterpolator(zp_chord, cp_chord)(zq)

  zp_twist= twist_data['grid']
  twp_twist= twist_data['values']
  twq= interpol.PchipInterpolator(zp_twist, twp_twist)(zq)

  zp_pitchax= pitch_axis_data['grid']
  pap_pitchax= pitch_axis_data['values']
  paq= interpol.PchipInterpolator(zp_pitchax, pap_pitchax)(zq)

  ndsxp_refax= reference_axis_data['x']['grid']
  xp_refax= reference_axis_data['x']['values']
  xq_refax= interpol.PchipInterpolator(ndsxp_refax, xp_refax)(zq)
  ndsyp_refax= reference_axis_data['y']['grid']
  yp_refax= reference_axis_data['y']['values']
  yq_refax= interpol.PchipInterpolator(ndsyp_refax, yp_refax)(zq)
  ndszp_refax= reference_axis_data['z']['grid']
  zp_refax= reference_axis_data['z']['values']
  zq_refax= interpol.PchipInterpolator(ndszp_refax, zp_refax)(zq)

  return cq, twq, paq, (xq_refax, yq_refax, zq_refax)

def create_plot_splined(data_dict_list,
                        xlabel= None, ylabel= None):
  """
  create a plot of a quantity that should be interpreted via a PCHIP lofting
  spline
  """

  if type(data_dict_list) == dict:
    data_dict_list= [data_dict_list,]
  Nplot= 201
  fig, ax= plt.subplots()
  for data_dict in data_dict_list:
    xp= data_dict['x']
    yp= data_dict['y']
    x= np.linspace(np.min(xp), np.max(xp), Nplot)
    y= interpol.PchipInterpolator(xp, yp)(x)
    ax.plot(x, y, '.-', label= data_dict['label'])
  ax.set_xlabel(xlabel)
  ax.set_ylabel(ylabel)
  ax.grid(False)
  fig.legend()

  return fig, ax

def loft_foils(yaml_data, zq, Ninterp= 101):

  # get the spanwise data
  airfoil_section_data, chord_data, twist_data, pitch_axis_data, reference_axis_data= \
    extract_blade_vectors(yaml_data)
  
  z_list= airfoil_section_data['grid']
  foilname_list= airfoil_section_data['labels']

  # airfoil 
  af_interp_dict= {}
  for z_p, foilname_p in zip(z_list, foilname_list):
    af_here= extract_airfoil(yaml_data, foilname_p)[foilname_p]

    # get the airfoil coords
    x_coord= af_here['coordinates']['x']
    y_coord= af_here['coordinates']['y']

    # interpolate the airfoil spec onto a common number of points, assumed to be
    # equidistant in either angular or arc space
    basis_ref= np.linspace(0.0, 1.0, Ninterp)
    basis_coord= np.linspace(0.0, 1.0, len(x_coord))
    x_interp= np.zeros((Ninterp,))
    y_interp= np.zeros((Ninterp,))
    for i in range(Ninterp):
      x_interp[i]= interpol.PchipInterpolator(basis_coord, x_coord)(basis_ref[i])
      y_interp[i]= interpol.PchipInterpolator(basis_coord, y_coord)(basis_ref[i])

    af_interp_dict[z_p]= {
      'x': x_interp,
      'y': y_interp,
    }
  
  # now "interpolate" foil by interpolating x & y vars at a given index along span
  x_blended= np.zeros((Ninterp,))
  y_blended= np.zeros((Ninterp,))
  for i in range(Ninterp):
    # along a stringer
    x_stringer= np.array([af_interp_dict[key]['x'][i] for key in sorted(af_interp_dict.keys())])
    y_stringer= np.array([af_interp_dict[key]['y'][i] for key in sorted(af_interp_dict.keys())])
    z_stringer= np.array(sorted(af_interp_dict.keys()))
    x_blended[i]= interpol.PchipInterpolator(z_stringer, x_stringer)(zq)
    y_blended[i]= interpol.PchipInterpolator(z_stringer, y_stringer)(zq)

  return x_blended, y_blended

def main():

  # load the stylesheet for good plots
  plt.style.use(plot_cvf.get_stylesheets(dark= True))

  # get the filename after checking input for obvious issues
  filenames= check_input(sys.argv)

  chords_to_plot= []
  twists_to_plot= []

  airfoil_dict= {}

  for filename in filenames:

    # try to safe read
    with open(filename, 'r') as infile:
      data= yaml.safe_load(infile)

    display_name= os.path.split(os.path.splitext(filename)[0])[-1]

    # get the blade profile data
    airfoil_position_data, chord_data, twist_data, pitch_axis_data, reference_axis_data= extract_blade_vectors(data)

    fig= plt.figure()
    ax= fig.add_subplot(projection= '3d')    # get the airfoils 

    # for airfoil_location, airfoil_label in zip(*airfoil_position_data.values()):
    #   print("%6f" % airfoil_location, airfoil_label)
    for airfoil_location in np.linspace(0.0, 1.0, 31):
      
      # raw airfoil coordinate
      x_airfoil, y_airfoil= loft_foils(data, airfoil_location)
      # # raw airfoil coordinate
      # x_airfoil= copy.deepcopy(airfoil_dict[airfoil_label]['coordinates']['x'])
      # y_airfoil= copy.deepcopy(airfoil_dict[airfoil_label]['coordinates']['y'])

      # get splined data
      chord_here, twist_here, pitch_axis_here, (x_refax, y_refax, z_refax)= \
          get_splined_section(data, airfoil_location)

      # transform, translating pitch axis first
      x_airfoil -= pitch_axis_here
      # then multiplying by chord everywhere
      x_airfoil *= chord_here
      y_airfoil *= chord_here
      # now rotate by twist
      x_airfoil_old= copy.deepcopy(x_airfoil)
      y_airfoil_old= copy.deepcopy(y_airfoil)
      x_airfoil= x_airfoil_old*np.cos(twist_here) + y_airfoil_old*np.sin(twist_here)
      y_airfoil= y_airfoil_old*np.cos(twist_here) - x_airfoil_old*np.sin(twist_here)
      # now move to the reference axis
      x_airfoil += x_refax
      y_airfoil += y_refax

      plt.plot(x_airfoil, y_airfoil, np.ones_like(x_airfoil)*z_refax)
      
    ax.axis('square')
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.zaxis.set_visible(False)
    ax.grid(False)
    plt.show()

    chords_to_plot.append({'x': chord_data['grid'], 'y': chord_data['values'], 'label': display_name})
    twists_to_plot.append({'x': twist_data['grid'], 'y': twist_data['values'], 'label': display_name})

  # # plot each airfoil at reference scale
  # fig, ax= plt.subplots()
  # for airfoil_label, airfoil_data in airfoil_dict.items():
  #   print(airfoil_label)
  #   print(airfoil_data)
  #   ax.plot(airfoil_data['coordinates']['x'], airfoil_data['coordinates']['y'], '.-',
  #           label= airfoil_label)
  # ax.axis('square')
  # fig.legend()
  # plt.show()

  fig, ax= plt.subplots()
  

  # # make a plot
  # fig, ax= create_plot_splined(chords_to_plot,
  #                              xlabel= "non-dimensional span", ylabel= "chord")
  # fig, ax= create_plot_splined(twists_to_plot,
  #                              xlabel= "non-dimenstional span", ylabel= "twist")
  # plt.show()

if __name__ == '__main__':
  main()
