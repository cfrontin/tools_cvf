
# system-level libraries
import sys
import os.path

# IO libraries
import copy
import yaml

# plotting
import matplotlib.pyplot as plt
import plot_cvf

def check_input(input):

  # input checking
  assert len(input) == 2, "this code is only configured for one input right now"
  filename= input[1]
  assert os.path.isfile(filename), "this code only works on yaml files"
  fname_base, extension= os.path.splitext(filename)
  assert extension in [".yaml", ".yml"], "this code only works on yaml files"

  return filename

def extract_blade_vectors(yaml_data):

  # get the data we will want to perturb
  airfoil_position_data= yaml_data['components']['blade']['outer_shape_bem']['airfoil_position']
  chord_data= yaml_data['components']['blade']['outer_shape_bem']['chord']
  twist_data= yaml_data['components']['blade']['outer_shape_bem']['twist']
  pitch_axis_data= yaml_data['components']['blade']['outer_shape_bem']['pitch_axis']
  reference_axis_data= yaml_data['components']['blade']['outer_shape_bem']['reference_axis']
  
  return airfoil_position_data, chord_data, twist_data, pitch_axis_data, reference_axis_data

def main():

  # get the filename after checking input for obvious issues
  filename= check_input(sys.argv)

  # try to safe read
  with open(filename, 'r') as infile:
    data= yaml.safe_load(infile)
  
  # get the blade profile data
  airfoil_position_data, chord_data, twist_data, pitch_axis_data, reference_axis_data= extract_blade_vectors(data)

  plt.style.use(plot_cvf.get_stylesheets(dark= True))
  
  plt.plot(chord_data.grid, chord_data.values)

  plt.show()


if __name__ == '__main__':
  main()
