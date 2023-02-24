
import os.path

file_dir= os.path.split(__file__)[0]
stylesheet_seaborn_base_name= 'stylesheet_seaborn.mplstyle'
stylesheet_cvf_base_name= 'stylesheet_seaborn.mplstyle'

def get_stylesheets(style= None, dark= False,
                    seaborn_base= True, cvf_base= True,
                    use_latex= True):
  out_list= []
  
  # use the matplotlib color scheme by default, dark if specified
  if dark: out_list.append("dark_background")

  # building up, use the seaborn base settings I extracted from their github
  if seaborn_base:
    out_list.append(os.path.join(file_dir, stylesheet_seaborn_base_name))

  # next, apply cory's custom changes
  if cvf_base and use_latex:
    out_list.append(os.path.join(file_dir, stylesheet_cvf_base_name))
  elif cvf_base:
    raise NotImplementedError("need to implement a latex-free custom stylesheet still.")
  
  return out_list # kick out the result
