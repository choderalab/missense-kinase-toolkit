# mkt.app

This package provides a Streamlit app to view various results and outputs from other sub-packages.

To generate the reference structure, we loaded `1gag`, manually adjusted so that the N-terminus faced upwards and the C-helix faced rightward, and followed the transform object instructions [here](https://pymolwiki.org/index.php/Transform_object) and below:
```
cv=list(cmd.get_view())
cmd.transform_selection("all", \
  cv[0:3]+[0.0]+ \
  cv[3:6]+[0.0]+ \
  cv[6:9]+[0.0]+ \
  cv[12:15]+[1.0], transpose=1)
cmd.reset()
```

https://github.com/pkiruba/rmsd_using_pymol/blob/master/rmsd_analysis.py
https://colab.research.google.com/github/daveminh/Chem456-2022F/blob/main/exercises/02-structural_visualization.ipynb
http://www.bahargroup.org/prody/tutorials/structure_analysis/pdbfiles.html#parse-a-file
