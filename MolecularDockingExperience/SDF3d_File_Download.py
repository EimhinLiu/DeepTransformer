import os
import pubchempy as pcp

with open("SDF_CID.txt", "r") as f: # Open file.
    data = f.read().split(' ') # Read file.
    for i in range(len(data)):
        download_path = os.path.join('C:/....../MolecularDockingExperience/ligand/', 
                                     "{}.sdf".format(data[i]))
        try:
            pcp.download('SDF', download_path, overwrite=True, identifier=data[i], record_type='3d')
        except pcp.NotFoundError as e:
            print('No 3d Conformer for {}.'.format(data[i]))