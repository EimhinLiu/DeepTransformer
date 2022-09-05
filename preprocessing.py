import pandas as pd
import numpy as np
import math

TARGET_CONVERSION = { "a": '0001', "b": '0002', "c": '0003', "d": '0004', "e": '0005', "f": '0006', 
                     "g": '0007', "h": '0008', "i": '0009', "j": '0010', "k": '0011', "l": '0012', 
                     "m": '0013', "n": '0014', "o": '0015', "p": '0016', "q": '0017', "r": '0018', 
                     "s": '0019', "t": '0020', "u": '0021', "v": '0022', "w": '0023', "x": '0024', 
                     "y": '0025', "z": '0026' }

data = pd.read_csv(r".\data\CTI.csv")

target = []
for tar in data['receptor_name']:
    # print(tar)
    t_c = ''
    for t in tar:
        # print(t)
        if t not in TARGET_CONVERSION:
            t_c += t
            # print(t_c)
        if t in TARGET_CONVERSION:
            # print(TARGET_CONVERSION[t])
            t_c += TARGET_CONVERSION[t]
    # print(t_c)
    target.append(t_c)
# print(target)
# print(len(target))

chemical = []
for che in data['ligand_name']:
    che = che.split('_')[0]
    # print(che)
    chemical.append(che)
# print(chemical)
# print(len(chemical))

node = []
node_id = []
for i in range(len(data)):
    # print(int(target[i]+chemical[i]))
    node.append(target[i] + '_' + chemical[i] + '\t')
    node_id.append(i)
# print(node)
# print(node_id)


node_data = pd.DataFrame({'node_id':node_id, 'node':node})
node_data.to_csv(r".\data\graph\raw\node.csv", index = False)
print('node.csv save success!')


node_label = []
for sco in data['scores']:
    # print(sco)
    if sco <= -7.0:
        node_label.append('1')
    else:
        node_label.append('0')
# print(node_label)


node_label_data = pd.DataFrame(node_label)
node_label_data.to_csv(r".\data\graph\raw\node_label.csv", index = False, header = False)
print('node_label.csv save success!')


score = []
for sco in data['scores']:
    # print(sco)
    score.append(sco)
# print(score)
    
Molecular_Weight = []
XLogP3_AA = []
Hydrogen_Bond_Donor_Count = []
Hydrogen_Bond_Acceptor_Count = []
Rotatable_Bond_Count = []
Heavy_Atom_Count = []
Covalently_Bonded_Unit_Count = []

Number_of_amino_acids = []
Molecular_weight = []
Instability_index = []
Aliphatic_index = []
Grand_average_of_hydropathicity = []

Chemical_Features = pd.read_csv(r".\data\Chemical_Features.csv", header = None)
# print(Chemical_Features[0])
Protein_Features = pd.read_csv(r".\data\Protein_Features.csv", header = None)
# print(Protein_Features[0])
for i in range(len(node)):
    # print(node[i].strip().split('_')[1])
    nodes_chemical = node[i].strip().split('_')[1]
    # print(nodes_chemical)
    nodes_target = node[i].strip().split('_')[0]
    # print(nodes_target)
    for c in range(len(Chemical_Features[0])):
        if nodes_chemical == str(Chemical_Features[0][c]):
            Molecular_Weight.append(Chemical_Features[1][c])
            XLogP3_AA.append(Chemical_Features[2][c])
            Hydrogen_Bond_Donor_Count.append(Chemical_Features[3][c])
            Hydrogen_Bond_Acceptor_Count.append(Chemical_Features[4][c])
            Rotatable_Bond_Count.append(Chemical_Features[5][c])
            Heavy_Atom_Count.append(Chemical_Features[6][c])
            Covalently_Bonded_Unit_Count.append(Chemical_Features[7][c])
            break
    for p in range(len(Protein_Features[0])):
        # print(Protein_Features[0][p])
        t_c = ''
        for t in Protein_Features[0][p]:
            if t not in TARGET_CONVERSION:
                t_c += t
            if t in TARGET_CONVERSION:
                t_c += TARGET_CONVERSION[t]
        # print(t_c)
        if nodes_target == t_c:
            Number_of_amino_acids.append(Protein_Features[1][p])
            Molecular_weight.append(Protein_Features[2][p])
            Instability_index.append(Protein_Features[3][p])
            Aliphatic_index.append(Protein_Features[4][p])
            Grand_average_of_hydropathicity.append(Protein_Features[5][p])
            break


node_feature_data = pd.DataFrame({'score':score, 
                                  'Molecular_Weight':Molecular_Weight, 
                                  'XLogP3_AA':XLogP3_AA, 
                                  'Hydrogen_Bond_Donor_Count':Hydrogen_Bond_Donor_Count, 
                                  'Hydrogen_Bond_Acceptor_Count':Hydrogen_Bond_Acceptor_Count, 
                                  'Rotatable_Bond_Count':Rotatable_Bond_Count, 
                                  'Heavy_Atom_Count':Heavy_Atom_Count, 
                                  'Covalently_Bonded_Unit_Count':Covalently_Bonded_Unit_Count, 
                                  'Number_of_amino_acids':Number_of_amino_acids, 
                                  'Molecular_weight':Molecular_weight, 
                                  'Instability_index':Instability_index, 
                                  'Aliphatic_index':Aliphatic_index, 
                                  'Grand_average_of_hydropathicity':Grand_average_of_hydropathicity})
node_feature_data.to_csv(r".\data\graph\raw\node_feature.csv", header=False, index = False)
print('node_feature.csv save success!')


Chemical_Similar = pd.read_csv(r".\data\Chemical_Similar.csv", header = None)
Protein_Interaction = pd.read_csv(r".\data\Protein_Interaction.csv", header = None)

edge1 = []
edge2 = []

index_target = []
unique_target = np.unique(target)
# print(unique_target)
for ut in unique_target:
    # print(ut)
    ut_index = []
    for _, tar in enumerate(target):
        if tar == ut:
            ut_index.append(_)
    # print(ut_index)
    index_target.append(ut_index)
# print(index_target)
for ind in index_target:
    # print(ind)
    for i in ind:
        # print(node[i])
        # print(node_id[i])
        for j in range(i, ind[-1]+1):
            if j+1 in ind:
                edge1.append(node_id[i])
                edge2.append(node_id[j+1])

for pi in range(len(Protein_Interaction[0])):
    t_c0 = ''
    for t in Protein_Interaction[0][pi]:
        if t not in TARGET_CONVERSION:
            t_c0 += t
        if t in TARGET_CONVERSION:
            t_c0 += TARGET_CONVERSION[t]
    t_c1 = ''
    for t in Protein_Interaction[1][pi]:
        if t not in TARGET_CONVERSION:
            t_c1 += t
        if t in TARGET_CONVERSION:
            t_c1 += TARGET_CONVERSION[t]
    # print(t_c0)
    # print(t_c1)
    t_c0_index = []
    t_c1_index = []
    for _, tar in enumerate(target):
        # print(tar)
        if tar == t_c0:
            t_c0_index.append(_)
        if tar == t_c1:
            t_c1_index.append(_)
    # print(t_c0_index)
    # print(t_c1_index)
    for i in t_c0_index:
        for j in t_c1_index:
            # print(node[j])
            # print(node_id[j])
            edge1.append(node_id[i])
            edge2.append(node_id[j])

index_chemical = []
unique_chemical = np.unique(chemical)
# print(unique_chemical)
for uc in unique_chemical:
    uc_index = []
    for _, che in enumerate(chemical):
        if che == uc:
            uc_index.append(_)
    index_chemical.append(uc_index)
# print(index_chemical)
for ind in index_chemical:
    # print(ind)
    n = 1
    for i in ind:
        for j in range(len(ind)):
            try:
                if ind[j+n] in ind:
                    edge1.append(node_id[i])
                    edge2.append(node_id[ind[j+n]])
            except IndexError:
                n += 1
                break
            
for cs in range(len(Chemical_Similar[0])):
    c0_index = []
    c1_index = []
    for _, che in enumerate(chemical):
        if che == str(Chemical_Similar[0][cs]):
            c0_index.append(_)
        if che == str(Chemical_Similar[1][cs]):
            c1_index.append(_)
    # print(c0_index)
    # print(c1_index)
    for i in c0_index:
        for j in c1_index:
            # print(node[j])
            # print(node_id[j])
            edge1.append(node_id[i])
            edge2.append(node_id[j])

# Delete duplicate edges.
_edge = []
for e in range(len(edge1)):
    _e = str(edge1[e]) + '_' + str(edge2[e])
    _edge.append(_e)
unique_edge = np.unique(_edge)
edge1 = []
edge2 = []
for ue in unique_edge:
    edge1.append(int(ue.split('_')[0]))
    edge2.append(int(ue.split('_')[1]))


edge_data = pd.DataFrame({'edge1':edge1, 'edge2':edge2})
edge_data.to_csv(r".\data\graph\raw\edge.csv", header=False, index = False)
edge_data_copy = pd.DataFrame({'edge1':edge2, 'edge2':edge1})
edge_data_copy.to_csv(r".\data\graph\raw\edge.csv", mode='a', header=False, index = False)
print('edge.csv save success!')


edge_feature = []
for i in range(len(edge1)):
    chemical_similar = False
    target_interaction = False
    # print(edge1[i])
    # print(edge2[i])
    edge1_chemical = node[edge1[i]].strip().split('_')[1]
    edge1_target = node[edge1[i]].strip().split('_')[0]
    edge2_chemical = node[edge2[i]].strip().split('_')[1]
    edge2_target = node[edge2[i]].strip().split('_')[0]
    # print(edge1_chemical)
    # print(edge1_target)
    # print(edge2_chemical)
    # print(edge2_target)
    for cs in range(len(Chemical_Similar[0])):
        if (edge1_chemical == str(Chemical_Similar[0][cs]) and edge2_chemical == str(Chemical_Similar[1][cs])) or (edge1_chemical == str(Chemical_Similar[1][cs]) and edge2_chemical == str(Chemical_Similar[0][cs])):
            chemical_similar = True
    for pi in range(len(Protein_Interaction[0])):
        t_c0 = ''
        for t in Protein_Interaction[0][pi]:
            if t not in TARGET_CONVERSION:
                t_c0 += t
            if t in TARGET_CONVERSION:
                t_c0 += TARGET_CONVERSION[t]
        t_c1 = ''
        for t in Protein_Interaction[1][pi]:
            if t not in TARGET_CONVERSION:
                t_c1 += t
            if t in TARGET_CONVERSION:
                t_c1 += TARGET_CONVERSION[t]
        if (edge1_target == t_c0 and edge2_target == t_c1) or (edge1_target == t_c1 and edge2_target == t_c0):
            target_interaction = True
    if (edge1_chemical == edge2_chemical or chemical_similar == True) and (edge1_target == edge2_target or target_interaction == True):
        edge_feature.append(1)
    elif edge1_chemical == edge2_chemical or chemical_similar == True or edge1_target == edge2_target or target_interaction == True:
        edge_feature.append(0.5)


edge_feature_data = pd.DataFrame(edge_feature)
edge_feature_data.to_csv(r".\data\graph\raw\edge_feature.csv", index = False, header = False)
edge_feature_data_copy = pd.DataFrame(edge_feature)
edge_feature_data_copy.to_csv(r".\data\graph\raw\edge_feature.csv", mode='a', index = False, header = False)
print('edge_feature.csv save success!')


train = []
valid = []
test = []
evaluate = []
TrainValidTest_Num = 3682 # Number of internal train, valid and test sets.
train_num = math.ceil(TrainValidTest_Num * 0.8) # 2946
valid_num = train_num + math.ceil((TrainValidTest_Num - train_num) * 0.5) # 3314
for i in range(len(node)):
    if i < train_num:
        train.append(node_id[i])
    elif i >= train_num and i < valid_num:
        valid.append(node_id[i])
    elif i >= valid_num and i < TrainValidTest_Num:
        test.append(node_id[i])
    else:
        evaluate.append(node_id[i]) # 653 nodes are used as external test sets.


trains = pd.DataFrame(train)
trains.to_csv(r".\data\graph\split\train.csv", index = False, header = False)
print('train.csv save success!')
valids = pd.DataFrame(valid)
valids.to_csv(r".\data\graph\split\valid.csv", index = False, header = False)
print('valid.csv save success!')
tests = pd.DataFrame(test)
tests.to_csv(r".\data\graph\split\test.csv", index = False, header = False)
print('test.csv save success!')
evaluates = pd.DataFrame(evaluate)
evaluates.to_csv(r".\data\graph\split\evaluate.csv", index = False, header = False)
print('evaluate.csv save success!')