import numpy as np

# Load the .npy file
data = np.load('dataset.npy')

# Check the shape or content
print(data.shape)
data = data.reshape((514, 700, 57))
print(data.shape)  # should print (514, 700, 57)
#print(data)
import numpy as np

def q8_to_q3(q8_onehot):
    """
    Convert Q8 one-hot encoded array to Q3 one-hot encoded array.
    
    q8_onehot shape: (N, 700, 9)  # 9 Q8 features including NoSeq
    
    Returns q3_onehot with shape (N, 700, 4) for H, E, C, NoSeq
    """
    # Define Q8 to Q3 mapping by indices:
    # H = G(3), I(4), H(5)
    # E = B(1), E(2)
    # C = L(0), S(6), T(7)
    # NoSeq = 8
    
    N, L, _ = q8_onehot.shape
    
    q3_onehot = np.zeros((N, L, 4))
    
    # Map H (Helix)
    q3_onehot[:, :, 0] = q8_onehot[:, :, 3] + q8_onehot[:, :, 4] + q8_onehot[:, :, 5]
    # Map E (Strand)
    q3_onehot[:, :, 1] = q8_onehot[:, :, 1] + q8_onehot[:, :, 2]
    # Map C (Coil)
    q3_onehot[:, :, 2] = q8_onehot[:, :, 0] + q8_onehot[:, :, 6] + q8_onehot[:, :, 7]
    # NoSeq
    q3_onehot[:, :, 3] = q8_onehot[:, :, 8]
    
    return q3_onehot

# Example usage:
# Assuming `data_reshaped` is your (N, 700, 57) numpy array

q8_onehot = data[:, :, 22:31]  # Extract Q8 secondary structure features
q3_onehot = q8_to_q3(q8_onehot)

print('Q3 one-hot shape:', q3_onehot.shape)



# Assuming you already have:
# data with shape (514, 700, 57)
# q3_onehot with shape (514, 700, 4)

# Remove Q8 slice (features 22 to 30)
before_q8 = data[:, :, :22]        # features before Q8
#after_q8 = data[:, :, 31:]         # features after Q8

# Concatenate before_q8 + q3_onehot + after_q8 along the features axis (axis=2)
data_q3 = np.concatenate([before_q8, q3_onehot], axis=2)

print('Original data shape:', data.shape)          # (514, 700, 57)
print('New data with Q3 shape:', data_q3.shape)    # (514, 700, 26)

# Decode amino acid sequences and secondary structures from one-hot encoded data


# aa_order = ['A', 'C', 'E', 'D', 'G', 'F', 'I', 'H', 'K', 'M', 'L',
#             'N', 'Q', 'P', 'S', 'R', 'T', 'W', 'V', 'Y', 'X', 'NoSeq']
# noseq_index = 21  # index of 'NoSeq' in aa_order

# ss_order = ['H', 'E', 'C', 'NoSeq']
# ss_noseq_index = 3  # index of 'NoSeq' in ss_order

# def trim_noseq_inputs_labels(data, aa_onehot_start=0, aa_onehot_end=22, noseq_index=21, label_start=22, label_end=26):
#     N, L, F = data.shape
#     inputs = []
#     labels = []
    
#     for i in range(N):
#         aa_onehot = data[i, :, aa_onehot_start:aa_onehot_end]
#         noseq_mask = aa_onehot[:, noseq_index] == 1
        
#         if np.all(noseq_mask):
#             trimmed_length = 0
#         else:
#             trimmed_length = np.max(np.where(~noseq_mask)[0]) + 1
        
#         # Trim inputs (AA one-hot)
#         trimmed_input = data[i, :trimmed_length, aa_onehot_start:aa_onehot_end]
#         # Trim labels (secondary structure one-hot)
#         trimmed_label = data[i, :trimmed_length, label_start:label_end]
        
#         inputs.append(trimmed_input)
#         labels.append(trimmed_label)
    
#     return inputs, labels

# def onehot_to_string(onehot_array, order, noseq_index):
#     indices = np.argmax(onehot_array, axis=1)
#     # Find last residue not equal to noseq_index (defensive, though trimmed already)
#     valid_indices = np.where(indices != noseq_index)[0]
#     if len(valid_indices) == 0:
#         return ''  # empty sequence
#     last_valid = valid_indices[-1] + 1
#     trimmed_indices = indices[:last_valid]
#     return ''.join(order[i] for i in trimmed_indices)

# # Trim the data first
# inputs_trimmed, labels_trimmed = trim_noseq_inputs_labels(data_q3)

# # Decode and print first 5 sequences and secondary structures after trimming
# for i in range(min(5, len(inputs_trimmed))):
#     seq_str = onehot_to_string(inputs_trimmed[i], aa_order, noseq_index)
#     ss_str = onehot_to_string(labels_trimmed[i], ss_order, ss_noseq_index)
#     print(f"Sequence {i+1} (length {inputs_trimmed[i].shape[0]}):\n{seq_str}\n")
#     print(f"Secondary Structure {i+1} (length {labels_trimmed[i].shape[0]}):\n{ss_str}\n")

# print("Total sequences:", len(inputs_trimmed))
# print("Total labels:", len(labels_trimmed))

aa_order = ['A', 'C', 'E', 'D', 'G', 'F', 'I', 'H', 'K', 'M', 'L',
            'N', 'Q', 'P', 'S', 'R', 'T', 'W', 'V', 'Y', 'X', 'NoSeq']

ss_order = ['H', 'E', 'C', 'NoSeq']

inputs = data_q3[:, :, 0:22]
labels = data_q3[:, :, 22:26]

def onehot_to_string_full(onehot_array, order):
    indices = np.argmax(onehot_array, axis=2)
    N, L = indices.shape
    strings = []
    for i in range(N):
        seq = ''.join(order[idx] for idx in indices[i])
        strings.append(seq)
    return strings

seq_strings = onehot_to_string_full(inputs, aa_order)
ss_strings = onehot_to_string_full(labels, ss_order)

for i in range(5):
    print(f"Sequence {i+1} (length 700):\n{seq_strings[i]}\n")
    print(f"Secondary Structure {i+1} (length 700):\n{ss_strings[i]}\n")
    
print ("sequence length:", len(seq_strings))
print("label length:", len(ss_strings))



# def split_dataset(inputs, labels, train_frac=0.7, val_frac=0.15, test_frac=0.15, shuffle=True, random_seed=42):
#     assert len(inputs) == len(labels), "Inputs and labels must have the same length"
#     N = len(inputs)

#     if shuffle:
#         np.random.seed(random_seed)
#         indices = np.random.permutation(N)
#     else:
#         indices = np.arange(N)
    
#     train_end = int(train_frac * N)
#     val_end = train_end + int(val_frac * N)
    
#     train_idx = indices[:train_end]
#     val_idx = indices[train_end:val_end]
#     test_idx = indices[val_end:]
    
#     train_inputs = [inputs[i] for i in train_idx]
#     train_labels = [labels[i] for i in train_idx]
    
#     val_inputs = [inputs[i] for i in val_idx]
#     val_labels = [labels[i] for i in val_idx]
    
#     test_inputs = [inputs[i] for i in test_idx]
#     test_labels = [labels[i] for i in test_idx]
    
#     return (train_inputs, train_labels), (val_inputs, val_labels), (test_inputs, test_labels)

# # Example usage:
# (train_inputs, train_labels), (val_inputs, val_labels), (test_inputs, test_labels) = split_dataset(inputs_trimmed, labels_trimmed)

# print(f'Training samples: {len(train_inputs)}')
# print(f'Validation samples: {len(val_inputs)}')
# print(f'Test samples: {len(test_inputs)}')
