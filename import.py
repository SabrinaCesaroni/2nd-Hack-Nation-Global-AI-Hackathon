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

import numpy as np

# Assuming you already have:
# data with shape (514, 700, 57)
# q3_onehot with shape (514, 700, 4)

# Remove Q8 slice (features 22 to 30)
before_q8 = data[:, :, :22]        # features before Q8
after_q8 = data[:, :, 31:]         # features after Q8

# Concatenate before_q8 + q3_onehot + after_q8 along the features axis (axis=2)
data_q3 = np.concatenate([before_q8, q3_onehot, after_q8], axis=2)

print('Original data shape:', data.shape)          # (514, 700, 57)
print('New data with Q3 shape:', data_q3.shape)    # (514, 700, 52)
