import numpy as np

feature_map=np.array([
    [6,3,1,4],
    [2,6,5,9],
    [4,2,7,8],
    [3,1,2,5]
])

def max_pooling(matrix,size=2,stride=2):
    output_shape=((matrix.shape[0]-size) // stride + 1, (matrix.shape[1]-size) //stride+1)
    pooled=np.zeros(output_shape)

    for i in range (0, matrix.shape[0] - size+1, stride):
        for j in range (0,matrix.shape[1]-size+1,stride):
            pooled[i//stride][j//stride]=np.max(matrix[i:i+size,j:j+size])
    return pooled


def average_pooling(matrix,size=2,stride=2):
    output_shape=((matrix.shape[0]-size)//stride+1,(matrix.shape[1] - size)//stride+1)
    pooled = np.zeros(output_shape)

    for i in range(0,matrix.shape[0]-size+1,stride):
        for j in range(0,matrix.shape[1]-size+1,stride):
            pooled[i//stride][j//stride]=np.mean(matrix[i:i+size,j:j+size])

    return pooled

max_pooled_output=max_pooling(feature_map)
print("Max Pooled Output:\n", max_pooled_output)

avg_pooled_output=average_pooling(feature_map)
print("\nAverage Pooled Output:\n",avg_pooled_output)

flattened_max = max_pooled_output.flatten()
flattened_avg = avg_pooled_output.flatten()

print("\nFlattened Max Pooled Output:\n", flattened_max)
print("\nFlattened Average Pooled Output:\n", flattened_avg)