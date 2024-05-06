import numpy as np


def retain_avg_vector(vectors):
	avg_vector = np.mean(vectors)
	vectors_distance = np.zeros(len(vectors))

	for idx, vector in enumerate(vectors):
		vector_minus = np.array(vector - avg_vector)
		euclidean_distance = np.linalg.norm(vector_minus)
		vectors_distance[idx] = euclidean_distance

	avg_vectors_distance = np.mean(vectors_distance)
	results = [vector for idx, vector in enumerate(vectors) if vectors_distance[idx] < avg_vectors_distance]

	return results


def merge_close_vectors(vectors, threshold):
	# Calculate pairwise Euclidean distances between vectors
	dist_matrix = np.linalg.norm(vectors[:, None] - vectors[None, :], axis=-1)

	# Identify close vectors (below the threshold)
	close_indices = dist_matrix < threshold

	# To avoid double counting, ensure indices comparison is only for i < j
	close_indices = np.triu(close_indices, 1)

	# Initialize merged indices set and list for storing merged vectors
	merged_indices = set()
	merged_vectors = []

	for i, vec in enumerate(vectors):
		if i in merged_indices:
			continue

		# Indices of vectors close to the current one, including the current one
		indices = np.where(close_indices[i])[0].tolist()
		indices.append(i)  # Add current vector's index

		if indices:
			# Merge and append the average vector
			merged_vec = np.mean(vectors[indices], axis=0)
			merged_vectors.append(merged_vec)
			merged_indices.update(indices)
		else:
			# If no close vectors, add the vector as is
			merged_vectors.append(vec)

	return np.array(merged_vectors)


# Example usage
# vectors = np.array([[1, 2, 3], [2, 3, 4], [10, 11, 12], [20, 21, 22], [100, 110, 120]])
# threshold = 5
# merged_vectors = merge_close_vectors(vectors, threshold)
# print("Merged vectors:\n", merged_vectors)
