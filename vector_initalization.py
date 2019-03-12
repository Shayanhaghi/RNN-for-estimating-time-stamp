import numpy as np




def extract_user_number(user_string):
    return int(user_string[-3:])


if __name__ == "__main__":
    user_embedding_matrix = np.load("user_embedding_matrix.npy")
    user2number_dict = np.load("user2number_dict")
    number2user_dict = np.load("number2user_dict")
    # print(user_embedding_matrix.shape)
    user_embedding_matrix_sorted = np.zeros([1100, 100])
    print(user_embedding_matrix_sorted.dtype)
    print(user_embedding_matrix_sorted)
    for i in range(len(user_embedding_matrix)):
        row = user_embedding_matrix[i]
        user_number = extract_user_number(number2user_dict[i])
        # user_embedding_matrix_sorted[]
        user_embedding_matrix_sorted[user_number, :] = row
    print(user_embedding_matrix_sorted.mean())
    print(user_embedding_matrix.mean())
    user_embedding_matrix_sorted = np.float32(user_embedding_matrix_sorted)
    np.save("user_embedding_matrix_sorted.npy", user_embedding_matrix_sorted)
    # user_000095


