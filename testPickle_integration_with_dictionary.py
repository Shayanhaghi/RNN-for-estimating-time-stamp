from matrix_factorization import pickle_load
from matrix_factorization import pickle_save
dictionary_1 = {}
dictionary_2 = {}
dictionary_1[0] = "1098"
dictionary_2[3] = "34"
name1 = "test"
name2 = "test434"
pickle_save(name1, dictionary_1)
pickle_save(name2, dictionary_2)
data = pickle_load(name1)
data2 = pickle_load(name2)
print(data)
print(data2)

