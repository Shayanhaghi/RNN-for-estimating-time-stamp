x = set()
x.add("shayan")
x.add("maryan")
x.add("mahta")
x.add("sepehr")
print(x)
list_x = list(x)
print(list_x)
index2name_dict = {}
name2index_dict = {}
for index, name in enumerate(list_x):
    index2name_dict[index] = name
    name2index_dict[name] = index
print(name2index_dict["shayan"])
print(index2name_dict[1])

