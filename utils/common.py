def apply_to_dict_values(dict, f):
    for key, value in dict.items():
        dict[key] = f(value)
