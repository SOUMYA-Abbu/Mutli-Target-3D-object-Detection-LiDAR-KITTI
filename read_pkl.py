# import pickle
# from pprint import pprint

# # 1) Open and load
# with open('/raid/cs21resch15003/mmdetection3d/VoxAttenEncoder/res_dir/pred_instances_3d.pkl', 'rb') as f:
#     data = pickle.load(f)

# # 2) Check what you got
# print(type(data))       # e.g., dict, list, numpy.ndarray, custom class, etc.
# print()

# # 3) Inspect contents
# if isinstance(data, dict):
#     print("Keys:", list(data.keys()))
#     pprint(data)        # pretty-print the full dict
# elif isinstance(data, list):
#     print(f"List of length {len(data)}; first element:")
#     pprint(data[0])
# elif hasattr(data, 'head'):  # often pandas.DataFrame or Series
#     print(data.head())
# else:
#     # Fallback: try a generic print (or dir() to see attributes)
#     print(data)
#     print("Attributes:", [attr for attr in dir(data) if not attr.startswith('_')])



import pickle

with open('/raid/cs21resch15003/mmdetection3d/data/kitti/files/kitti_infos_train.pkl', 'rb') as f:
    data = pickle.load(f)

print(data['data_list'][0].keys())

