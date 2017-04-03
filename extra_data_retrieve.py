

import json
from os import listdir
from os.path import isfile, join

path = "part_of_data"
with open(join(path, "github-users-stats.json")) as infile:
    profiles = json.load(infile)
    extra_names = [prof['login'] for prof in profiles]

print(extra_names)
print(len(profiles))