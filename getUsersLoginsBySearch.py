

import github
import json
import time
from os import listdir
from os.path import isfile, join
from random import shuffle

token = "8bc3a7e4882a27ed699ea57611d7a211fdf8657b"
g = github.Github(token)

searchQueries = [
    "location:Germany  type:user followers:8..8",
    "location:Germany  type:user followers:9..9",
    "location:Germany  type:user followers:10..11",
    "location:Germany  type:user followers:12..13",
    "location:Germany  type:user followers:14..16",
    "location:Germany  type:user followers:17..21",
    "location:Germany  type:user followers:22..32",
    "location:Germany  type:user followers:33..66",
    "location:Germany  type:user followers:>=67",
]

i = 0
for query in searchQueries:
    print("Query:", query)
    if isfile(join("CVsGermany", query)):
        print("It has been already completed.")
        continue
    userLogins = []
    res = g.search_users(query)

    for user in res:
        time.sleep(1)
        i += 1
        login = user.login

        print(i, login)
        userLogins.append(login)

    print("Safe query result:", query, "Entities:", len(userLogins))

    with open(join("CVsGermany/",query), 'w+') as outfile:
        json.dump(userLogins, outfile, indent=4)

userLogins = []
path = "CVsGermany/"
for file in listdir("CVsGermany/"):
    with open(join(path, file)) as infile:
        userLogins += json.load(infile)
print(userLogins)
print(len(userLogins))

shuffle(userLogins)

print(userLogins)
print(len(userLogins))

with open(join("CVsGermany/", "GermanUsersAll.json"), 'w+') as outfile:
    json.dump(userLogins, outfile, indent=4)