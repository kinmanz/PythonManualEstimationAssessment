import json
import os
import time
from os.path import join

from someCodeForEmbeding.PCA.Requests import get_repos_ql, get_repo_contribution

token = "8bc3a7e4882a27ed699ea57611d7a211fdf8657b"

objective_users = []
# objective_users += ["jacquev6", "krinkin", "voytovichs", "qwwdfsad", "ALEXSSS", "ilya-afanasev", "pacahon", "learp",
#                    "Sviftel", "istarion", "AuthenticEshkinKot", "howareyouman", "shishovv"]
# objective_users += ["mbostock", "tfoote", "jasondavies", "kitmonisit", "dhh", "fxn", "jeremy", "kinmanz"
#                    , "tenderlove", "josevalim", "rafaelfranca", "spastorino", "lifo", "josh", "carlosantoniodasilva"
#                    , "senny", "jonleighton", "vijaydev", "sgrif", "amatsuda", "NZKoz", "technoweenie", "kamipo"
#                    , "drogus", "miloops", "pixeltrix", "kaspth", "arunagw", "radar", "wycats", "y-yagi", "minimaxir"
#                    , "kennethreitz", "QuincyLarson", "abhishekbanthia", "BerkeleyTrue", "ryanmcdermott", "shockcao"
#                    , "garryyan", "apaszke", "soumith", "colesbury", "fchollet", "farizrahman4u", "jondot", "jacknagel"
#                    , "MikeMcQuaid", "adamv", "mxcl", "shelhamer", "kaizensoze", "kilimchoi", "rastating"
#                    , "GrahamCampbell", ]


path_to_extra_data = "CVsGermany"
path_to_CVs = join(path_to_extra_data, "CVs")
with open(join(path_to_extra_data, "GermanUsersAll.json")) as infile:
    objective_users += json.load(infile)

print("Users:", len(objective_users))

userNum = 0
for name in objective_users:
    print("=======", name, "========")
    userNum += 1
    print(userNum , ")", sep="", end=" ")
    if os.path.isfile(join(path_to_CVs, name + ".json")):
        print("It has been already completed.")
        continue

    # time.sleep(4)
    time.sleep(2)
    try:
        ans = get_repos_ql(name, token)['data']['user']
    except:
        print("Error on user !!!")
        continue

    try:
        repos = ans['contributedRepositories']['edges']
    except:
        print("error on edges")
        continue

    for i in range(3):
        for repo in repos:
            repo = repo['node']
            repo_name = repo['name']

            if 'contribution' in repo and repo['contribution'] != 0:
                print(repo_name, ":", "parsed")
                continue

            try:
                repo['contribution'] = get_repo_contribution(repo_name, name, token)
                print(repo_name, ":", repo['contribution'])
            except:
                repo['contribution'] = -1
                print(repo_name, ":", "error")

            time.sleep(1)
        print("====", i + 1, "====")

    with open(join(path_to_CVs, name + ".json"), 'w+') as outfile:
        json.dump(ans, outfile, indent=4)