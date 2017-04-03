import json
from os import listdir
from os.path import isfile, join
from math import log
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing

# cv_path = "CVs"
cv_path = join("CVsGermany", "CVs")
CVs = [f for f in listdir(cv_path) if isfile(join(cv_path, f))]

print(CVs)


def estimate_repo(repo):
    watchers = repo["watchers"]["totalCount"]
    forks = repo["forks"]["totalCount"]
    contribution = repo["contribution"]
    stars = repo["stargazers"]["totalCount"]

    if contribution == 0 or contribution == -1:
        contribution = {"adds": 0, "dels": 0, "total_commits": 0, "num_of_weeks": 0}

    b, l = max(contribution["adds"], contribution["dels"]), min(contribution["adds"], contribution["dels"])

    N, S, W, F = abs(b - l/2), stars, watchers, forks
    return 2 * log(1 + (N/(3*10**4)) * (S + W * 3 + F * S / 100))
    # return 10 * N/(10**5) * log(1 + ((S + W * 3 + F * S / 100))/10)
    # return 5*log(1 + (N/(10**5) * (S + W * 2 + F * S / 100)))
    # return N/(10**10)*(S + W * 2 + F * S / 1000)


def estimate_profile(profile):
    followers = profile['followers']['totalCount']
    gists = profile["gists"]['totalCount']
    pullRequests = profile["pullRequests"]['totalCount']
    repos = profile['contributedRepositories']['edges']
    following = profile["following"]['totalCount']
    contributedRepositories = profile["contributedRepositories"]['totalCount']
    starredRepositories = profile["starredRepositories"]['totalCount']
    organizations = profile["organizations"]['totalCount']
    watching = profile["watching"]['totalCount']

    ball_user = 10 * log(1 + followers)
    ball_repo = 0

    sorted_repos = sorted(repos, key=lambda repo: -repo['node']["stargazers"]["totalCount"])
    sorted_repos = [node["node"]["name"] for node in sorted_repos if isinstance(node["node"]["contribution"], dict)]
    # print(sorted_repos)
    contribute_repos = sorted_repos[:10]

    conStars, conForks, conWatchers, conAdds, conDels = 0, 0, 0, 0, 0
    for repo in repos:
        repo = repo['node']
        if repo["name"] in contribute_repos:
            conStars += repo["stargazers"]["totalCount"]
            conForks += repo["forks"]["totalCount"]

            # only for manual version, not for neural network
            conWatchers += repo["watchers"]["totalCount"]
            conAdds += repo["contribution"]["adds"]
            conAdds += repo["contribution"]["dels"]

        ball_repo += estimate_repo(repo)

    if len(contribute_repos) > 0:
        conStars /= len(contribute_repos)
        conForks /= len(contribute_repos)


    return (ball_user, ball_repo, ball_user + ball_repo, followers, gists, pullRequests, conStars, conForks,
            conWatchers, len(contribute_repos), conAdds, conDels)
# , fullStars)
objective_users = CVs
# objective_users = ["ALEXSSS","jacquev6", "mbostock" ,"kinmanz", "dhh", "jasondavies", "tfoote"]
# objective_users = ["mbostock", "kinmanz"]

data = []
for name in objective_users:
    with open(join(cv_path, name)) as infile:
        profile = json.load(infile)
    ball = estimate_profile(profile)
    data.append(ball)
    print(name, ":", ball[:3])

    # # manual version
    # cS, cF, cW, cLen, cAdds, cDels = ball[-6:]
    # cAdds /= cLen
    # cDels /= cLen
    # cS /= cLen
    # cF /= cLen
    # cW /= cLen
    # b, l = max(cAdds, cDels), min(cAdds, cDels)
    # N = b - l
    # ball_user_manual = ball[0]
    # ball_repo_manual = cLen * (2 * log(1 + (N/(3*10**4)) * (cS + cW * 3 + cF * cS / 100)))
    # print(name, "- manual:", (ball_user_manual, ball_repo_manual, ball_user_manual + ball_repo_manual))
    # print("- - - - -")
# exit()
data = np.array(data)

# ===================== GRAPH ============
# plt.figure()
# plt.scatter(data[:, 0], data[:, 2], s=70, alpha=0.4)
# plt.show()
# exit()

# ======================= Part 2? create data ===================

Y = data[:, 2]
pinned = data[:, -6:]
print(pinned[:10, :])
X = []
for name in objective_users:
    with open(join(cv_path, name)) as infile:
        profile = json.load(infile)

    followers = profile['followers']['totalCount']
    following = profile["following"]['totalCount']
    contributedRepositories = profile["contributedRepositories"]['totalCount']
    starredRepositories = profile["starredRepositories"]['totalCount']
    pullRequests = profile["pullRequests"]['totalCount']
    organizations = profile["organizations"]['totalCount']
    watching = profile["watching"]['totalCount']
    gists = profile["gists"]['totalCount']

    # X.append([followers, following, contributedRepositories, starredRepositories,
    #           pullRequests, organizations, watching, gists])
    X.append([followers,  contributedRepositories,
              pullRequests,  gists])


X = np.array(X)
X = np.concatenate((X, pinned), axis=1)
X = preprocessing.scale(X)
# print(Y)
# print(X)

from sklearn.neural_network import MLPRegressor

reg = MLPRegressor(max_iter=10000, alpha=1e-7, hidden_layer_sizes=(50, 2))

# ( 200 1 alpha: 1e-05 )
# ( 100 2 alpha: 1e-06 )
# ( 50 5 alpha: 1e-05 )
#  ( 50 2 alpha: 1e-06 ) best
# ( 50 1 alpha: 1e-08 )
# ( 50 1 alpha: 1e-06 )
# ( 40 2 alpha: 1e-07 )
# ( 30 2 alpha: 1e-08 )
# ( 30 1 alpha: 1e-06 )
# ( 20 3 alpha: 1e-08 )


reg.fit(X[25:], Y[25:])
print("/////////////////////////////////")
# print(X[15:20])
# print("/////////////////////////////////")
# print(X[:20])

for i in range(25):
    # print(X[i])
    print("Predicted:", reg.predict([X[i]]),"Actual value:", Y[i])


def J(reg, Xs, ys):
    predicted = reg.predict(Xs)
    temp = (predicted - ys)
    Jval = (1 / len(Xs)) * temp.dot(temp)
    return Jval

Xtest = X[:60]
Ytest = Y[:60]
Xtrain = X[60:]
Ytrain = Y[60:]

print(len(Xtrain))
print(len(Xtest))

m = []
Jcv = []
Jtrain = []
# for s in [ 300, 325, 350, 400, 450, 500, 550, 600, 650, 700, 750, len(Xtrain)]:
#     print(s)
#     reg = MLPRegressor(max_iter=5000, alpha=1e-7, hidden_layer_sizes=(400, 1))
#     reg.fit(Xtrain[ : s], Ytrain[: s])
#     m.append(len(Xtrain[: s]))
#     Jcv.append(J(reg, Xtest, Ytest))
#     Jtrain.append(J(reg, Xtrain, Ytrain))
#
# print(J(reg, Xtest, Ytest))
# print(J(reg, Xtrain, Ytrain))
# plt.plot(m, Jcv, 'r--', m, Jtrain, "bs")
# plt.show()


models = [
    (15, 1),
    (15, 2),
    (20, 1),
    (20, 2),
    (20, 3),
    (30, 1),
    (30, 2),
    (40, 1),
    (40, 2),
    (50, 1),
    (50, 2),
    (50, 3),
    (50, 4),
    (50, 5),
    (50, 6),
    (100, 1),
    (100, 2),
    (100, 3),
    (150, 1),
    (150, 2),
    (170, 1),
    (170, 2),
    (200, 1),
    (200, 2),
    (300, 1),
    (370, 1),
]
for neurons_number, layers in models:
    for alpha in [1e-5, 1e-6, 1e-7, 1e-8]:
        print("======+ (", neurons_number, layers, "alpha:", alpha,") ==========")
        for i in range(3):
            reg = MLPRegressor(max_iter=10000, alpha=alpha, hidden_layer_sizes=(neurons_number, layers))
            reg.fit(Xtrain, Ytrain)

            print("alpha:", alpha, end=", ")
            print("Jtrain:", J(reg, Xtrain, Ytrain), end=", ")
            print("Jcv:", J(reg, Xtest, Ytest), end="\n")

# Review

