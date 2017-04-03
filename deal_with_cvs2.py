
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
objective_users = CVs

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


def get_activity(profile):
    repos = profile['contributedRepositories']['edges']
    res = 0
    for repo in repos:
        repo = repo['node']
        contribution = repo["contribution"]

        if contribution == 0 or contribution == -1:
            contribution = {"adds": 0, "dels": 0, "total_commits": 0, "num_of_weeks": 0}

        res += contribution["total_commits"]
    return res

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

    sorted_repos = sorted(repos, key=lambda repo: -repo['node']["watchers"]["totalCount"])
    sorted_repos = [node["node"]["name"] for node in sorted_repos if isinstance(node["node"]["contribution"], dict)]
    # print(sorted_repos)
    contribute_repos = sorted_repos[:6]
    if profile["pinnedRepositories"]["totalCount"] > 0:
        contribute_repos = [repo["name"] for repo in profile["pinnedRepositories"]["nodes"]]

    conStars, conForks, conWatchers, conAdds, conDels = 0, 0, 0, 0, 0
    for repo in repos:
        repo = repo['node']
        if repo["name"] in contribute_repos:
            conStars += repo["stargazers"]["totalCount"]
            conForks += repo["forks"]["totalCount"]


        ball_repo += estimate_repo(repo)

    if len(contribute_repos) > 0:
        conStars /= len(contribute_repos)
        conForks /= len(contribute_repos)


    return (ball_user, ball_repo, ball_user + ball_repo, followers,
            gists, pullRequests, conStars, conForks,
            len(contribute_repos), get_activity(profile))


data = []

for name in objective_users:
    with open(join(cv_path, name)) as infile:
        profile = json.load(infile)
    ball = estimate_profile(profile)
    data.append(ball)
    print(name, ":", ball[:3])

data = np.array(data)

# ===================== GRAPH ============
# plt.figure()
# plt.scatter(data[:, 0], data[:, 1], s=70, alpha=0.4)
# plt.show()
# exit()


Y = data[:, 2]
X = data[:, -7:]
# X = np.concatenate((X, standard), axis=1)
X = preprocessing.scale(X)

print(X.shape)
print(X[:3, :])
# exit()
from sklearn.neural_network import MLPRegressor
reg = MLPRegressor(max_iter=10000, alpha=1e-5, hidden_layer_sizes=(50, 6))

reg.fit(X[25:], Y[25:])

for i in range(25):
    # print(X[i])
    print("Predicted:", reg.predict([X[i]]),"Actual value:", Y[i])

def J(reg, Xs, ys):
    predicted = reg.predict(Xs)
    temp = (predicted - ys)
    Jval = (1 / len(Xs)) * temp.dot(temp)
    return Jval


Xtest = X[:150]
Ytest = Y[:150]
Xtrain = X[150:]
Ytrain = Y[150:]



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

