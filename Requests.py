
import requests
import json
import pprint
import os
import github

token = "8bc3a7e4882a27ed699ea57611d7a211fdf8657b"


# "%(name)s"
def get_repos_ql(name, token):
    query = """
    {
  user(login: "%(name)s") {
    pinnedRepositories(first: 100) {
      totalCount
      nodes {
        name
      }
    }
    id
    avatarURL
    name
    login
    location
    followers {
      totalCount
    }
    following{
      totalCount
    }
    starredRepositories{
      totalCount
    }
    watching{
      totalCount
    }
    pullRequests{
      totalCount
    }
    gists{
      totalCount
    }
    company
    email
    websiteURL
    bio
    organizations(first: 100) {
      totalCount
      nodes {
        name
        login
        url
        avatarURL
      }
    }
    contributedRepositories(first: 100) {
      totalCount
      pageInfo {
        endCursor
        hasNextPage
      }
      edges {
        node {
          id
          name
          watchers {
            totalCount
          }
          forks {
            totalCount
          }
          stargazers {
            totalCount
          }
          isFork
          description
          homepageURL
          createdAt
          updatedAt
          primaryLanguage{
            color
            id
            name
          }
          languages(first:100){
            edges{
              size
              node{
                name
                id
                color
              }
            }
          }
        }
      }
    }
  }
}


    """ % {"name": name}

    headers = {'Authorization': 'token ' + token}
    r2 = requests.post('https://api.github.com/graphql', json.dumps({"query": query}), headers=headers)

    return r2.json()

print(get_repos_ql("ALEXSSS", token))

def get_repo_contribution(repo_name, user_name, token):
    con = github.Github(token)
    repo = con.get_user(user_name).get_repo(repo_name)

    contributes = repo.get_stats_contributors()

    if contributes is None: return 0
    our = next((x for x in contributes if x.author.login == user_name), None)
    if our is None: return 0

    weeks = [w for w in our.weeks if w.c > 0]
    return {
        "adds": sum([w.a for w in weeks]),
        "dels": sum([w.d for w in weeks]),
        "total_commits": our.total,
        "num_of_weeks" : len(weeks),
        # "weeks": [w.raw_data for w in weeks]
    #     for PCA it's now redundant for us
    }

# =========================

# ans = get_repo_contribution("working", "kinmanz", token)
# print(ans)


#======================================================

# repos = ans['contributedRepositories']['nodes']
# totalCount = ans['contributedRepositories']['totalCount']
# followers = ans['followers']['totalCount']
#
# for repo in repos:
#     print(repo['name'])
# print(followers)
# print(len(repos))
# print(totalCount)


# pprint.pprint(ans)