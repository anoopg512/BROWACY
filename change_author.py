import os
import subprocess

os.environ["OLD_EMAIL"] = "hunjanriya@gmail.com"
os.environ["CORRECT_NAME"] = "anoopgupta"
os.environ["CORRECT_EMAIL"] = "anoopgupta7122002@gmail.com"

filter_repo_cmd = [
    "git", "filter-repo", "--commit-callback",
    '''
    if commit.author_email == os.environ["OLD_EMAIL"]:
        commit.author_name = os.environ["CORRECT_NAME"]
        commit.author_email = os.environ["CORRECT_EMAIL"]
    if commit.committer_email == os.environ["OLD_EMAIL"]:
        commit.committer_name = os.environ["CORRECT_NAME"]
        commit.committer_email = os.environ["CORRECT_EMAIL"]
    '''
]

subprocess.run(filter_repo_cmd)

print("Author information has been updated.")
