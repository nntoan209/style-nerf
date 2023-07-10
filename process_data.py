import os

categories = os.listdir("data/artbench-10-imagefolder")
for category in categories:
    file_names = os.listdir(f"data/artbench-10-imagefolder/{category}")
    for file_name in file_names:
        if file_name.startswith("."):
            os.remove(f"data/artbench-10-imagefolder/{category}/{file_name}")
