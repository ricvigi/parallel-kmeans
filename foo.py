path = "strongScaling/800k/"
data800k = {"8":[], "16":[], "32":[], "64":[]}
for i in range(1,11):
    for k in [8, 16, 32, 64]:
      try:
        with open(f"{path}{i}-{k}.log", "r") as f:
            a = f.read().split("\n")
            for line in a:
                if "Computation: " in line:
                    time = float(line.split()[1])
                    data800k[str(k)].append(time)
      except:
        continue
for key in data800k:
  if (len(data800k[key]) > 0):
    data800k[key] = sum(data800k[key]) / len(data800k[key])
  else:
    continue

path = "weakScaling/"
data = {"8":[], "16":[], "32":[], "64":[]}
folders = ["100k/", "200k/", "400k/", "800k/"]
for idx in range(len(folders)):
    for i in range(1,11):
        try:
            print(f"{path}{folders[idx]}{i}.log")
            with open(f"{path}{folders[idx]}{i}.log", "r") as f:
                a = f.read().split("\n")
                for line in a:
                    if "Computation: " in line:
                        time = float(line.split()[1])
                        data[list(data.keys())[idx]].append(time)
        except:
            print("Something happened...")
for idx in range(len(folders)):
    if (len(data[list(data.keys())[idx]]) > 0):
        data[list(data.keys())[idx]] = sum(data[list(data.keys())[idx]]) / len(data[list(data.keys())[idx]])
    else:
        print(f"{path}{folders[idx]} has some issues")

path = "sequentialOut/"
folders = ["2D", "2D2", "10D", "20D", "100D", "100k", "200k", "400k", "800k", "1600k"]
data = {x:[] for x in folders}
for idx in range(len(folders)):
    for i in range(1,21):
        try:
            with open(f"{path}{folders[idx]}/{i}.log") as f:
                a = f.read().split("\n")
                for line in a:
                    if "Computation: " in line:
                        time = float(line.split()[1])
                        data[folders[idx]].append(time)
        except:
            print("ERROR")
for idx in range(len(folders)):
    if (len(data[folders[idx]]) > 0):
        data[folders[idx]] = sum(data[folders[idx]]) / len(data[folders[idx]])






