path = "strongScaling2MPI/100D2/"
data100D2 = {"2":[], "4":[], "8":[], "16":[], "32":[], "64":[]}
for i in range(1,11):
    for k in [2, 4, 8, 16, 32, 64]:
      try:
        with open(f"{path}{i}-{k}.log", "r") as f:
            a = f.read().split("\n")
            for line in a:
                if "Computation: " in line:
                    time = float(line.split()[1])
                    data100D2[str(k)].append(time)
      except:
        continue
for key in data100D2:
  if (len(data100D2[key]) > 0):
    data100D2[key] = sum(data100D2[key]) / len(data100D2[key])
  else:
    continue
