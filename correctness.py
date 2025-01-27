with open("test.out", "r") as f:
    a = f.read().split("\n")
    a = [x for x in a if x != ""]
with open("testseq.out", "r") as f:
    b = f.read().split("\n")
    b = [x for x in b if x != ""]
with open("resultsCuda/test100D2.out", "r") as f:
    c = f.read().split("\n")
    c = [x for x in c if x != ""]

# variable b contains output of sequential version
print(f"MPI + OpenMP version has the same output as given implementation: {a == b}")
if (a == c):
    print(f"CUDA version has the same output as given implementation: {a == c}")
else:
    assert (len(c) == len(a)), "[*]ERROR: sequential output and CUDA output have different lengths"
    tot = len(a)
    equal = 0
    for i in range(len(a)):
        if a[i] == c[i]:
            equal += 1
    print(f"CUDA correctness: {(equal / tot) * 100}%")
