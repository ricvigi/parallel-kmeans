from random import randint
with open("test_files/weakScaling/10k", "w") as f:
    for i in range(10000):
        for j in range(100):
            if j == 99:
                string = f"{randint(-100,100)}\n"
                f.write(string)
                continue
            string = f"{randint(-100,100)}\t"
            f.write(string)
