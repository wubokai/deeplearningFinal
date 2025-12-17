import os

with open(r"E:\baselines\Wav2Lip\evaluation\test_filelists\lrs3.txt", "r") as file:
    with open("lrs3.txt", "a") as f:
        for line in file.readlines():
            a, b = line.strip().split()
            f.write(a + " " + a +  '\n' + b + " " + b + '\n' )
