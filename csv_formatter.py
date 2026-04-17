import pandas as pd
import numpy as np

buff = []
with open("CB_ARW2.csv", "r") as file:
    data = False
    for line in file:
        
        if line == "DATA_START\n":
            data = True
            continue
        
        if data == True:
            # line = line.split("\n")
            # line = line[0]
            line = line.rstrip()
            values = line.split(",")
            # print(values)
            buff.append(values)
            # print(line.strip())

    # print(buff[0])

    header = [b.lstrip("inertial-6286.188861:") for b in buff[0]]
    
    # print(header)
    # print(type(header))
    
    # print(type(buff))

    buff = np.array(buff)
    # print(buff[1:])

    # print(buff)
    # print(buff.shape)

# for line in buff:

    df = pd.DataFrame(data=buff[1:], columns=header)

print(df)