import pandas as pb
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import os.path

directory = 'C:\ProjectAnal\Recording Data'


def get_new_col(data):
    data['Speed_R'] = ((data['RX'].diff() / data['T'].diff()) ** 2 + (data['RY'].diff() / data['T'].diff()) ** 2) ** 0.5
    data['Speed_L'] = ((data['LX'].diff() / data['T'].diff()) ** 2 + (data['LY'].diff() / data['T'].diff()) ** 2) ** 0.5
    data['F_R'] = data['Speed_R'].apply(np.floor)
    data['F_L'] = data['Speed_L'].apply(np.floor)
    data['S_R'] = ((data['RX'].diff()) ** 2 + (data['RY'].diff()) ** 2) ** 0.5
    data['S_L'] = ((data['LX'].diff()) ** 2 + (data['LY'].diff()) ** 2) ** 0.5
    data['Speed_Delta'] = data['Speed_R'] - data['Speed_L']
    return data


def read_files_in_directory(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".txt") and file != 'Subject codes.txt':
                file_path = os.path.join(root, file)

                df = pd.read_csv(file_path, sep='\t', index_col=None, decimal=",")
                df = get_new_col(df)
                fig, ax = plt.subplots()
                if root[-1] in ['1']:
                    plt.xlabel("Брак Д")
                elif root[-1] == '2':
                    plt.xlabel("Брак М")
                elif root[-1] == '3':
                    plt.xlabel("Норм Д")
                else:
                    plt.xlabel("Норм М")
                plt.plot(df['RX'], df['RY'], df['LX'], df['LY'])
                plt.show()



read_files_in_directory('C:\ProjectAnal\Recording Data')


