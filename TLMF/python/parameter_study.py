import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d


all_train = "./run_all-tag-LOSS_train_normal_rmse_modified.csv"
# all_train = "./run_com-tag-LOSS_train_normal_rmse.csv"
all_val = "./run_all-tag-MSE_rmse.csv"
# all_val = "./run_com-tag-MSE_rmse.csv"

all_matrix = np.loadtxt(all_train, dtype=float, delimiter=',', skiprows=1)
all_step = all_matrix[::, 1]
all_value = all_matrix[::, 2]

all_val_matrix = np.loadtxt(all_val, dtype=float, delimiter=',', skiprows=1)
val_step = all_val_matrix[::, 1]
val_value = all_val_matrix[::, 2]

# print (all_value)
# print (all_step)

T = all_step # np.array([6, 7, 8, 9, 10, 11, 12])
power = all_value # np.array([1.53E+03, 5.92E+02, 2.04E+02, 7.24E+01, 2.72E+01, 1.10E+01, 4.70E+00])

f_train = interp1d(T, power, kind='cubic')
seq = np.linspace(T.min(), T.max(), 4000)

a = val_step
b = val_value

f_val = interp1d(a, b, kind='cubic')

plt.plot(seq, f_train(seq), 'r', label='train')

seq = np.linspace(a.min(), a.max(), 4000)
plt.plot(seq, f_val(seq), 'b', label='test')
plt.legend()

plt.grid()

plt.xlabel('step')
plt.ylabel('rmse')

plt.savefig("com.pdf", dpi=300, format='pdf')

plt.show()
