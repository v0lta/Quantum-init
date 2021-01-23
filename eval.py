import pickle
import numpy as np
import matplotlib.pyplot as plt

res = pickle.load(open("stats.pickle", "rb"))

pseudo_acc = []
pseudo_loss = []
qntrnd_acc = []
qntrnd_loss = []
for exp in res:
    if exp['args'].pseudo_init:
        pseudo_acc.append(exp['test_acc_lst'])
        pseudo_loss.append(exp['test_loss_lst'])
    else:
        qntrnd_acc.append(exp['test_acc_lst'])
        qntrnd_loss.append(exp['test_loss_lst'])

pseudo_acc = np.array(pseudo_acc)
pseudo_loss = np.array(pseudo_loss)


acc_mean = np.mean(pseudo_acc, axis=0)
acc_std = np.std(pseudo_acc, axis=0)
x = np.array(range(len(acc_mean)))
plt.errorbar(x, acc_mean, yerr=acc_std)
plt.ylabel('acc')
plt.xlabel('epochs')
plt.show()

print('plots saved.')