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

qntrand_acc = np.array(qntrnd_acc)
qntrnd_loss = np.array(qntrnd_loss)

pseudo_acc_mean = np.mean(pseudo_acc, axis=0)
pseudo_acc_std = np.std(pseudo_acc, axis=0)
x = np.array(range(len(pseudo_acc_mean)))
qntrnd_acc_mean = np.mean(qntrnd_acc, axis=0)
qntrnd_acc_std = np.std(qntrnd_acc, axis=0)

plt.errorbar(x, pseudo_acc_mean, yerr=pseudo_acc_std, label='pseudo')
plt.errorbar(x, qntrnd_acc_mean, yerr=qntrnd_acc_std, label='quantum')
plt.ylabel('acc')
plt.xlabel('epochs')
plt.legend()
plt.show()
plt.savefig('random_eval.png')
print('plots saved.')
