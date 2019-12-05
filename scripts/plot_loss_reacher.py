import numpy as np
import matplotlib.pyplot as plt

train_chars_range = np.concatenate((np.array([1]),np.arange(2,19,step=2)))
rdn_loss = np.loadtxt('rdn_loss.txt')
rffn_loss = np.loadtxt('rffn_loss.txt')

# statistics for reacher delan
rdn_mean = np.mean(rdn_loss, axis=1)
rdn_sigma = np.std(rdn_loss, axis=1)
rdn_upper_95conf = rdn_mean + 2 * rdn_sigma
rdn_lower_95conf = rdn_mean - 2 * rdn_sigma

# statistics for reacher ff-nn
rffn_mean = np.mean(rffn_loss, axis=1)
rffn_sigma = np.std(rffn_loss, axis=1)
rffn_upper_95conf = rffn_mean + 2 * rffn_sigma
rffn_lower_95conf = rffn_mean - 2 * rffn_sigma

# generate test error plot
plt.plot(train_chars_range, np.mean(rdn_loss, axis=1), c='red',label='Reacher DeLaN')
plt.plot(train_chars_range, np.mean(rffn_loss, axis=1), c='blue',label='Reacher FF-NN')
plt.fill_between(train_chars_range,rdn_lower_95conf,rdn_upper_95conf,where=rdn_upper_95conf >= rdn_lower_95conf, facecolor='red', interpolate=True, alpha=0.5)
plt.fill_between(train_chars_range,rffn_lower_95conf,rffn_upper_95conf,where=rffn_upper_95conf >= rffn_lower_95conf, facecolor='blue', interpolate=True, alpha=0.5)
# plt.yscale('log')
plt.xticks(train_chars_range)
# plt.yticks(np.array([10e-6,10e-5,10e-4,10e-3,10e-2,10e-1,10]))

plt.ylabel('MSE')
plt.xlabel('Unique Training Characters')
plt.legend()
plt.title('Reacher DeLaN vs FF-NN Test Error')
plt.savefig('delan_vs_ff_test_error.png')
plt.show()
plt.close()
