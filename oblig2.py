
import numpy as np
from numpy.random import normal
from numpy.linalg import inv
from scipy.stats import chi2
import matplotlib.pyplot as plt

N = 10000 #Number of particles
n = 10 #Number of detectors 
k = 2

std = 30e-6
detector_pos = np.linspace(0.04, 0.4, 10)
H = np.zeros((n,2))
H[:,0] = 1
H[:,1] = detector_pos
V = np.identity(10)*std
H_trans = np.transpose(H)
Htrans_H = np.dot(H_trans,H)
Htrans_H_inv = inv(Htrans_H)
Htrans_H_inv_Htrans = np.dot(Htrans_H_inv,H_trans)
V_inv = inv(V)
H_trans_V_inv = np.dot(H_trans,V_inv)
cov_theta_hat= inv(np.dot(H_trans_V_inv,H))



inclination = 0.1

error = np.zeros(N)

part_pos_measured = np.zeros((N,10))
part_pos_true = inclination*detector_pos
error = np.zeros((N,10))
theta_hat = np.zeros((N,2))


for j in range(N):
	for i in range(n):
		error[j,i] = normal(0,std)
		part_pos_measured[j,i] = part_pos_true[i] + error[j,i]
	theta_hat[j] = np.dot(Htrans_H_inv_Htrans,part_pos_measured[j,:])

"""
res_b = theta_hat[:,0]
res_a = theta_hat[:,1] - 0.1


standard_dev_b = np.std(res_b)
standard_dev_a = np.std(res_a)

print standard_dev_a
print standard_dev_b

plt.hist(res_a, 50)
plt.show()

plt.hist(res_b,50)
plt.show()

res_a_norm = res_a/standard_dev_a
res_b_norm = res_b/standard_dev_b	 

plt.hist(res_a_norm,50)
plt.show()
plt.hist(res_b_norm,50)
plt.show()
"""

theta = np.array([0,0.1])
chi_square = np.zeros(N)

for i in range(N):
	theta_diff = theta_hat[i] - theta
	theta_diff_trans = np.transpose(theta_diff)
	inv_cov = inv(cov_theta_hat)
	theta_diff_trans_cov = np.dot(theta_diff_trans,inv_cov)
	chi_square[i] = np.dot(theta_diff_trans_cov,theta_diff)


p = np.zeros(N)

for i in range(N):
	p[i] = 1- chi2.cdf(chi_square[i],k)


plt.hist(p,50)
plt.show()
