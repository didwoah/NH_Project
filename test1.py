from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

class Data:
    def __init__(self, mts, tot, byn_sll):
        self.mts = mts
        self.tot = tot
        self.byn_sll = byn_sll

    def __str__(self):
        return 'mts : {}, tot : {}, byn_sll : {}'.format(self.mts, self.tot, self.byn_sll)

def complexToPolar(i):
    return np.arctan(i.imag/i.real), abs(i)
def polarToComplex(theta, r):
    x = r/((np.tan(theta)**2+1)**0.5)
    y = (r**2-x**2)**0.5
    return x+y*1j


cus_ifo = pd.read_csv("./cus_ifo.csv", header=0, usecols=['act_no', 'mts_mm_access_type', 'tot_ivs_te_sgm_cd'])
cus_itg = pd.read_csv('./cus_itg_sct_bnc.csv', header=0, usecols=['act_no', 'itg_byn_cns_qty', 'itg_sll_cns_qty'])

cus_ifo['mts_mm_access_type'] = cus_ifo['mts_mm_access_type'].astype('string')
cus_ifo['act_no'] = cus_ifo['act_no'].astype('string')
cus_itg['act_no'] = cus_itg['act_no'].astype('string')

data_mts = {}
data_tot = {}
for row_i, row in cus_ifo.iterrows():
    if row_i > 0:
        data_mts[row['act_no']] = row['mts_mm_access_type'].count('1')/6
        tmp = row['tot_ivs_te_sgm_cd']
        if tmp == 99:
            tmp = 1
        data_tot[row['act_no']] = tmp/6

data_byn_sll = {}
for row_i, row in cus_itg.iterrows():
    if row_i > 0:
        byn = row['itg_byn_cns_qty']
        sll = row['itg_sll_cns_qty']
        if row['act_no'] in data_byn_sll:
            tmp = data_byn_sll[row['act_no']]
            data_byn_sll[row['act_no']] = tmp+byn+sll
        else:
            data_byn_sll[row['act_no']] = byn+sll
        

data_set = {}
for key in data_byn_sll:
    if key in data_mts and data_byn_sll[key] <= 10000:
        data_set[key] = Data(data_mts[key], data_tot[key], data_byn_sll[key]/10000)


# print(data_mts)
# print(data_tot)
# print(data_byn_sll)
# print(data_set)
# for i, key in enumerate(data_set):
#     if i < 5:
#         print('{} : {}'.format(key, data_set[key]))
# print(len(data_set))

base1 = 1j
base2 = -(3**0.5)*0.5+0.5j
base3 = (3**0.5)*0.5+0.5j

d1 = []
d2 = []
d3 = []
for data in data_set.values():
    d1.append(data.mts)
    d2.append(data.tot)
    d3.append(data.byn_sll)
bd = np.array([d1, d2, d3])
bd = bd.transpose()
dd = np.array([x[0]*base1+x[1]*base2+x[2]*base3 for x in bd])
dd1 = np.array([np.arctan(x.imag/(x.real+0.0001)) for x in dd])
dd2 = np.array([abs(x) for x in dd])

X = np.array(bd)

kmeans = KMeans(n_clusters=2)
kmeans.fit(X)

print(kmeans.cluster_centers_)

fig = plt.figure()
ax = fig.add_subplot(111, projection='polar')
c = ax.scatter(dd1, dd2, s=3)
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='polar')
ax.scatter(dd1, dd2, c=kmeans.labels_, cmap="rainbow", s=2)

c1 = kmeans.cluster_centers_[0][0]*base1+kmeans.cluster_centers_[0][1]*base2+kmeans.cluster_centers_[0][2]*base3
print(kmeans.cluster_centers_[0][0], kmeans.cluster_centers_[0][1], kmeans.cluster_centers_[0][2])

c2 = kmeans.cluster_centers_[1][0]*base1+kmeans.cluster_centers_[1][1]*base2+kmeans.cluster_centers_[1][2]*base3
print(kmeans.cluster_centers_[1][0], kmeans.cluster_centers_[1][1], kmeans.cluster_centers_[1][2])

ax.scatter(np.arctan(c1.imag/c1.real), abs(c1), c='blue', s=20)
ax.scatter(np.arctan(c2.imag/c2.real), abs(c2), c='red', s=20)
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(d1, d2, d3, c=kmeans.labels_, cmap="rainbow", s=2)

print(kmeans.cluster_centers_[0][0], kmeans.cluster_centers_[0][1], kmeans.cluster_centers_[0][2])
print(kmeans.cluster_centers_[1][0], kmeans.cluster_centers_[1][1], kmeans.cluster_centers_[1][2])

ax.scatter(kmeans.cluster_centers_[0][0], kmeans.cluster_centers_[0][1], kmeans.cluster_centers_[0][2], c='blue', s=20)
ax.scatter(kmeans.cluster_centers_[1][0], kmeans.cluster_centers_[1][1], kmeans.cluster_centers_[1][2], c='red', s=20)
plt.show()

n_clusters = range(1, 10)
kmeans = [KMeans(n_clusters=i) for i in n_clusters]

score = [kmeans[i].fit(X).inertia_ for i in range(len(kmeans))]

plt.plot(n_clusters, score)
plt.xlabel("Number of Clusters")
plt.ylabel("Score")
plt.title("Elbow Curve")
plt.show()