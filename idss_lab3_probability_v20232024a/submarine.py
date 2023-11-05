import scipy.stats
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import skimage.io


np.random.seed(2023)

def make_pmf():
    blobs = [(5, 5, 2), (10, 3, 3), (8, 5, 2), (9, 7, 1), (6, 6, 1.0)]

    _pmf = np.zeros((16, 16))

    for i in range(16):
        for j in range(16):
            for x, y, r in blobs:
                ix = i - x
                jy = j - y
                _pmf[i, j] += (
                    scipy.stats.norm(0, r).pdf(np.sqrt((ix) ** 2 + (jy) ** 2))
                    * np.abs(np.cos(0.5 * (ix * jy + jy)))
                ) + 0.005

    _pmf = _pmf / np.sum(_pmf)
    return _pmf

submarine_pmf = make_pmf()

def make_other_pmfs():
    _pmf_1 = np.zeros((16, 16))
    a16 = np.arange(16)
    mx, my = np.meshgrid(a16, a16)

    _pmf_1 = scipy.stats.norm(0, 5.0).pdf(np.sqrt((mx - 5) ** 2 + (my - 7.5) ** 2))
    _pmf_1 = _pmf_1 / np.sum(_pmf_1)

    _pmf_2 = scipy.stats.norm(0, 2.0).pdf(np.sqrt((mx - 8) ** 2 + (my - 10) ** 2))

    _pmf_3 = np.cos(_pmf_1 * 2000)
    _pmf_3 = _pmf_3 / np.sum(_pmf_3)

    _pmf_4 = np.ones((16, 16)) / 256.0
    _pmf_5 = np.full((16, 16), 2.0)

    proposed_pmfs = [_pmf_1, _pmf_2, _pmf_3, _pmf_4, _pmf_5]
    return proposed_pmfs

proposed_pmfs = make_other_pmfs()

# def submarine_pmf(x,y):
#     assert x>=0 and y>=0 and x<_pmf.shape[0] and y<_pmf.shape[1], "Invalid coordinates"
#     return _pmf[x, y]

def show_pmf(p, title=""):
    back_map = skimage.io.imread("imgs/map.png")
    plt.figure()
    plt.imshow(back_map, extent=[0, 16, 0, 16])
    plt.imshow(p.T, vmin=0, alpha=0.75, extent=[0, 16, 16, 0], cmap="magma")
    plt.xticks(np.arange(16))
    plt.yticks(np.arange(16))
    plt.title(title)
    plt.gca().xaxis.tick_top()
    plt.grid(c="#000000", alpha=0.3)
    plt.xlabel("X")
    plt.ylabel("Y")

def sample_submarine(n, pmf=submarine_pmf):
    possibilities = pmf.ravel()
    coords = [[i, j] for i in range(16) for j in range(16)]
    ixs = np.arange(256)
    return np.array([coords[np.random.choice(ixs, p=possibilities)] for i in range(n)])

def search_submarine(x, y):
    true_location = 7.5, 10.5
    a16 = np.arange(16)
    my, mx = np.meshgrid(a16, a16)
    d_sub = np.sqrt((mx - true_location[0]) ** 2 + (my - true_location[1]) ** 2)
    dist = np.sqrt((true_location[0] - x) ** 2 + (true_location[1] - y) ** 2)
    search = np.exp(-dist ** 2 / 30)
    p_a = scipy.stats.norm(0, 0.1 + dist / 3).pdf(d_sub)
    p_b = np.ones((16, 16))
    p_a = p_a / np.sum(p_a)
    p_b = p_b / np.sum(p_b)
    return search * p_a + p_b

seq1 = sample_submarine(100)
seq2 = np.random.uniform(0, 15, (100, 2)).astype(np.int32)
seq3 = sample_submarine(100, pmf=proposed_pmfs[0])
seq4 = np.full((100, 2), 11)
inv_pmf = 1 - submarine_pmf
inv_pmf = inv_pmf / np.sum(inv_pmf)
seq5 = sample_submarine(100, pmf=inv_pmf)

submarine_samples = {
    "crater": seq4,
    "zeeman_shift": seq1,
    "oblov_1": seq2,
    "minority": seq5,
    "inviscid": seq3,
}