import numpy as np
import copy
import pprint as pp
import datetime as dt
import math
import time

L = np.array([[1,0],
              [1,0],
              [1,0],
              [1,1]])
Y = np.array([[0,1],
              [1,1],
              [0,1],
              [0,1]])
T = np.array([[1,1,1],
              [0,1,0],
              [0,1,0]])
N = np.array([[0,1],
              [0,1],
              [1,1],
              [1,0]])
P = np.array([[1,1],
              [1,1],
              [1,0]])
U = np.array([[1,0,1],
              [1,1,1]])
V = np.array([[1,0,0],
              [1,0,0],
              [1,1,1]])
F = np.array([[0,1,1],
              [1,1,0],
              [0,1,0]])
Z = np.array([[1,1,0],
              [0,1,0],
              [0,1,1]])
W = np.array([[1,0,0],
              [1,1,0],
              [0,1,1]])
I = np.array([[1,1,1,1,1]])
X = np.array([[0,1,0],
              [1,1,1],
              [0,1,0]])

# keys = [L, Y, T]
keys = [L, Y, T, P]
# keys = [L, Y, T, P, W]
# keys = [L, Y, T, P, W, Z]
# keys = [L, Y, T, P, W, Z, V]
# keys = [L, Y, T, P, W, Z, V, N]
# keys = [N, P, U]
# keys = [N, P, U, L]
# keys = [N, P, U, L, Z]
# keys = [N, P, U, L, Z, Y]
# keys = [N, P, U, L, Z, Y, T]
# keys = [N, P, U, L, Z, Y, T, W]
# keys = [L, V, P]
# keys = [L, V, P, Y]
# keys = [L, V, P, Y, N]
# keys = [L, V, P, Y, N, U]
# keys = [L, V, P, Y, N, U, Z]
# keys = [L, V, P, Y, N, U, Z, F]
# keys = [Y, P, U]
# keys = [Y, P, U, N]
# keys = [Y, P, U, N, V]
# keys = [Y, P, U, N, V, F]
# keys = [Y, P, U, N, V, F, W]
# keys = [Y, P, U, N, V, F, W, T]
# keys = [L, N, V]
# keys = [L, N, V, Z]
# keys = [L, N, V, Z, U]
# keys = [L, N, V, Z, U, T]
# keys = [L, N, V, Z, U, T, Y]
# keys = [L, N, V, Z, U, T, Y, W]
# keys = [P, U, F]
# keys = [P, U, F, Y]
# keys = [P, U, F, Y, T]
# keys = [P, U, F, Y, T, N]
# keys = [P, U, F, Y, T, N, L]
# keys = [P, U, F, Y, T, N, L, W]
# keys = [L, V, P]
# keys = [L, V, P, Z]
# keys = [L, V, P, Z, Y]
# keys = [L, V, P, Z, Y, W]
# keys = [L, V, P, Z, Y, W, N]
# keys = [L, V, P, Z, Y, W, N, F]
#base = np.zeros((5,5), dtype=int)
base = np.zeros((5, len(keys)), dtype=int)

status = []
resolve = []

def fitbase(k, b, offset):
    if offset[0] + k.shape[0] > b.shape[0] or offset[1] + k.shape[1] > b.shape[1]:
        return False, None
    k_ope = np.pad(k, [(offset[0], b.shape[0] - k.shape[0] - offset[0]),(offset[1], b.shape[1] - k.shape[1] - offset[1])], 'constant')
    return True, k_ope

def put(k, b):
    #if offset[0] + k.shape[0] > b.shape[0] or offset[1] + k.shape[1] > b.shape[1]:
    #    return False, None
    #k_ope = np.pad(k, [(offset[0], b.shape[0] - k.shape[0] - offset[0]),(offset[1], b.shape[1] - k.shape[1] - offset[1])], 'constant')
    res = b + k
    if res.max() > 1:
        return False, res
    else:
        return True, res

def remove(k, b):
    #if offset[0] + k.shape[0] > b.shape[0] or offset[1] + k.shape[1] > b.shape[1]:
    #    return False, None
    #k_ope = np.pad(k, [(offset[0], b.shape[0] - k.shape[0] - offset[0]),(offset[1], b.shape[1] - k.shape[1] - offset[1])], 'constant')
    res = b - k
    if res.min() < 0:
        return False, res
    else:
        return True, res

key_pat = {}
def key_variation(K, isFirst=False):
    K_ser = ''.join(map(str, K.flat))
    if K_ser in key_pat:
        return key_pat[K_ser]
    elif np.array_equal(K, I):
        key_pat[K_ser] = [K] if K.shape == (1, 5) else [np.rot90(K)]
        return key_pat[K_ser]
    else:
        if isFirst:
            # There is no need to consider about flipped shape for the first item.
            pat = [K, np.rot90(K), np.rot90(K, 2), np.rot90(K, 3)]
        else:
            pat = [K, np.rot90(K), np.rot90(K, 2), np.rot90(K, 3), np.flipud(K), np.rot90(np.flipud(K)), np.rot90(np.flipud(K), 2), np.rot90(np.flipud(K), 3)]
        pat_uni = []
        for p in pat:
            if not any(map(lambda k: np.array_equal(k, p), pat_uni)):
                pat_uni.append(p)
        key_pat[K_ser] = pat_uni
        return pat_uni

def find_closed_areas(base):
    visited = np.zeros(base.shape, dtype=int)
    closed_areas_size = []
    for row in range(0, base.shape[0]):
        for col in range(0, base.shape[1]):
            if base[row, col] == 0 and visited[row, col] == 0:
                closed_area = []
                explore((row, col), base, visited, closed_area)
                closed_areas_size.append(len(closed_area))
    return closed_areas_size

closed_areas = []
def find_any_lower5_closed_areas(base):
    visited = np.zeros(base.shape, dtype=int)
    global closed_areas
    closed_areas = []
    for row in range(0, base.shape[0]):
        for col in range(0, base.shape[1]):
            if base[row, col] == 0 and visited[row, col] == 0:
                closed_area = []
                explore((row, col), base, visited, closed_area)
                # if len(closed_area) < 5: return True
                if len(closed_area) % 5 != 0: return True
                closed_areas.append(closed_area)
    # print(closed_areas)
    return False

search_queue = []
def explore(index, base, visited, closed_area):
    if base[index] == 1 or visited[index] == 1:
        visited[index] = 1
        return
    else:
        closed_area.append(index)
        visited[index] = 1
        # see right
        right_next = (index[0], index[1] + 1)
        if right_next[1] < base.shape[1] and base[right_next] == 0 and visited[right_next] == 0 and right_next not in search_queue:
            search_queue.append(right_next)
        # see below
        below_next = (index[0] + 1, index[1])
        if below_next[0] < base.shape[0] and base[below_next] == 0 and visited[below_next] == 0 and below_next not in search_queue:
            search_queue.append(below_next)
        # see left
        left_next = (index[0], index[1] - 1)
        if left_next[1] >= 0 and base[left_next] == 0 and visited[left_next] == 0 and left_next not in search_queue:
            search_queue.append(left_next)
        # see above
        above_next = (index[0] - 1, index[1])
        if above_next[0] >= 0 and base[above_next] == 0 and visited[above_next] == 0 and above_next not in search_queue:
            search_queue.append(above_next)
        # go next
        if len(search_queue) > 0:
            explore(search_queue.pop(0), base, visited, closed_area)

def has_lower5_closed_area(base):
    # closed_areas = find_closed_areas(base)
    # return min(closed_areas) < 5
    return find_any_lower5_closed_areas(base)

def solve(keys, part, b):
    # if K is the last item, just check the empty area
    if len(closed_areas) == 1 and len(closed_areas[0]) == 5:
        empty_area = np.zeros(b.shape, dtype=int
        )
        for i in closed_areas[0]:
            empty_area[i] = 1
        rows = list(c[0] for c in closed_areas[0])
        cols = list(c[1] for c in closed_areas[0])
        empty_area = empty_area[min(rows): max(rows) + 1, min(cols): max(cols) + 1]
        if keys[part].shape != empty_area.shape and np.rot90(keys[part]).shape != empty_area.shape: return
        for K in key_variation(keys[part], False):
            # print(K)
            # print(empty_area)
            if K.shape == empty_area.shape and np.array_equal(K, empty_area):
                fit, K_fit = fitbase(K, b, (min(rows), min(cols)))
                ret, nextbase = put(K_fit, b)
                status.append(K_fit)
                pretty_print(status)
                time.sleep(0.5)
                print("solved!!")
                resolve.append(copy.deepcopy(status))
                status.pop()
                return
        else:
            return
        
    # if K is the first item, there is no need to consider about flipped shape.
    for K in key_variation(keys[part], part==0):
    # for K in key_variation(keys[part]):
        # If K is the first item, only quarter area should be considered.
        row_range = range(0, b.shape[0] - K.shape[0] + 1) if part > 0 else range(0, min(b.shape[0] - K.shape[0] + 1, math.ceil(b.shape[0] / 2)))
        col_range = range(0, b.shape[1] - K.shape[1] + 1) if part > 0 else range(0, min(b.shape[1] - K.shape[1] + 1, math.ceil(b.shape[1] / 2)))
        # if part == 0: print(b.shape, row_range, col_range)
        for row in row_range:
            for col in col_range:
                fit, K_fit = fitbase(K, b, (row, col))
                # if part == 0: print(K_fit)
                if fit:
                    ret, nextbase = put(K_fit, b)
                    print(nextbase)
                    # print(ret, nextbase)
                    if ret:
                        status.append(K_fit)
                        pretty_print(status)
                        time.sleep(0.5)
                        if part == len(keys) - 1:
                            print("solved!!")
                            resolve.append(copy.deepcopy(status))
                        else:
                            # if there is no lower-five closed area, then go next
                            if not has_lower5_closed_area(nextbase):
                                # print('Well done, go ahead with next key!')
                                solve(keys, part + 1, nextbase)
                            else:
                                print('Ooops, it has incapable closed area.\n')
                        status.pop()
        # else:
            # pretty_print(status)
            # print('Oh, there is no space which can accept next key..')
            # pretty_print([K], part + 1)
            # print('Go next key variation.\n')
    # else:
        # print('Go next key.\n')

def pretty_print(statuses, start_symbol=1):
    base = np.zeros(statuses[0].shape, dtype=int)
    ret = '|{}|\n'.format('-' * (base.shape[1] * 2 + 1))
    # print(statuses)
    # print(base)
    for i, st in enumerate(statuses):
        base += st * (i + start_symbol)
    for row in range(0, base.shape[0]):
        ret += '| '
        for col in range(0, base.shape[1]):
            if base[row, col] == 0:
                ret += ' '
            else:
                ret += str(base[row, col])
            ret += ' '
        ret += '|\n'
    ret += '|{}|'.format('-' * (base.shape[1] * 2 + 1))
    print(ret)


str_keys = ''
all_keys = np.zeros((max(map(lambda k: k.shape[0], keys)), 1), dtype=int)
for i, k in enumerate(keys):
    # all_keys = np.concatenate(all_keys, k * (i + 1))
    # print(k * (i +1))
    pretty_print([k], i + 1)

dt_start = dt.datetime.now()
print(dt_start)
solve(keys, 0, base)
dt_end = dt.datetime.now()
print(dt_end)
print(dt_end - dt_start)
#print(resolve)
#pp.pprint(resolve)

for solv in resolve:
    solv_out = copy.deepcopy(base)
    for i in range(0, len(solv)):
        solv_out += solv[i] * (i + 1)
    print(solv_out)

