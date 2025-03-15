from math import inf

import numpy as np
from numba import njit, b1, i4, f4, deferred_type, optional
from numba.core.types import Array, Record
from numba.experimental import jitclass

kdtree2_result = np.dtype([('dis', 'f4'), ('idx', 'i4')])

# A heap-based priority queue lets one efficiently implement the following
# operations, each in log(N) time, as opposed to linear time.
#
# 1)  add a datum (push a datum onto the queue, increasing its length)
# 2)  return the priority value of the maximum priority element
# 3)  pop-off (and delete) the element with the maximum priority, decreasing
#     the size of the queue.
# 4)  replace the datum with the maximum priority with a supplied datum
#     (of either higher or lower priority), maintaining the size of the
#     queue.
#
#
# In the k-d tree case, the 'priority' is the square distance of a point in
# the data set to a reference point.   The goal is to keep the smallest M
# distances to a reference point.  The tree algorithm searches terminal
# nodes to decide whether to add points under consideration.
#
# A priority queue is useful here because it lets one quickly return the
# largest distance currently existing in the list.  If a new candidate
# distance is smaller than this, then the new candidate ought to replace
# the old candidate.  In priority queue terms, this means removing the
# highest priority element, and inserting the new one.
#
# Algorithms based on Cormen, Leiserson, Rivest, _Introduction
# to Algorithms_, 1990, with further optimization by the author.
#
# Originally informed by a C implementation by Sriranga Veeraraghavan.
#
# This module is not written in the most clear way, but is implemented such
# for speed, as it its operations will be called many times during searches
# of large numbers of neighbors.


spec = {
    'elements': Array(Record([
        ('dis', {'type': f4, 'offset': 0}), ('idx', {'type': i4, 'offset': 4})
    ], 8, False), 1, 'C'),
    'heap_size': i4
}


@jitclass(spec=spec)
class PQ(object):
    def __init__(self):
        self.heap_size = 0


@njit(nogil=True, error_model='numpy', boundscheck=True)
def pq_create(results_in) -> PQ:
    n_alloc = np.shape(results_in)[0]
    assert n_alloc >= 1, 'PQ_CREATE: error, input arrays must be allocated.'
    # res = PQ(results_in, 0)
    res = PQ()
    res.elements = results_in
    res.heap_size = 0
    return res


# def heapify(a, i_in):
#     i = i_in
#     while True:
#         l = 2 * i  # left(i)
#         r = l + 1  # right(i)
#
#         if l > a.heap_size:
#             break
#         else:
#             pri_i = a.elements['dis'][i]
#             pri_l = a.elements['dis'][l]
#             if pri_l > pri_i:
#                 largest = l
#                 pri_largest = pri_l
#             else:
#                 largest = i
#                 pri_largest = pri_i
#
#             if r <= a.heap_size:
#                 pri_r = a.elements['dis'][r]
#                 if pri_r > pri_largest:
#                     largest = r
#
#         if largest != i:
#             temp = a.elements[i]
#             a.elements[i] = a.elements[largest]
#             a.elements[largest] = temp
#
#             i = largest
#             continue
#         else:
#             return


# def pq_max(a):
#     if a.heap_size > 0:
#         return a.elements[0]
#     else:
#         raise ValueError('PQ_MAX: ERROR, heap_size < 1')


# def pq_maxpri(a):
#     if a.heap_size > 0:
#         return a.elements['dis'][0]
#     else:
#         raise ValueError('PQ_MAX_PRI: ERROR, heapsize < 1')


# def pq_extract_max(a):
#     if a.heap_size >= 1:
#         e = a.elements[0]
#         a.elements[0] = a.elements(a.heap_size)
#         a.heap_size = a.heap_size - 1
#         heapify(a, 1)
#         return e
#     else:
#         raise ValueError('PQ_EXTRACT_MAX: error, attempted to pop non-positive PQ')


# 结果所要求的点数未满时，以二叉树的形式保存结果，父节点的距离始终大于子节点的距离，并返回所保存的最大距离值
@njit(nogil=True, error_model='numpy', boundscheck=True)
def pq_insert(a, dis, idx):
    a.heap_size += 1
    i = a.heap_size

    while i > 1:
        # 父节点的位置索引
        is_parent = int(i / 2)
        parent_dis = a.elements['dis'][is_parent - 1]
        if dis > parent_dis:
            # move what was in i's parent into i.
            a.elements['dis'][i - 1] = parent_dis
            a.elements['idx'][i - 1] = a.elements['idx'][is_parent - 1]
            i = is_parent
        else:
            break

    # insert the element at the determined position
    a.elements['dis'][i - 1] = dis
    a.elements['idx'][i - 1] = idx

    return a.elements['dis'][0]


# def pq_adjust_heap(a, i):
#     e = a.elements[i]
#
#     parent = i
#     child = 2 * i
#     n = a.heap_size
#
#     while child <= n:
#         if child < n:
#             if a.elements['dis'][child] < a.elements['dis'][child + 1]:
#                 child += 1
#
#         pri_child = a.elements['dis'][child]
#         if e['dis'] >= pri_child:
#             break
#         else:
#             # move child into parent.
#             a.elements[parent] = a.elements[child]
#             parent = child
#             child = 2 * parent
#
#     a.elements[parent] = e


# 结果所要求的点数已满时，通过二叉树搜索距离最大的点并替换掉，同时要更新树结构
@njit(nogil=True, error_model='numpy', boundscheck=True)
def pq_replace_max(a, dis, idx):
    n = a.heap_size
    if n >= 1:
        parent = 1
        child = 2

        while child <= n:
            # pre_child取每一个递归层级的距离较大值
            pri_child = a.elements['dis'][child - 1]

            if child < n:
                pri_child_p1 = a.elements['dis'][child]
                if pri_child < pri_child_p1:
                    child += 1
                    pri_child = pri_child_p1

            # 如果插入点的距离比该层级的距离较大值大，则前面的递归替换过程已完成，跳出循环
            if dis >= pri_child:
                break
            # 否则向前递归替换 覆盖掉上一层级的点
            else:
                a.elements['dis'][parent - 1] = a.elements['dis'][child - 1]
                a.elements['idx'][parent - 1] = a.elements['idx'][child - 1]
                parent = child
                child = 2 * parent

        a.elements['dis'][parent - 1] = dis
        a.elements['idx'][parent - 1] = idx

        # 返回新的最大距离
        return a.elements['dis'][0]
    else:
        # todo??? 不会有这个情况吧
        a.elements['dis'][0] = dis
        a.elements['idx'][0] = idx
        return dis


# def pq_delete(a, i):
#     if (i < 1) or (i > a.heap_size):
#         raise ValueError('PQ_DELETE: error, attempt to remove out of bounds element.')
#
#     a.elements[i] = a.elements[a.heap_size]
#     a.heap_size = a.heap_size - 1
#
#     heapify(a, i)


bucket_size = 12

interval = np.dtype([('lower', 'f4'), ('upper', 'f4')])

node_type = deferred_type()
spec = {
    'cut_dim': i4,
    'cut_val': f4,
    'cut_val_left': f4,
    'cut_val_right': f4,
    'l': i4,
    'u': i4,
    'left': optional(node_type),
    'right': optional(node_type),
    'box': Array(Record([
        ('lower', {'type': f4, 'offset': 0}), ('upper', {'type': f4, 'offset': 4})
    ], 8, False), 1, 'C'),
}


@jitclass(spec=spec)
class TreeNode(object):
    def __init__(self):
        self.left = None
        self.right = None


node_type.define(TreeNode.class_type.instance_type)

spec = {
    'dimension': i4,
    'n': i4,
    'the_data': f4[:, :],
    'ind': i4[:],
    'sort': b1,
    'rearrange': b1,
    'rearranged_data': f4[:, :],
    'root': optional(node_type),
}


@jitclass(spec=spec)
class KDTree2(object):
    def __init__(self):
        self.dimension = 0
        self.n = 0
        self.sort = False
        self.rearrange = False
        self.root = None


spec = {
    'dimension': i4,
    'nn': i4,
    'n_found': i4,
    'ball_size': f4,
    'center_idx': i4,
    'cor_rel_time': i4,
    'n_alloc': i4,
    'rearrange': b1,
    'overflow': b1,
    'qv': f4[:],
    'results': Array(Record([
        ('dis', {'type': f4, 'offset': 0}), ('idx', {'type': i4, 'offset': 4})
    ], 8, False), 1, 'C'),
    'pq': optional(PQ.class_type.instance_type),
    'data': f4[:, :],
    'ind': i4[:],
}


@jitclass(spec=spec)
class TreeSearchRecord(object):
    def __init__(self):
        self.center_idx = 999
        self.cor_rel_time = 9999
        # self.root = None


@njit(nogil=True, error_model='numpy', boundscheck=True)
def kdtree2_create(input_data, dim=None, sort=None, rearrange=None):
    mr = KDTree2()
    mr.the_data = input_data
    # pointer assignment

    if dim is None:
        mr.dimension = np.shape(input_data)[0]
    else:
        mr.dimension = dim

    mr.n = np.shape(input_data)[1]

    assert mr.dimension <= mr.n, (
        'KD_TREE_TRANS: likely user error.'
        'KD_TREE_TRANS: note, that new format is data[:D,:N]'
        'KD_TREE_TRANS: with usually N >> D.   If N =approx= D, then a k-d tree'
        'KD_TREE_TRANS: is not an appropriate data structure.'
    )

    build_tree(mr)

    if sort is None:
        mr.sort = False
    else:
        mr.sort = sort

    if rearrange is None:
        mr.rearrange = True
    else:
        mr.rearrange = rearrange

    if mr.rearrange:
        mr.rearranged_data = np.empty((mr.dimension, mr.n), 'f4')
        for i in range(mr.n):
            mr.rearranged_data[:, i] = mr.the_data[:, mr.ind[i]]
    # else:
    #     mr.rearranged_data = None

    return mr


@njit(nogil=True, error_model='numpy', boundscheck=True)
def build_tree(tp: KDTree2):
    dummy = None
    # dummy = TreeNode()

    # tp.ind = np.arange(tp.n)
    tp.ind = np.empty(tp.n, 'i4')
    for j in range(tp.n):
        tp.ind[j] = j

    # tp.root = build_tree_for_range(tp, 1, tp.n, dummy)
    # todo
    tp.root = build_tree_for_range(tp, 0, tp.n - 1, dummy)


# 递归返回节点
@njit(nogil=True, error_model='numpy', boundscheck=True)
def build_tree_for_range(tp: KDTree2, l, u, parent):
    dimension = tp.dimension
    res = TreeNode()
    res.box = np.empty(dimension, dtype=interval)

    # 首先，计算与该节点关联的所有点的近似边界框
    if u < l:
        return None

    if (u - l) <= bucket_size:
        # 总是计算终端节点的真实边界框
        for i in range(dimension):
            res.box['lower'][i], res.box['upper'][i] = spread_in_coordinate(tp, i, l, u)

        res.cut_dim = 0
        res.cut_val = 0.0
        res.l = l
        res.u = u
        res.left = None
        res.right = None
    else:
        # 修改近似边界框。这将是对真实边界框的高估，因为我们只是重新计算父级分割的维度的边界框。
        # 进行真正的边界框计算会显着增加构建树所需的时间，并且通常只有很小的差异。
        # 此框不用于搜索，而仅用于决定要分割的坐标。
        for i in range(dimension):
            # 父节点所分割的维度的边框需要重新计算，其他维度继承自父节点
            # 这样计算得到的边界框比真实的边界框要大些
            # recompute = True
            # if parent is not None:
            #     if i != parent.cut_dim:
            #         recompute = False
            #
            # if recompute:
            #     res.box['lower'][i], res.box['upper'][i] = spread_in_coordinate(tp, i, l, u)
            # else:
            #     res.box['upper'][i] = parent.box['upper'][i]
            #     res.box['lower'][i] = parent.box['lower'][i]

            if parent is not None and i != parent.cut_dim:
                res.box['upper'][i] = parent.box['upper'][i]
                res.box['lower'][i] = parent.box['lower'][i]
            else:
                res.box['lower'][i], res.box['upper'][i] = spread_in_coordinate(tp, i, l, u)

        # 依据近似的边界框选择一个好的分割维度及分割点
        c = np.argmax(res.box['upper'][:dimension] - res.box['lower'][:dimension], axis=0).item()
        tmp = np.empty(u - l + 1, 'f4')
        for k in range(l, u + 1):
            tmp[k - l] = tp.the_data[c, tp.ind[k]]
        average = np.mean(tmp)
        # average = np.mean(tp.the_data[c, tp.ind[l:u + 1]])

        res.cut_val = average
        # 按照分割点进行分割，<= m的索引值放在左侧，> m的索引值放在右侧
        m = select_on_coordinate_value(tp.the_data, tp.ind, c, average, l, u)

        # moves indexes around
        res.cut_dim = c
        res.l = l
        res.u = u

        res.left = build_tree_for_range(tp, l, m, res)
        res.right = build_tree_for_range(tp, m + 1, u, res)

        # 左枝值小，右枝值大
        if res.right is None:
            for i in range(dimension):
                res.box['upper'][i] = res.left.box['upper'][i]
                res.box['lower'][i] = res.left.box['lower'][i]
            res.cut_val_left = res.left.box['upper'][c]
            res.cut_val = res.cut_val_left
        elif res.left is None:
            for i in range(dimension):
                res.box['upper'][i] = res.right.box['upper'][i]
                res.box['upper'][i] = res.right.box['upper'][i]
            res.cut_val_right = res.right.box['lower'][c]
            res.cut_val = res.cut_val_right
        else:
            res.cut_val_right = res.right.box['lower'][c]
            res.cut_val_left = res.left.box['upper'][c]
            res.cut_val = (res.cut_val_left + res.cut_val_right) / 2

            # 现在为自己重新制作真正的边界框。由于我们采用树结构的并集，这比对所有点进行详尽搜索要快得多
            for i in range(dimension):
                res.box['upper'][i] = np.maximum(res.left.box['upper'][i], res.right.box['upper'][i])
                res.box['lower'][i] = np.minimum(res.left.box['lower'][i], res.right.box['lower'][i])

    return res


@njit(nogil=True, error_model='numpy', boundscheck=True)
def select_on_coordinate_value(v, ind, c, alpha, li, ui):
    # Algorithm (matt kennel).

    # 将列表视为具有三个部分：在左侧为 <= alpha 的点。右边是 > alpha 的点，中间是当前未知的点
    # 该算法是扫描未知点，从左侧开始，并交换它们，以便将它们添加到左侧堆栈或右侧堆栈，视情况而定
    # 当未知堆栈为空时，算法结束

    # 已知 <= alpha 的点在 [l,lb-1] 中
    # 已知 > alpha 的点在 [rb+1,u] 中
    # 因此，我们酌情在 lb 或 rb 中添加新点。当 lb=rb 时，我们就完成了。
    # 返回最后一个 <= alpha 的点的位置

    lb = li
    rb = ui
    while lb < rb:
        if v[c, ind[lb]] <= alpha:
            lb += 1
        else:
            ind[lb], ind[rb] = ind[rb], ind[lb]
            rb -= 1

    if ui == lb:
        lb -= 1
        rb = lb

    if v[c, ind[lb]] <= alpha:
        res = lb
    else:
        res = lb - 1

    return res


# def select_on_coordinate(v, ind, c, k, li, ui):
#     l = li
#     u = ui
#     while l < u:
#         t = ind[l]
#         m = l
#         for i in range(l + 1, u):
#             if v[c, ind[i]] < v[c, t]:
#                 m = m + 1
#                 s = ind[m]
#                 ind[m] = ind[i]
#                 ind[i] = s
#
#         s = ind[l]
#         ind[l] = ind[m]
#         ind[m] = s
#         if m <= k:
#             l = m + 1
#         if m >= k:
#             u = m - 1

# todo
@njit(nogil=True, error_model='numpy', boundscheck=True)
def spread_in_coordinate(tp: KDTree2, c, l, u):
    v = tp.the_data[:, :]
    ind = tp.ind[:]

    s_min = v[c, ind[l]]
    s_max = s_min

    for i in range(l + 1, u + 1):
        if v[c, ind[i]] < s_min:
            s_min = v[c, ind[i]]
        elif v[c, ind[i]] > s_max:
            s_max = v[c, ind[i]]

    # u_local = u
    # for i in range(l + 2, u_local, 2):
    #     l_min = v[c, ind[i - 1]]
    #     l_max = v[c, ind[i]]
    #     if l_min > l_max:
    #         l_min, l_max = l_max, l_min
    #     if s_min > l_min:
    #         s_min = l_min
    #     if s_max < l_max:
    #         s_max = l_max
    #
    # if i == u_local + 1:
    #     last = v[c, ind[u_local]]
    #     if s_min > last:
    #         s_min = last
    #     if s_max < last:
    #         s_max = last

    # s_min = np.amin(v[c, ind[l:u + 1]])
    # s_max = np.amax(v[c, ind[l:u + 1]])

    return s_min, s_max


@njit(nogil=True, error_model='numpy', boundscheck=True)
def kdtree2_n_nearest(tp: KDTree2, qv, nn, results):
    sr = TreeSearchRecord()

    sr.ball_size = inf
    sr.qv = qv
    sr.nn = nn
    sr.n_found = 0
    sr.center_idx = -1
    sr.cor_rel_time = 0
    sr.overflow = False
    sr.results = results
    sr.n_alloc = nn  # will be checked
    sr.ind = tp.ind
    sr.rearrange = tp.rearrange

    if tp.rearrange:
        sr.data = tp.rearranged_data
    else:
        sr.data = tp.the_data
    sr.dimension = tp.dimension

    validate_query_storage(nn, sr)
    sr.pq = pq_create(results)

    search(tp.root, sr)

    if tp.sort:
        kdtree2_sort_results(nn, results)


# def kdtree2_n_nearest_around_point(tp, idx_in, cor_rel_time, nn, results):
#     sr.qv = np.empty(tp.dimension)
#     sr.qv = tp.the_data[:, idx_in]  # copy the vector
#     sr.ball_size = huge(1.0)  # the largest real number
#     sr.center_idx = idx_in
#     sr.cor_rel_time = cor_rel_time
#
#     sr.nn = nn
#     sr.n_found = 0
#
#     sr.dimension = tp.dimension
#     sr.n_alloc = nn
#
#     sr.results = results
#
#     sr.ind = tp.ind
#     sr.rearrange = tp.rearrange
#
#     if sr.rearrange:
#         sr.data = tp.rearranged_data
#     else:
#         sr.data = tp.the_data
#
#     validate_query_storage(nn)
#     sr.pq = pq_create(results)
#
#     search(tp.root)
#
#     if tp.sort:
#         kdtree2_sort_results(nn, results)


# def kdtree2_r_nearest(tp, qv, r2, n_alloc, results):
#     sr.qv = qv
#     sr.ball_size = r2
#     sr.nn = 0  # flag for fixed ball search
#     sr.n_found = 0
#     sr.center_idx = -1
#     sr.cor_rel_time = 0
#     sr.results = results
#
#     validate_query_storage(n_alloc)
#
#     sr.n_alloc = n_alloc
#     sr.overflow = False
#     sr.ind = tp.ind
#     sr.rearrange = tp.rearrange
#
#     if tp.rearrange:
#         sr.data = tp.rearranged_data
#     else:
#         sr.data = tp.the_data
#
#     sr.dimension = tp.dimension
#
#     search(tp.root)
#     n_found = sr.n_found
#     if tp.sort:
#         kdtree2_sort_results(n_found, results)
#
#     if sr.overflow:
#         print('KD_TREE_TRANS: warning# return from kdtree2_r_nearest found more neighbors')
#         print('KD_TREE_TRANS: than storage was provided for.  Answer is NOT smallest ball')
#         print('KD_TREE_TRANS: with that number of neighbors#  I.e. it is wrong.')
#
#     return n_found


# def kdtree2_r_nearest_around_point(tp, idx_in, cor_rel_time, r2, n_alloc, results):
#     sr.qv = np.empty(tp.dimension)
#     sr.qv = tp.the_data[:, idx_in]  # copy the vector
#     sr.ball_size = r2
#     sr.nn = 0  # flag for fixed r search
#     sr.n_found = 0
#     sr.center_idx = idx_in
#     sr.cor_rel_time = cor_rel_time
#
#     sr.results = results
#
#     sr.n_alloc = n_alloc
#     sr.overflow = False
#
#     validate_query_storage(n_alloc)
#
#     sr.ind = tp.ind
#     sr.rearrange = tp.rearrange
#
#     if tp.rearrange:
#         sr.data = tp.rearranged_data
#     else:
#         sr.data = tp.the_data
#
#     sr.rearrange = tp.rearrange
#     sr.dimension = tp.dimension
#
#     search(tp.root)
#     n_found = sr.n_found
#     if tp.sort:
#         kdtree2_sort_results(n_found, results)
#
#     if sr.overflow:
#         print('KD_TREE_TRANS: warning# return from kdtree2_r_nearest found more neighbors')
#         print('KD_TREE_TRANS: than storage was provided for.  Answer is NOT smallest ball')
#         print('KD_TREE_TRANS: with that number of neighbors#  I.e. it is wrong.')
#
#     return n_found


# def kdtree2_r_count(tp, qv, r2):
#     sr.qv = qv
#     sr.ball_size = r2
#
#     sr.nn = 0  # flag for fixed r search
#     sr.n_found = 0
#     sr.center_idx = -1
#     sr.cor_rel_time = 0
#
#     sr.results = None  # for some reason, FTN 95 chokes on '= null()'
#
#     sr.n_alloc = 0  # we do not allocate any storage but that's OK
#     # for counting.
#     sr.ind = tp.ind
#     sr.rearrange = tp.rearrange
#     if tp.rearrange:
#         sr.data = tp.rearranged_data
#     else:
#         sr.data = tp.the_data
#
#     sr.dimension = tp.dimension
#
#     sr.overflow = False
#
#     search(tp.root)
#
#     n_found = sr.n_found
#     return n_found


# def kdtree2_r_count_around_point(tp, idx_in, cor_rel_time, r2):
#     sr.qv = np.empty(tp.dimension)
#     sr.qv = tp.the_data[:, idx_in]
#     sr.ball_size = r2
#
#     sr.nn = 0  # flag for fixed r search
#     sr.n_found = 0
#     sr.center_idx = idx_in
#     sr.cor_rel_time = cor_rel_time
#     sr.results = None
#
#     sr.n_alloc = 0  # we do not allocate any storage but that's OK
#     # for counting.
#
#     sr.ind = tp.ind
#     sr.rearrange = tp.rearrange
#     if sr.rearrange:
#         sr.data = tp.rearranged_data
#     else:
#         sr.data = tp.the_data
#     sr.dimension = tp.dimension
#     sr.overflow = False
#
#     search(tp.root)
#
#     n_found = sr.n_found
#     return n_found

@njit(nogil=True, error_model='numpy', boundscheck=True)
def validate_query_storage(n, sr):
    # if np.shape(sr.results)[0] < n:
    #     raise ValueError('KD_TREE_TRANS:  you did not provide enough storage for results[:n]')
    assert np.shape(sr.results)[0] >= n, 'KD_TREE_TRANS:  you did not provide enough storage for results[:n]'


# def square_distance(d, iv, qv):
#     res = sum((iv[:d] - qv[:d]) ** 2)
#     return res

@njit(nogil=True, error_model='numpy', boundscheck=True)
def search(node: TreeNode, sr):
    if node.left is None or node.right is None:
        # 终端节点
        # todo process_terminal_node_fixed_ball不会被执行
        if sr.nn == 0:
            process_terminal_node_fixed_ball(node, sr)
        else:
            process_terminal_node(node, sr)
    else:
        # 非终端节点
        qv = sr.qv[:]
        cut_dim = node.cut_dim
        q_val = qv[cut_dim]

        if q_val < node.cut_val:
            n_closer = node.left
            n_farther = node.right
            dis = (node.cut_val_right - q_val) ** 2
        else:
            n_closer = node.right
            n_farther = node.left
            dis = (node.cut_val_left - q_val) ** 2

        # todo 条件恒为真
        # if n_closer is not None:
        # 首先搜索同侧的节点
        search(n_closer, sr)

        # we may need to search the second node.
        # todo 条件恒为真
        # if n_farther is not None:
        ball_size = sr.ball_size
        # 如果分割维度上的距离比找到的点中的最大距离小，则在另一侧有可能存在较近的节点
        if dis < ball_size:
            box = node.box[:]
            for i in range(sr.dimension):
                if i != cut_dim:
                    dis += dis2_from_bnd(qv[i], box['lower'][i], box['upper'][i])
                    if dis > ball_size:
                        return

            # 另一侧的节点有可能存在比现有的点更距离的点的情况下搜索
            search(n_farther, sr)


@njit(nogil=True, error_model='numpy', boundscheck=True)
def dis2_from_bnd(x, a_min, a_max):
    if x > a_max:
        return (x - a_max) ** 2
    elif x < a_min:
        return (a_min - x) ** 2
    else:
        return 0.0


# def box_in_search_range(node, sr):
#     dimension = sr.dimension
#     ball_size = sr.ball_size
#     dis = 0.0
#     res = True
#     for i in range(dimension):
#         l = node.box['lower'][i]
#         u = node.box['upper'][i]
#         dis += (dis2_from_bnd(sr.qv[i], l, u))
#         if dis > ball_size:
#             return False
#     return True

@njit(nogil=True, error_model='numpy', boundscheck=True)
def process_terminal_node(node: TreeNode, sr):
    qv = sr.qv[:]
    pqp = sr.pq
    dimension = sr.dimension
    ball_size = sr.ball_size
    rearrange = sr.rearrange
    ind = sr.ind[:]
    data = sr.data[:, :]
    center_idx = sr.center_idx
    cor_rel_time = sr.cor_rel_time

    # 终端节点直接对每个值进行暴力搜索
    for i in range(node.l, node.u + 1):
        if rearrange:
            sd = np.sum(np.square(data[:, i] - qv[:])).item()
            if sd > ball_size:
                continue
            index_of_i = ind[i]  # only read it if we have not broken out
        else:
            index_of_i = ind[i]
            sd = np.sum(np.square(data[:, index_of_i] - qv[:])).item()
            if sd > ball_size:
                continue

        # todo 好像没用到
        # if center_idx > 0:  # doing correlation interval?
        #     if abs(index_of_i - center_idx) < cor_rel_time:
        #         continue

        if sr.n_found < sr.nn:
            sr.n_found += 1
            new_pri = pq_insert(pqp, sd, index_of_i)
            if sr.n_found == sr.nn:
                ball_size = new_pri
        else:
            ball_size = pq_replace_max(pqp, sd, index_of_i)

    sr.ball_size = ball_size


@njit(nogil=True, error_model='numpy', boundscheck=True)
def process_terminal_node_fixed_ball(node: TreeNode, sr):
    # 在节点中寻找实际的近邻，并在 sr 数据结构上更新搜索结果

    qv = sr.qv[:]
    dimension = sr.dimension
    ball_size = sr.ball_size
    rearrange = sr.rearrange
    ind = sr.ind[:]
    data = sr.data[:, :]
    center_idx = sr.center_idx
    cor_rel_time = sr.cor_rel_time
    nn = sr.nn  # number to search for
    n_found = sr.n_found

    # search through terminal bucket.
    for i in range(node.l, node.u + 1):
        # 对任何一点都有两种情况，即结果列表要么不足，要么已满。
        # 如果结果列表不足，那么无条件添加该点及其距离。
        # 如果添加的点填满了工作列表，则将 sr.ball_size列表上的最大距离，而不是初始化的 +infinity。
        # 如果结果列表已满，则计算距离，但如果它大于 sr.ball_size，则它不是邻近点。
        # 如果小于 sr.ball_size，则删除前一个最大元素并添加新元素。

        if rearrange:
            sd = np.sum(np.square(data[:, i] - qv[:])).item()
            if sd > ball_size:
                continue
            index_of_i = ind[i]  # only read it if we have not broken out
        else:
            index_of_i = ind[i]
            sd = np.sum(np.square(data[:, index_of_i] - qv[:])).item()
            if sd > ball_size:
                continue

        # todo 好像没用到
        # if center_idx > 0:  # doing correlation interval?
        #     if abs(index_of_i - center_idx) < cor_rel_time:
        #         continue

        n_found += 1
        if n_found > sr.n_alloc:
            # oh nuts, we have to add another one to the tree but
            # there isn't enough room.
            sr.overflow = True
        else:
            sr.results['dis'][n_found] = sd
            sr.results['idx'][n_found] = index_of_i

    sr.n_found = n_found


# def kdtree2_n_nearest_brute_force(tp, qv, nn, results):
#     allocate(all_distances(tp.n))
#     for i in range(tp.n):
#         all_distances[i] = square_distance(tp.dimension, qv, tp.the_data[:, i])
# 
#     # now find 'n' smallest distances
#     for i in range(nn):
#         results['dis'][i] = huge(1.0)
#         results['idx'][i] = -1
# 
#     for i in range(tp.n):
#         if all_distances[i] < results['dis'][nn]:
#             # insert it somewhere on the list
#             for j in range(nn):
#                 if all_distances[i] < results['dis'][j]:
#                     break
#             # now we know 'j'
#             for k in range(nn - 1, j, -1):
#                 results[k + 1] = results[k]
#             results['dis'][j] = all_distances[i]
#             results['idx'][j] = i


# def kdtree2_r_nearest_brute_force(tp, qv, r2, n_found, results):
#     allocate(all_distances(tp.n))
#     for i in range(tp.n):
#         all_distances[i] = square_distance(tp.dimension, qv, tp.the_data[:, i])
# 
#     n_found = 0
#     n_alloc = size(results, 1)
# 
#     for i in range(tp.n):
#         if all_distances[i] < r2:
#             # insert it somewhere on the list
#             if n_found < n_alloc:
#                 n_found = n_found + 1
#                 results(n_found)['dis'] = all_distances[i]
#                 results(n_found)['idx'] = i
# 
#     kdtree2_sort_results(n_found, results)

@njit(nogil=True, error_model='numpy', boundscheck=True)
def kdtree2_sort_results(n_found, results):
    if n_found > 1:
        heapsort_struct(results, n_found)


# def heapsort(a, ind, n):
#     i_left = n / 2 + 1
#     i_right = n
#
#     if n == 1:
#         return
#
#     while True:
#         if i_left > 1:
#             i_left -= 1
#             value = a[i_left]
#             i_value = ind[i_left]
#         else:
#             value = a[i_right]
#             i_value = ind[i_right]
#             a[i_right] = a[0]
#             ind[i_right] = ind[0]
#             i_right -= 1
#             if i_right == 1:
#                 a[0] = value
#                 ind[0] = i_value
#                 return
#         i = i_left
#         j = 2 * i_left
#         while j <= i_right:
#             if j < i_right:
#                 if a[j] < a[j + 1]:
#                     j += 1
#             if value < a[j]:
#                 a[i] = a[j]
#                 ind[i] = ind(j)
#                 i = j
#                 j = j + j
#             else:
#                 j = i_right + 1
#
#         a[i] = value
#         ind[i] = i_value

# 整段代码先做了降序 后做了升序
# 升序每次把最大值放在数组的最后面即i_right处，然后重新排列剩余的节点
# @njit(nogil=True, error_model='numpy', boundscheck=True)
# def heapsort_struct(a, n):
#     # 降序的时候移动左边界索引
#     # 下一层级的左边界索引
#     i_left = n // 2 + 1
#     # 升序的时候移动右边界索引
#     # 下一层级的右边界索引
#     i_right = n
#
#     if n == 1:
#         return
#
#     while True:
#         if i_left > 1:
#             # 暂存上一层级的值为value
#             i_left -= 1
#             value = a['dis'][i_left - 1], a['idx'][i_left - 1]
#         else:
#             value = a['dis'][i_right - 1], a['idx'][i_right - 1]
#             a['dis'][i_right - 1] = a['dis'][0]
#             a['idx'][i_right - 1] = a['idx'][0]
#             i_right -= 1
#             if i_right == 1:
#                 a['dis'][0], a['idx'][0] = value
#                 return
#
#         # 上一层级节点索引
#         i = i_left
#         # 下一层级节点索引（从1开始） j、j+1
#         j = 2 * i_left
#         while j <= i_right:
#
#             # j取下一层级2个节点中较大值的索引
#             if j < i_right:
#                 # 取较大值
#                 if a['dis'][j - 1] < a['dis'][j]:
#                     j += 1
#
#             if value[0] < a['dis'][j - 1]:
#                 a['dis'][i - 1] = a['dis'][j - 1]
#                 a['idx'][i - 1] = a['idx'][j - 1]
#                 # 并递归向下比较
#                 i = j
#                 j = j + j
#             else:
#                 break
#                 # j = i_right + 1
#
#         a['dis'][i - 1], a['idx'][i - 1] = value

# 整段代码先做了降序 后做了升序
# 升序每次把最大值放在数组的最后面即i_right处，然后重新排列剩余的节点
@njit(nogil=True, error_model='numpy', boundscheck=True)
def heapsort_struct(a, n):
    # 降序的时候移动左边界索引
    # 下一层级的左边界索引
    i_left = n // 2
    # 升序的时候移动右边界索引
    # 下一层级的右边界索引
    i_right = n - 1

    if n == 1:
        return

    while True:
        if i_left > 0:
            # 暂存上一层级的值为value
            i_left -= 1
            value = a['dis'][i_left], a['idx'][i_left]
        else:
            value = a['dis'][i_right], a['idx'][i_right]
            a['dis'][i_right] = a['dis'][0]
            a['idx'][i_right] = a['idx'][0]
            i_right -= 1
            if i_right == 0:
                a['dis'][0], a['idx'][0] = value
                return

        # 上一层级节点索引
        i = i_left
        # 下一层级节点索引（从1开始） j、j+1
        j = 2 * i_left + 1
        while j <= i_right:

            # j取下一层级2个节点中较大值的索引
            if j < i_right:
                # 取较大值
                if a['dis'][j] < a['dis'][j]:
                    j += 1

            if value[0] < a['dis'][j]:
                a['dis'][i] = a['dis'][j]
                a['idx'][i] = a['idx'][j]
                # 并递归向下比较
                i = j
                j = j + j + 1
            else:
                break
                # j = i_right + 1

        a['dis'][i], a['idx'][i] = value


if __name__ == '__main__':
    my_array = np.random.random((2, 100)).astype('f4')

    nnbrute = 5
    query_vec = np.random.random(2).astype('f4')
    results1 = np.empty(nnbrute, kdtree2_result)
    tree = kdtree2_create(my_array, sort=True, rearrange=True)
    kdtree2_n_nearest(tp=tree, qv=query_vec, nn=nnbrute, results=results1)

    print(my_array)
    print()
    print(query_vec)
    print()
    print(results1['dis'] ** 0.5)  # 这里存储的结果是scipy结果中距离的平方
    print(results1['idx'])
    print()

    from scipy import spatial

    tree = spatial.KDTree(my_array.T)
    print(tree.data)
    print()
    print(tree.query(query_vec, nnbrute))
    print()
