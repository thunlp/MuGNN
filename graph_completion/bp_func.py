import numpy as np
import igraph as ig
import gc, time, itertools


class P(object):
    lambda_3 = 0.7


def bootstrapping(ref_ent_sr, ref_ent_tg, ref_sim_mat, labeled_alignment, top_k):
    # ref_sim_mat shape = [test_num, test_num]
    th = P.lambda_3
    n = ref_sim_mat.shape[0]
    curr_labeled_alignment = find_potential_alignment(ref_sim_mat, th, top_k, n)
    if curr_labeled_alignment is not None:
        labeled_alignment = update_labeled_alignment(labeled_alignment, curr_labeled_alignment, ref_sim_mat, n)
        labeled_alignment = update_labeled_alignment_label(labeled_alignment, ref_sim_mat, n)
        del curr_labeled_alignment
    # OK
    # labeled_alignment = curr_labeled_alignment
    if labeled_alignment is not None:
        # ref_ent list of ref_ent ids
        ents1 = [ref_ent_sr[pair[0]] for pair in labeled_alignment]
        ents2 = [ref_ent_tg[pair[1]] for pair in labeled_alignment]
    else:
        ents1, ents2 = None, None
    del ref_sim_mat
    gc.collect()
    ent_seeds = list(zip(ents1, ents2))
    return labeled_alignment, ent_seeds


def find_potential_alignment(sim_mat, sim_th, k, total_n):
    t = time.time()
    potential_aligned_pairs = generate_alignment(sim_mat, sim_th, k, total_n)
    if potential_aligned_pairs is None or len(potential_aligned_pairs) == 0:
        return None
    t1 = time.time()
    selected_pairs = mwgm_igraph(potential_aligned_pairs, sim_mat)
    check_alignment(selected_pairs, total_n, context="selected_pairs")
    del potential_aligned_pairs
    print("mwgm costs time: {:.3f} s".format(time.time() - t1))
    print("selecting potential alignment costs time: {:.3f} s".format(time.time() - t))
    return selected_pairs


def generate_alignment(sim_mat, sim_th, k, all_n):
    potential_aligned_pairs = filter_mat(sim_mat, sim_th)
    if len(potential_aligned_pairs) == 0:
        return None
    check_alignment(potential_aligned_pairs, all_n, context="after sim filtered")
    neighbors = search_nearest_k(sim_mat, k)
    if neighbors is not None:
        potential_aligned_pairs &= neighbors
        if len(potential_aligned_pairs) == 0:
            return None, None
        check_alignment(potential_aligned_pairs, all_n, context="after sim and neighbours filtered")
    del neighbors
    return potential_aligned_pairs


def filter_mat(mat, threshold, greater=True, equal=False):
    if greater and equal:
        x, y = np.where(mat >= threshold)
    elif greater and not equal:
        x, y = np.where(mat > threshold)
    elif not greater and equal:
        x, y = np.where(mat <= threshold)
    else:
        x, y = np.where(mat < threshold)
    return set(zip(x, y))


def check_alignment(aligned_pairs, all_n, context="", is_cal=True):
    if aligned_pairs is None or len(aligned_pairs) == 0:
        print("{}, Empty aligned pairs".format(context))
        return
    num = 0
    for x, y in aligned_pairs:
        if x == y:
            num += 1
    print("{}, right alignment: {}/{}={:.3f}".format(context, num, len(aligned_pairs), num / len(aligned_pairs)))
    if is_cal:
        precision = round(num / len(aligned_pairs), 6)
        recall = round(num / all_n, 6)
        if recall > 1.0:
            recall = round(num / all_n, 6)
        f1 = round(2 * precision * recall / (precision + recall), 6)
        print("precision={}, recall={}, f1={}".format(precision, recall, f1))


def search_nearest_k(sim_mat, k):
    if k == 0:
        return None
    neighbors = set()
    ref_num = sim_mat.shape[0]
    for i in range(ref_num):
        rank = np.argpartition(-sim_mat[i, :], k)
        pairs = [j for j in itertools.product([i], rank[0:k])]
        neighbors |= set(pairs)
        # del rank
    assert len(neighbors) == ref_num * k
    return neighbors


def mwgm_igraph(pairs, sim_mat):
    if not isinstance(pairs, list):
        pairs = list(pairs)
    index_id_dic1, index_id_dic2 = dict(), dict()
    index1 = set([pair[0] for pair in pairs])
    index2 = set([pair[1] for pair in pairs])
    for index in index1:
        index_id_dic1[index] = len(index_id_dic1)
    off = len(index_id_dic1)
    for index in index2:
        index_id_dic2[index] = len(index_id_dic2) + off
    assert len(index1) == len(index_id_dic1)
    assert len(index2) == len(index_id_dic2)
    edge_list = [(index_id_dic1[x], index_id_dic2[y]) for (x, y) in pairs]
    weight_list = [sim_mat[x, y] for (x, y) in pairs]
    leda_graph = ig.Graph(edge_list)
    leda_graph.vs["type"] = [0] * len(index1) + [1] * len(index2)
    leda_graph.es['weight'] = weight_list
    res = leda_graph.maximum_bipartite_matching(weights=leda_graph.es['weight'])
    print(res)
    selected_index = [e.index for e in res.edges()]
    matched_pairs = set()
    for index in selected_index:
        matched_pairs.add(pairs[index])
    return matched_pairs


def update_labeled_alignment(labeled_alignment, curr_labeled_alignment, sim_mat, all_n):
    # all_alignment = labeled_alignment | curr_labeled_alignment
    # check_alignment(labeled_alignment, all_n, context="before updating labeled alignment")
    labeled_alignment_dict = dict(labeled_alignment)
    n, n1 = 0, 0
    for i, j in curr_labeled_alignment:
        if labeled_alignment_dict.get(i, -1) == i and j != i:
            n1 += 1
        if i in labeled_alignment_dict.keys():
            jj = labeled_alignment_dict.get(i)
            old_sim = sim_mat[i, jj]
            new_sim = sim_mat[i, j]
            if new_sim >= old_sim:
                if jj == i and j != i:
                    n += 1
                labeled_alignment_dict[i] = j
        else:
            labeled_alignment_dict[i] = j
    print("update wrongly: ", n, "greedy update wrongly: ", n1)
    labeled_alignment = set(zip(labeled_alignment_dict.keys(), labeled_alignment_dict.values()))
    check_alignment(labeled_alignment, all_n, context="after editing labeled alignment (<-)")
    # selected_pairs = mwgm(all_alignment, sim_mat, mwgm_igraph)
    # check_alignment(selected_pairs, context="after updating labeled alignment with mwgm")
    return labeled_alignment


def update_labeled_alignment_label(labeled_alignment, sim_mat, all_n):
    # check_alignment(labeled_alignment, all_n, context="before updating labeled alignment label")
    labeled_alignment_dict = dict()
    updated_alignment = set()
    for i, j in labeled_alignment:
        ents_j = labeled_alignment_dict.get(j, set())
        ents_j.add(i)
        labeled_alignment_dict[j] = ents_j
    for j, ents_j in labeled_alignment_dict.items():
        if len(ents_j) == 1:
            for i in ents_j:
                updated_alignment.add((i, j))
        else:
            max_i = -1
            max_sim = -10
            for i in ents_j:
                if sim_mat[i, j] > max_sim:
                    max_sim = sim_mat[i, j]
                    max_i = i
            updated_alignment.add((max_i, j))
    check_alignment(updated_alignment, all_n, context="after editing labeled alignment (->)")
    return updated_alignment
