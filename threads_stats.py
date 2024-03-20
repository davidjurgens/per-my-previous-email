
import gzip
import json
import networkx as nx
from datetime import datetime
import pandas as pd


def get_tree(l, id1):
    id2 = l['headers']['message_id']
    tree = []
    if id1!='': tree.append([id1, id2])
    if 'replies' in l:
        for l1 in l['replies']:
            tree.extend(get_tree(l1, id2))
    return tree

def _users_in_tree(node_dict, user_set):
    user_set.add(node_dict["headers"]["from"])
    if "replies" in node_dict:
        for node_dict_child in node_dict["replies"]:
            user_set = user_set.union(_users_in_tree(node_dict_child, user_set))
    return user_set

if __name__ == '__main__':
    with gzip.open("/shared/4/datasets/gmane/reply-threads.jsonl.gz") as f:
        errors = 0
        roots = list()
        properties_per_thread = dict()
        for line_idx, line in enumerate(f):
            try:
                line = json.loads(line.decode())
                root_id = line['headers']['message_id']
                tree = get_tree(line, '')
            except (NameError, TypeError) as e:
                errors += 1
            g = nx.from_edgelist(tree, create_using=nx.DiGraph())
            tree_depth = max(nx.shortest_path_length(g, root_id).values())
            if tree_depth > 2:
                #depths[root_id] = tree_depth
                roots.append(root_id)
                cur_leaves = sum([g.out_degree(x) == 0 for x in g.nodes()])
                #leaves[root_id] = sum([g.out_degree(x) == 0 for x in g.nodes()])
                #posts[root_id] = list(set([x for y in tree for x in y if x != root_id]))
                cur_width = max([g.out_degree(x) for x in g.nodes()])
                #tree_width[root_id] = cur_width
                #depth_width_balanced[root_id] = cur_width/cur_width
                cur_branches = sum([g.out_degree(x) > 1 for x in g.nodes()])
                #branches[root_id] = sum([g.out_degree(x) > 1 for x in g.nodes()])
                users_in_tree = _users_in_tree(line, set())
                num_users_in_tree = len(users_in_tree)
                #users[root_id] = num_users_in_tree
                #try:
                #    sizes[root_id] = line['size']
                #    groups[root_id] = line['group']
                #    dates[root_id] = line['headers']['date']
                #except TypeError:
                #    errors += 1
                properties_per_thread[root_id] = {'depths': tree_depth,
                                                  'leaves': cur_leaves,
                                                  'width': cur_width,
                                                  'depth_width_balanced': round(tree_depth*1.0/cur_width*1.0, 3),
                                                  'branches': cur_branches,
                                                  'num_users': num_users_in_tree}
            if line_idx % 100000 == 0:
                print(f"Passed {line_idx} cases. Found {len(roots)} valid trees. Errors cnt: {errors} Time: {datetime.now()}")
    # creating and saving a df
    threads_stats = pd.DataFrame.from_dict(properties_per_thread, orient='index')
    threads_stats.to_csv('threads_stats.csv', index=True)
