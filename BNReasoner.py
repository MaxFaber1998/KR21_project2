import copy
import random
from typing import Union

from BayesNet import BayesNet
import itertools
from enum import IntEnum
import networkx as nx
import pandas as pd


class ValveType(IntEnum):
    CONVERGENT = 1
    DIVERGENT = 2
    SEQUENTIAL = 3


class BNReasoner:
    def __init__(self, net: Union[str, BayesNet]):
        """
        :param net: either file path of the bayesian network in BIFXML format or BayesNet object
        """
        if type(net) == str:
            # constructs a BN object
            self.bn = BayesNet()
            # Loads the BN from an BIFXML file
            self.bn.load_from_bifxml(net)
        else:
            self.bn = net

    def is_d_separated(self, X: set, Y: set, Z: set) -> tuple[list[str], bool]:
        """
        Checks whether X and Y are d-separated by Z, which implies that no alternative open connection is found
        between X and Y.
        :param X: a set of X nodes.
        :param Y: a set of Y nodes.
        :param Z: a set of nodes which are d-separating
        :return is an empty path on d-separation or a counter example (open path)
        """
        edge_index_graph = self.__get_edge_index_graph__(self.bn)
        all_nodes = X.union(Y).union(Z)

        assert all_nodes.issubset(set(self.bn.structure.nodes)), "Some passed nodes are non-existent."
        assert X.isdisjoint(Y) and X.isdisjoint(Z) and Y.isdisjoint(Z), "All given sets must be disjoint."
        for x, y in itertools.product(X, Y):
            for path in nx.all_simple_paths(edge_index_graph, source=x, target=y):
                # is a direct node and thus never seperated by any w
                if len(path) == 2:
                    return path, False
                open_valve = True
                for i in range(len(path) - 2):
                    n_x = path[i]
                    n_w = path[i + 1]
                    n_y = path[i + 2]
                    v_t = self.__get_valve_type__(self.bn, n_x, n_w, n_y)
                    if v_t == ValveType.CONVERGENT:
                        w_and_succ = set(self.__get_all_succ__(self.bn, n_w)).union({n_w})
                        if w_and_succ.isdisjoint(Z):
                            open_valve = False
                            break
                    else:
                        open_valve = n_w not in Z
                        if n_w in Z:
                            open_valve = False
                            break

                if open_valve:
                    return path, False
        return [], True

    def prune(self, X: set[str], Y: set[str], Z: dict[str, bool]) -> BayesNet:
        """
        Prunes the BayesNet based on given X, Y and Z sets (disjoint). This reduces the network size.
        Do iteratively:
            Step 1: delete all leaf-node, not in union(X,Y,Z)
            Step 2: delete all outgoing nodes from nodes in Z
        :param X: nodes in X.
        :param Y: nodes in Y.
        :param Z: nodes in Z.
        :return a copied BayesNet which is pruned based on given input.
        """
        nodes_to_keep = X.union(Y).union(Z.keys())
        assert nodes_to_keep.issubset(set(self.bn.structure.nodes)), "Some passed nodes are non-existent."
        cbn = copy.deepcopy(self.bn)

        leaf_nodes = self.__get_all_leaf_nodes__(cbn) - nodes_to_keep
        first = True
        while len(leaf_nodes) or first:
            for ln in leaf_nodes:
                cbn.del_var(ln)
            for z, tv in Z.items():
                for o_e in self.__get_all_outgoing_edges__(z, cbn):
                    cbn.del_edge((z, o_e))
                    c = cbn.get_cpt(o_e)
                    c = self.__reduce_cpt__(c, (z, tv))
                    cbn.update_cpt(o_e, c)
            leaf_nodes = self.__get_all_leaf_nodes__(cbn) - nodes_to_keep
            first = False
        return cbn

    def order_of_elimination(self, X: set[str], heuristic: str) -> list[str]:
        """
        Calculates the order of elimination given a set of variables and a heuristic
        :param X: set of nodes
        :param heuristic: a heuristic (min-degree or min-fill)
        :return the elimination order.
        """
        assert heuristic in ['min-degree', 'min-fill',
                             'random'], "Heuristic should be 'min-degree' or 'min-fill' or 'random'."
        ig = self.bn.get_interaction_graph()
        if heuristic == 'min-degree':
            copy_x = copy.copy(X)
            ordered_x = []
            while len(copy_x):
                # sort first by # of edges added and then alphabetically for consistency
                l_x = sorted(list(copy_x), key=lambda k: (ig.degree[k], k))
                node = l_x[0]
                new_edges = self.__get_new_edges__(node, ig)
                ordered_x.append(node)
                ig.remove_node(node)
                copy_x.remove(node)
                for i, j in new_edges:
                    ig.add_edge(u_of_edge=i, v_of_edge=j)
            return ordered_x
        elif heuristic == 'min-fill':
            # sort first by # of edges added and then alphabetically for consistency
            return sorted(list(X), key=lambda x: (len(self.__get_new_edges__(x, ig)), x))
        else:
            # random order of elimination
            list_x = list(X)
            random.shuffle(list_x)
            return list_x

    def compute_marginal_distribution(self, Q: set[str], E: dict[str, bool] = None, heuristic=None,
                                      aggr: str = 'sum', normalize=True) -> pd.DataFrame:
        E = E if E is not None else {}
        """
        Computes the marginal distribution for the variables in Q, with optional evidence E.
        :param Q: nodes to calculate the marginal distribution for
        :param E: evidence provided
        :param aggr: the aggregation type (sum/max)
        :param normalize: whether to normalize the result.
        :return a dict of nodes in Q, with the marginal distribution per truth value.
        """
        assert Q.union(E.keys()).issubset(set(self.bn.structure.nodes)), "Some passed nodes are non-existent."
        assert aggr in ['sum', 'max'], 'Please specify table group operator [sum, max].'
        self.bn = self.prune(Q, set(), E)

        not_Q = set(self.bn.get_all_variables()) - Q
        all_cpts = {x: self.bn.get_cpt(x) for x in Q.union(not_Q)}
        # zero-out contradicting nodes with the evidence
        for e, tv in E.items():
            for x, df in all_cpts.items():
                if e in df.columns:
                    for idx, row in df.iterrows():
                        if df.loc[idx, e] != tv:
                            df.loc[idx, 'p'] = 0
                    all_cpts[x] = df

        if heuristic is None:
            # build the order of elimination for efficiency purposes and check which is most efficient
            min_deg_ooe = self.order_of_elimination(not_Q, 'min-degree')
            min_fill_ooe = self.order_of_elimination(not_Q, 'min-fill')
            min_deg_width = self.__get_ooe_width__(self.bn.get_interaction_graph(), min_deg_ooe)
            min_fill_width = self.__get_ooe_width__(self.bn.get_interaction_graph(), min_fill_ooe)
            ooe = min_deg_ooe if min_deg_width < min_fill_width else min_fill_ooe
        else:
            ooe = self.order_of_elimination(not_Q, heuristic)

            # create a CPT list of all CPTs
        all_cpts_list = list(all_cpts.values())

        for i, pi in enumerate(ooe):
            # find all cpts with pi in it
            all_cpts_with_pi = [x for x in all_cpts_list if pi in x.columns]
            # keep all other cpts with not pi
            rest = [x for x in all_cpts_list if pi not in x.columns]
            # make one big cpt of all cpts with pi
            combined = all_cpts_with_pi[0]
            for i in range(1, len(all_cpts_with_pi)):
                combined = self.__multiply_cpts__(all_cpts_with_pi[i], combined)
            # remove pi from the big combined cpt
            combined = self.__group_cpt__(combined, {pi}, aggr)
            # save both the super table and the untouched other tables
            rest.append(combined)
            all_cpts_list = rest

        # multiply all leftover (Q) nodes (they are dependent) (could be one and thus no multiplication needed)
        current_cpt = all_cpts_list[0]
        for i in range(1, len(all_cpts_list)):
            current_cpt = self.__multiply_cpts__(current_cpt, all_cpts_list[i])

        # calculate the evidence factor and use it to normalize all results
        if normalize:
            evidence_factor = self.__get_evidence_factor__(E)
            for idx, row in current_cpt.iterrows():
                current_cpt.loc[idx, 'p'] /= evidence_factor
        return current_cpt

    def calculate_MAP(self, Q: set[str], E: dict[str, bool], heuristic=None) -> pd.DataFrame:
        """
        Calculates the Max A Posteriori for given Q and E
        :param Q: Q variables (to marginalize)
        :param E: evidence (must be disjoint with Q)
        :param heuristic: heuristic to use
        :return: pd.Dataframe with the MAP for given Q and E
        """
        marginal_q = self.compute_marginal_distribution(Q, E, aggr='sum', normalize=False, heuristic=heuristic)
        # This is the instantiation of Q with the highest probability given E
        return marginal_q.iloc[marginal_q['p'].idxmax()]

    def calculate_MPE(self, E: dict[str, bool], heuristic=None) -> pd.DataFrame:
        """
        Calculates the Most Probable Evidence for given Q and E, using the (self Bayesian Network)
        :param E: evidence.
        :param heuristic: heuristic to use
        :return: pd.Dataframe with the MPE of Q given E
        """
        Q = set(self.bn.get_all_variables())
        marginal_q = self.compute_marginal_distribution(Q, E, aggr='max', normalize=False, heuristic=heuristic)
        # This is the most probably evidence for of Q and E
        return marginal_q.iloc[marginal_q['p'].idxmax()]

    def __get_evidence_factor__(self, E: dict[str, bool]) -> float:
        """
        Gets the evidence factor (used for marginal division) for given evidence and their CPTs
        :param E: dict with all the evidences (k=evidence node, v=truth value)
        :return: float of the combined evidences factor
        """
        if len(E) == 0:
            return 1
        marginal = self.compute_marginal_distribution(set(E.keys()), {}, aggr='sum')
        for c, row in marginal.iterrows():
            valid = True
            f = 1
            # find the row (should be one) where all evidences are met
            for n, v in row.items():
                if n == 'p':
                    # if we encounter 'p', save the probability for efficiency
                    f = v
                # search for a contradicting row
                elif E[str(n)] != v:
                    valid = False
                    break
            if valid:
                return f
        # should never be encountered, otherwise the implementation is wrong
        raise Exception("Evidence not found?")

    @staticmethod
    def __reduce_cpt__(cpt: pd.DataFrame, x: tuple[str, bool]) -> pd.DataFrame:
        """
        Reduces a CPT by first removing the rows where x[bool] is false and then removing the column x
        cpt: cpt to remove x from
        x: tuple of node and evidence
        """
        x0 = x[0]
        tv = x[1]
        if x0 not in cpt.columns:
            return cpt
        cpt = cpt[cpt[x0] == tv]
        cpt = cpt.drop(columns=[x0])
        return cpt

    @staticmethod
    def __group_cpt__(cpt: pd.DataFrame, nodes_to_delete: set[str], agg_opr: str) -> pd.DataFrame:
        """
        Groups out the given CPT, grouping by the variables not in given nodes_to_delete
        :param cpt: CPT to group out
        :param nodes_to_delete: nodes to delete in the CPT
        :param agg_opr: aggregation operator using for the grouping.
        """
        assert nodes_to_delete.issubset(set(cpt.columns)), "Non-existent nodes passed."
        result = cpt.copy(deep=True)
        originals = list(result.columns)
        # delete the nodes to eliminate in the nodes to keep
        for n in nodes_to_delete:
            originals.remove(n)
        # remove the 'p', because this shouldn't be used in the aggregation
        originals.remove('p')
        if not len(originals):
            return pd.DataFrame()
        for n in nodes_to_delete:
            if n in result.columns:
                result = result.drop(labels=n, axis=1)
        result = result.groupby(originals).agg(agg_opr).reset_index()
        return result

    @staticmethod
    def __multiply_cpts__(cpt1: pd.DataFrame, cpt2: pd.DataFrame) -> pd.DataFrame:
        """
        Multiplies two CPTs and returns the combined one
        :param cpt1: the first CPT
        :param cpt2: the seconds CPT
        :return combined (multiplied) CPT
        """
        result = cpt1.copy(deep=True)
        cpt1_columns = set(cpt1.columns) - {'p'}
        cpt2_columns = set(cpt2.columns) - {'p'}

        diff = cpt2_columns.difference(cpt1_columns)

        # increase the table by adding the difference
        for d in diff:
            insert_idx = len(cpt1_columns) - 1
            if {True, False}.issubset(set(cpt2[d])):
                old = copy.deepcopy(result)
                result.insert(insert_idx, d, True)
                result = pd.concat([result, old]).fillna(False)
            else:
                for tv in [True, False]:
                    result.insert(insert_idx, d, tv)
            result = result.sort_values(by=list(result.columns)).reset_index(drop=True)

        # multiply the probabilities
        for idx_result_row, result_row in result.iterrows():
            for _, cpt2_row in cpt2.iterrows():
                if result_row[cpt2_columns].equals(cpt2_row[cpt2_columns]):
                    result.at[idx_result_row, 'p'] *= cpt2_row['p']

        return result

    @staticmethod
    def __get_ooe_width__(ig: nx.DiGraph, order_of_elimination: list[str]) -> int:
        """
        Gets the width of the order of elimination variables
        :param ig: the current interaction graph.
        :param order_of_elimination: list of variables to eliminate, ordered
        :return an integer of the width
        """
        assert len(set(order_of_elimination)) == len(order_of_elimination)
        w = 0
        for x in order_of_elimination:
            new_edges = BNReasoner.__get_new_edges__(x, ig)
            w = max(w, ig.degree[x])
            for i, j in new_edges:
                ig.add_edge(u_of_edge=i, v_of_edge=j)
            ig.remove_node(x)
        return w

    @staticmethod
    def __get_all_succ__(bn: BayesNet, x: str, succ: list[str] = None) -> list[str]:
        """
        Gets all successors of a given x node, recursively.
        :param bn: BayesNet instance.
        :param x: a node from the network.
        :param succ: updated instance of all successors.
        :return is a set of all successors of x in the BN.
        """
        if succ is None:
            succ = []
        cur_succ = bn.structure.succ[x]
        if not len(cur_succ):
            return succ
        for p in bn.structure.succ[x]:
            if p not in succ:
                succ.append(p)
            BNReasoner.__get_all_succ__(bn, p, succ)

    @staticmethod
    def __get_all_pred__(bn: BayesNet, x: str, pred: list[str] = None) -> list[str]:
        """
        Gets all predecessors of a given x node, recursively.
        :param bn: BayesNet instance.
        :param x: a node from the network.
        :param pred: updated instance of all predecessors.
        :return is a set of all predecessors of x in the BN.
        """
        if pred is None:
            pred = []
        cur_pred = bn.structure.pred[x]
        if not len(cur_pred):
            return pred
        for p in bn.structure.pred[x]:
            if p not in pred:
                pred.append(p)
            BNReasoner.__get_all_pred__(bn, p, pred)

    @staticmethod
    def __get_all_leaf_nodes__(bn: BayesNet) -> set[str]:
        """
        Gets all the leaf nodes from given BayesNet.
        :param bn: BayesNet instance.
        :return set with leaf nodes.
        """
        ln = []
        for node in bn.structure.nodes:
            if not len(bn.get_children(node)):
                ln.append(node)
        return set(ln)

    @staticmethod
    def __get_all_outgoing_edges__(x: str, bn: BayesNet) -> set[str]:
        """
        Gets all the outgoing edges from given node x and a BayesNet instance.
        :param x: a node in the network
        :return set with all outgoing edges, f.e. ['1', '3'] means x points to '1' and '3'.
        """
        edges = []
        for f, t in bn.structure.edges(nbunch=x):
            edges.append(t)
        return set(edges)

    @staticmethod
    def __get_new_edges__(deleted_x: str, ig: nx.DiGraph) -> list[tuple[str, str]]:
        """
        Gets new edges if a given 'x' would be deleted. New edges are between the direct neighbors of x and already
        existing ones are ignored
        :param deleted_x: x which is the centre of current context
        :param ig: a graph in which at least x should be present
        :return a list of new edges between direct neighbours of x
        """
        n = list(ig.neighbors(deleted_x))
        comb = list(itertools.combinations(n, r=2))
        ae_conn = list(filter(lambda u: u in ig.edges, comb))
        result = list(set(comb) - set(ae_conn))
        return result

    @staticmethod
    def __get_valve_type__(bn: BayesNet, n_x, n_w, n_y) -> ValveType:
        """
        Gets the valve type between x and y with w in between
        :param bn: BayesNet instance.
        :param n_x: node x.
        :param n_w: in-between node w
        :param n_y: node y.
        :return type of valve.
        """
        if n_x in bn.structure.pred[n_w]:
            if n_y in bn.structure.pred[n_w]:
                return ValveType.CONVERGENT
            if n_y in bn.structure.succ[n_w]:
                return ValveType.SEQUENTIAL
        if n_x in bn.structure.succ[n_w]:
            if n_y in bn.structure.succ[n_w]:
                return ValveType.DIVERGENT
        if n_y in bn.structure.pred[n_w]:
            if n_x in bn.structure.succ[n_w]:
                return ValveType.SEQUENTIAL
        raise Exception('No (direct) valve between x: "{}", w: "{}" and y: "{}"'.format(n_x, n_w, n_y))

    @staticmethod
    def __get_edge_index_graph__(bn: BayesNet) -> nx.Graph:
        """
        Generates an edge index graph for given BN.
        :param bn: BayesNet instance.
        :return is an nx.Graph
        """
        edge_index_graph = nx.Graph()
        edge_index_graph.add_edges_from(bn.structure.edges)
        return edge_index_graph


# TODO: remember to never use the same BNReasoner instance -> this can lead to unexpected behavior
if __name__ == '__main__':
    # Example usage
    print(BNReasoner('testing/lecture_example_pgm4.BIFXML').is_d_separated({'X'}, {'J'}, {'O'}))
    print(BNReasoner('testing/lecture_example_pgm4.BIFXML').compute_marginal_distribution({'J', 'I'}))
    print(BNReasoner('testing/lecture_example_pgm4.BIFXML').compute_marginal_distribution({'J', 'I'}, {'O': True}))
    print(BNReasoner('testing/lecture_example_pgm4.BIFXML').calculate_MPE({'O': True}))
    print(BNReasoner('testing/lecture_example_pgm4.BIFXML').calculate_MPE({'J': True, 'O': False}))
