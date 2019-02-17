import random, pickle
from graph_completion.TripleGraph import TripleGraph
from data.reader import read_mapping, read_triples, read_seeds, read_rules
from tools.print_time_info import print_time_info


def dict_union(dict1, dict2):
    '''
    use it with careful
    '''
    for key, value in dict2.items():
        dict1[key] = value
    return dict1


def _check(ori, new, num):
    if len(ori) != len(new):
        print_time_info('Check failed %d.' % num, dash_top=True)
        raise ValueError()


def _load_languge(directory, language):
    triples = read_triples(directory / ('triples_' + language + '.txt'))
    entity2id = read_mapping(directory / ('entity2id_' + language + '.txt'))
    relation2id = read_mapping(
        directory / ('relation2id_' + language + '.txt'))
    rules = read_rules(directory / 'AMIE' /
                       ('rule_for_triples_' + language + '.txt'), relation2id)
    id2entity = {i: entity for entity, i in entity2id.items()}
    id2relation = {i: relation for relation, i in relation2id.items()}
    return triples, id2entity, id2relation, rules


def _load_seeds(directory, train_seeds_ratio):
    relation_seeds = read_seeds(directory / 'relation_seeds.txt')
    entity_seeds = read_seeds(
        directory / 'JAPE' / ('0_' + str(int(10 * train_seeds_ratio))) / 'train_entity_seeds.txt')
    return entity_seeds, relation_seeds


def print_triple(triple, id2entity, id2relation, end='\n'):
    head, tail, relation = triple
    print_time_info(
        ' '.join([id2entity[head], id2relation[relation], id2entity[tail]]), end=end)


def print_rule(rule, id2relation):
    premises, hypothesis, conf = rule
    premises = [(premise[0], id2relation[premise[2]], premise[1])
                for premise in premises]
    hypothesis = [(hypothesis[0], id2relation[hypothesis[2]], hypothesis[1])]
    rule = premises + [['=>']] + hypothesis + [[str(conf)]]
    print_time_info('  '.join(' '.join(part) for part in rule))


def _print_new_rules(bi_new_rules, id2relation_sr, id2relation_tg):
    for language, rules in bi_new_rules.items():
        print_time_info(language, dash_top=True)
        for rule in random.choices(rules, k=10):
            if language == 'sr':
                print_rule(rule, id2relation_sr)
            else:
                print_rule(rule, id2relation_tg)


def _print_new_triple_confs(bi_new_triple_confs, id2entity_sr, id2entity_tg, id2relation_sr, id2relation_tg):
    for language, triple_confs in bi_new_triple_confs.items():
        print_time_info(language, dash_top=True)
        for triple in random.choices(list(triple_confs.keys()), k=10):
            conf = triple_confs[triple]
            if language == 'sr':
                print_triple(triple, id2entity_sr, id2relation_sr, end='')
            else:
                print_triple(triple, id2entity_tg, id2relation_tg, end='')
            print(' ', conf)


def get_relation2conf(rules):
    relation2conf = {}
    for premises, hypothesis, conf in rules:
        inferred_relation = hypothesis[2]
        if inferred_relation in relation2conf:
            relation2conf[inferred_relation].append(float(conf))
        else:
            relation2conf[inferred_relation] = [float(conf)]
    relation2conf = {relation: sum(confs)/len(confs)
                     for relation, confs in relation2conf.items()}
    return relation2conf

def get_relation2imp(triples, relation_num):
    relation2imp = {str(i):{'head': set(), 'tail': set()} for i in range(relation_num)}
    for head, tail, relation in triples:
        relation2imp[relation]['head'].add(head)
        relation2imp[relation]['tail'].add(tail)
    relation2imp = {relation: 1-min(1, len(ht['head'])/len(ht['tail'])) for relation, ht in relation2imp.items()}
    return relation2imp

def rule_based_graph_completion(triple_graph_sr, triple_graph_tg, rules_sr, rules_tg):
    '''
    triples = [(head, tail, relation)]
    return new [((head, tail, relation), conf)...]
    '''
    print_time_info('Rule based graph completion started!')

    def _rule_based_graph_completion(triple_graph, rules):
        triples = triple_graph.triples
        new_triple_confs = {}
        for rule in rules:
            new_triple_conf_candidates = triple_graph.inference_by_rule(rule)
            for new_triple, conf in new_triple_conf_candidates:
                if not new_triple in triples:
                    if new_triple not in new_triple_confs:
                        new_triple_confs[new_triple] = conf
                    else:
                        ori_conf = new_triple_confs[new_triple]
                        new_triple_confs[new_triple] = max(conf, ori_conf)
        return new_triple_confs

    new_triple_confs_sr = _rule_based_graph_completion(
        triple_graph_sr, rules_sr)
    new_triple_confs_tg = _rule_based_graph_completion(
        triple_graph_tg, rules_tg)
    print_time_info('Rule based graph completion finished!')
    return new_triple_confs_sr, new_triple_confs_tg


def rule_transfer(rules_sr, rules_tg, relation_seeds):
    '''
    目前仅能处理len(premises) <= 2的情况
    '''
    rule2conf_sr = {(premises, hypothesis): conf for premises,
                    hypothesis, conf in rules_sr}
    rule2conf_tg = {(premises, hypothesis): conf for premises,
                    hypothesis, conf in rules_tg}
    from pprint import pprint

    def _rule_transfer(rule2conf_sr, rule2conf_tg, r2r):
        new_rules = []
        for rule, conf in rule2conf_sr.items():
            premises, hypothesis = rule
            relations = [premise[2] for premise in premises] + [hypothesis[2]]
            feasible = True
            for relation in relations:
                if not relation in r2r:
                    feasible = False
            if feasible:
                premises = tuple([(head, tail, r2r[relation])
                                  for head, tail, relation in premises])
                hypothesis = (hypothesis[0], hypothesis[1], r2r[hypothesis[2]])
                if (premises, hypothesis) not in rule2conf_tg:
                    if len(premises) == 1:
                        new_rules.append((premises, hypothesis, conf))
                    else:
                        premises = (premises[1], premises[0])
                        if premises not in rule2conf_tg:
                            new_rules.append((premises, hypothesis, conf))
        return new_rules

    r2r = dict((relation_sr, relation_tg)
               for relation_sr, relation_tg in relation_seeds)
    new_rules_tg = _rule_transfer(rule2conf_sr, rule2conf_tg, r2r)
    r2r = dict((relation_tg, relation_sr)
               for relation_sr, relation_tg in relation_seeds)
    new_rules_sr = _rule_transfer(rule2conf_tg, rule2conf_sr, r2r)
    return new_rules_sr, new_rules_tg


def completion_by_aligned_entities(triples_sr, triples_tg, entity_seeds, relation_seeds):
    '''
    auto graph completion with (e1, r1, e2) -> (e1', r1', e2'), in which (e1, r1, e2) is in KG but (e1', r1', e2') is not in KG'
    triples_*: [(head, tail, relation)...]
    *_seeds: [(sr_id, tg_id)...]
    '''
    if not isinstance(triples_sr, list):
        print_time_info('sssssssssssss', dash_top=True)
        raise ValueError()
    if not isinstance(triples_tg, list):
        print_time_info('sssssssssssss', dash_top=True)
        raise ValueError()

    def _completion_by_aligned_entities(from_triples, to_triples, e2e, r2r):
        to_triples = set(to_triples)
        new_triples = []
        for head, tail, relation in from_triples:
            if head in e2e and tail in e2e and relation in r2r:
                to_triple_candidate = (e2e[head], e2e[tail], r2r[relation])
                if to_triple_candidate not in to_triples:
                    new_triples.append(to_triple_candidate)
        return new_triples

    e2e = dict(entity_seeds)
    _check(entity_seeds, e2e, 1)
    r2r = dict(relation_seeds)
    _check(relation_seeds, r2r, 2)

    new_triples_tg = _completion_by_aligned_entities(
        triples_sr, triples_tg, e2e, r2r)

    e2e = dict([(entity_tg, entity_sr)
                for entity_sr, entity_tg in entity_seeds])
    _check(entity_seeds, e2e, 3)
    r2r = dict([(relation_tg, relation_sr)
                for relation_sr, relation_tg in relation_seeds])
    _check(relation_seeds, r2r, 4)
    new_triples_sr = _completion_by_aligned_entities(
        triples_tg, triples_sr, e2e, r2r)
    new_triple_confs_sr = {triple: 1.0 for triple in new_triples_sr}
    new_triple_confs_tg = {triple: 1.0 for triple in new_triples_tg}
    return new_triple_confs_sr, new_triple_confs_tg


class CrossGraphCompletion(object):

    def __init__(self, directory, train_seeds_ratio):
        '''
        we followed the experiment setting of JAPE
        the folder under the directory JAPE/0_x contains the entity alignment dataset for train and test.
        '''
        if train_seeds_ratio not in {0.1, 0.2, 0.3, 0.4, 0.5}:
            print('----------------------------')
            print_time_info('Not a legal train seeds ratio: %f.' %
                            train_seeds_ratio)
            raise ValueError()

        self.directory = directory
        self.train_seeds_ratio = train_seeds_ratio
        language_sr, language_tg = directory.name.split('_')
        self.language_pair = {'sr': language_sr, 'tg': language_tg}

        self.entity_seeds = []
        self.relation_seeds = []
        self.triples_sr = []
        self.triples_tg = []
        self.new_triple_confs_sr = {}
        self.new_triple_confs_tg = {}
        self.rules_sr = []
        self.rules_tg = []
        self.rules_trans2_sr = []
        self.rules_trans2_tg = []
        self.id2entity_sr = {}
        self.id2entity_tg = {}
        self.id2relation_sr = {}
        self.id2relation_tg = {}

        # calculate the average PCA confidence of rules of which relation x is the tail
        # conf(r) = \frac{sum([pca\_conf | (premises, r, pca\_conf)\in rules])}{num((premises, r, pca\_conf)\in rules)}
        self.relation2conf_sr = {}
        self.relation2conf_tg = {}

        # calculate imp(r) = 1- min(\frac{num(head|(head, tail, r) \in triples)}{num(tail|(head, tail, r) \in triples)}, 1)
        self.relation2imp_sr = {}
        self.relation2imp_tg = {}

        self.triple_graph_sr = TripleGraph()
        self.triple_graph_tg = TripleGraph()

    def init(self):
        directory = self.directory

        # load from directory
        self.entity_seeds, self.relation_seeds = _load_seeds(
            directory, self.train_seeds_ratio)
        self.triples_sr, self.id2entity_sr, self.id2relation_sr, self.rules_sr = _load_languge(
            directory, self.language_pair['sr'])
        self.triples_tg, self.id2entity_tg, self.id2relation_tg, self.rules_tg = _load_languge(
            directory, self.language_pair['tg'])

        # print_time_info(self.language_pair)
        # print_time_info(len(self.rules_sr))
        # print_time_info(len(self.rules_tg))
        # return
        
        # completion_by_aligned_entities
        # new_triple_confs_sr, new_triple_confs_tg = completion_by_aligned_entities(
        #     self.triples_sr, self.triples_tg, self.entity_seeds, self.relation_seeds)
        # self.triples_sr += list(new_triple_confs_sr.keys())
        # self.triples_tg += list(new_triple_confs_tg.keys())
        # self.new_triple_confs_sr = dict_union(
        #     self.new_triple_confs_sr, new_triple_confs_sr)
        # self.new_triple_confs_tg = dict_union(
        #     self.new_triple_confs_tg, new_triple_confs_tg)
        # self._print_result_log({'sr': new_triple_confs_sr, 'tg': new_triple_confs_tg},
        #                        'completion_by_aligned_entities', 'triple')

        # rule transfer
        new_rules_sr, new_rules_tg = rule_transfer(
            self.rules_sr, self.rules_tg, self.relation_seeds)
        self.rules_sr += new_rules_sr
        self.rules_tg += new_rules_tg
        self.rules_trans2_sr += new_rules_sr
        self.rules_trans2_tg += new_rules_tg
        bi_new_rules = {'sr': new_rules_sr, 'tg': new_rules_tg}
        self._print_result_log(bi_new_rules, 'rule_transfer', 'rule')
        # _print_new_rules(bi_new_rules, self.id2relation_sr,
        #                  self.id2relation_tg)

        # get relation2conf
        self.relation2conf_sr = get_relation2conf(self.rules_sr)
        self.relation2conf_tg = get_relation2conf(self.rules_tg)
        print_time_info('sr r2conf num: ' + str(len(self.relation2conf_sr)) + ' average: ' + str(sum(self.relation2conf_sr.values())/len(self.relation2conf_sr)), dash_top=True)
        print_time_info('tg r2conf num: ' + str(len(self.relation2conf_tg)) + ' average: ' + str(sum(self.relation2conf_tg.values())/len(self.relation2conf_tg)), dash_top=True)

        # load triple into TripleGraph
        self.triple_graph_load(self.triples_sr, self.triples_tg)

        # rule_based_graph_completion
        new_triple_confs_sr, new_triple_confs_tg = rule_based_graph_completion(
            self.triple_graph_sr, self.triple_graph_tg,  self.rules_sr, self.rules_tg)
        self.triples_sr += list(new_triple_confs_sr.keys())
        self.triples_tg += list(new_triple_confs_tg.keys())
        self.new_triple_confs_sr = dict_union(
            self.new_triple_confs_sr, new_triple_confs_sr)
        self.new_triple_confs_tg = dict_union(
            self.new_triple_confs_tg, new_triple_confs_tg)
        bi_new_triple_confs = {
            'sr': new_triple_confs_sr, 'tg': new_triple_confs_tg}
        self._print_result_log(bi_new_triple_confs,
                               'rule_based_graph_completion', 'triple')
        # _print_new_triple_confs(bi_new_triple_confs, self.id2entity_sr,
        #                         self.id2entity_tg, self.id2relation_sr, self.id2relation_tg)

        # get relation2imp
        self.relation2imp_sr = get_relation2imp(self.triples_sr, len(self.id2relation_sr))
        self.relation2imp_tg = get_relation2imp(self.triples_tg, len(self.id2relation_tg))
        print_time_info('sr r2imp num: ' + str(len(self.relation2imp_sr)) + ' average: ' + str(sum(self.relation2imp_sr.values())/len(self.relation2imp_sr)), dash_top=True)
        print_time_info('tg r2imp num: ' + str(len(self.relation2imp_tg)) + ' average: ' + str(sum(self.relation2imp_tg.values())/len(self.relation2imp_tg)), dash_top=True)

    def triple_graph_load(self, triples_sr, triples_tg):
        self.triple_graph_sr.load(triples_sr)
        self.triple_graph_tg.load(triples_tg)

    def save(self, directory):
        if not directory.exists():
            directory.mkdir()
        save_path = directory / 'cgc.pkl'
        with open(save_path, 'wb') as f:
            pickle.dump(self, f)
        print_time_info('Successfully save cgc to %s.'%save_path)

    @classmethod
    def restore(cls, directory):
        load_path = directory / 'cgc.pkl'
        with open(load_path, 'rb') as f:
            new_one = pickle.load(f)
        print_time_info('Successfully loaded cgc from %s.'% load_path)
        return new_one

    def _print_result_log(self, bi_new_triples, method, data_name='triple'):
        print('------------------------------------------------------------')
        print_time_info('language_pair: ' +
                        '_'.join(self.language_pair.values()))
        print_time_info('Method: ' + method)
        for key, language in self.language_pair.items():
            print_time_info(
                language + ' new %s numbers: ' % data_name + str(len(bi_new_triples[key])))
        print('------------------------------------------------------------\n')
