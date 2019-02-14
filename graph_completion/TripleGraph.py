import rdflib


class TripleGraph(object):
    def __init__(self):
        self.graph = rdflib.Graph()
        self.prefix_query = '\n'.join([
            'BASE <http://www.entity.com#>'
            'PREFIX relation: <http://www.relation.org#>'
        ])
        self.prefix_load = '\n'.join([
            '@base <http://www.entity.com#> .'
            '@prefix relation: <http://www.relation.org#> .'
        ])

    def load(self, triples):
        data = self.prefix_load
        for head, tail, relation in triples:
            data += ('<' + head + '> ' + 'relation:' +
                     str(relation) + '<' + tail + '> .\n')
        self.graph.parse(data=data, format='turtle')

    def query(self, query):
        return list(self.graph.query(self.prefix_query + query))

    def inference_by_rule(self, rule):
        query = ''
        premises, hypothesis, conf = rule
        inferred_relation = hypothesis[2]
        for head, tail, relation in premises:
            query += '?' + head + ' relation:' + relation + ' ?' + tail + ' .\n'
        query = 'select distinct ?a ?b where {%s}' % query
        query_result = self.query(query)
        new_triple_confs = []
        for a, b in query_result:
            a = a.split('/')[-1]
            b = b.split('/')[-1]
            new_triple_confs.append(((a, b, inferred_relation), conf))
        return new_triple_confs
