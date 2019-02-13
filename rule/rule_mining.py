import re, json
from tools.print_time_info import print_time_info

# usage of AMIE+ executable file
'''
usage: java -jar amie_plus.jar [OPTIONS] <TSV FILES>

 -auta                             Avoid unbound type atoms, e.g., type(x,
                                   y), i.e., bind always 'y' to a type
 -bexr <body-excluded-relations>   Do not use these relations as atoms in
                                   the body of rules. Example:
                                   <livesIn>,<bornIn>
 -bias <e-name>                    Syntatic/semantic bias:
                                   oneVar|default|[Path to a subclass of
                                   amie.mining.assistant.MiningAssistant]D
                                   efault: default (defines support and
                                   confidence in terms of 2 head
                                   variables)
 -btr <body-target-relations>      Allow only these relations in the body.
                                   Provide a list of relation names
                                   separated by commas (incompatible with
                                   body-excluded-relations). Example:
                                   <livesIn>,<bornIn>
 -caos                             If a single variable bias is used
                                   (oneVar), force to count support always
                                   on the subject position.
 -const                            Enable rules with constants. Default:
                                   false
 -deml                             Do not exploit max length for speedup
                                   (requested by the reviewers of AMIE+).
                                   False by default.
 -dpr                              Disable perfect rules.
 -dqrw                             Disable query rewriting and caching.
 -ef <extraFile>                   An additional text file whose
                                   interpretation depends on the selected
                                   mining assistant (bias)
 -fconst                           Enforce constants in all atoms.
                                   Default: false
 -full                             It enables all enhancements: lossless
                                   heuristics and confidence approximation
                                   and upper bounds It overrides any other
                                   configuration that is incompatible.
 -hexr <head-excluded-relations>   Do not use these relations as atoms in
                                   the head of rules (incompatible with
                                   head-target-relations). Example:
                                   <livesIn>,<bornIn>
 -htr <head-target-relations>      Mine only rules with these relations in
                                   the head. Provide a list of relation
                                   names separated by commas (incompatible
                                   with head-excluded-relations). Example:
                                   <livesIn>,<bornIn>
 -maxad <max-depth>                Maximum number of atoms in the
                                   antecedent and succedent of rules.
                                   Default: 3
 -minc <min-std-confidence>        Minimum standard confidence threshold.
                                   This value is not used for pruning,
                                   only for filtering of the results.
                                   Default: 0.0
 -minhc <min-head-coverage>        Minimum head coverage. Default: 0.01
 -minis <min-initial-support>      Minimum size of the relations to be
                                   considered as head relations. Default:
                                   100 (facts or entities depending on the
                                   bias)
 -minpca <min-pca-confidence>      Minimum PCA confidence threshold. This
                                   value is not used for pruning, only for
                                   filtering of the results. Default: 0.0
 -mins <min-support>               Minimum absolute support. Default: 100
                                   positive examples
 -mt <mining-technique>            AMIE offers 2 multi-threading
                                   strategies: standard (traditional) and
                                   solidary (experimental)
 -nc <n-threads>                   Preferred number of cores. Round down
                                   to the actual number of cores in the
                                   system if a higher value is provided.
 -oout                             If enabled, it activates only the
                                   output enhacements, that is, the
                                   confidence approximation and upper
                                   bounds.  It overrides any other
                                   configuration that is incompatible.
 -optimcb                          Enable the calculation of confidence
                                   upper bounds to prune rules.
 -optimfh                          Enable functionality heuristic to
                                   identify potential low confident rules
                                   for pruning.
 -oute                             Print the rules at the end and not
                                   while they are discovered. Default:
                                   false
 -pm <pruning-metric>              Metric used for pruning of intermediate
                                   queries: support|headcoverage. Default:
                                   headcoverage
 -rl <recursivity-limit>           Recursivity limit
 -verbose                          Maximal verbosity
'''


def mine_rule_with_amie(path2triples, path2rules):
    '''
    '''
    import subprocess
    from project_path import executable_dir
    minpca = 0.8
    maxad = 3
    num_process = 2
    jar_patch_path = executable_dir / 'amie_plus.jar'
    command = 'java -jar %s -maxad %d -minpca %f -nc %d %s > %s &' % (
        jar_patch_path, maxad, minpca, num_process, path2triples, path2rules)
    res = subprocess.call(command, shell=True)
    if res == 0:
        print_time_info('Mining started.')
    else:
        print_time_info('Something went wrong.')


def rule_parser(file_path):
    '''
    Accept the output of an AMIE+ .jar software and transform to ...
    '''
    atom_regex = re.compile(
        r'\?([a-z])  <([0-9]*?)>  \?([a-z])')

    def atom_parser(string):
        atoms = []
        for atom in atom_regex.finditer(string):
            # (head, tail, relation)
            atoms.append((atom.group(1), atom.group(3), atom.group(2)))
        if not atoms:
            print('-------------------------')
            print_time_info(string)
            raise ValueError('Parse atom failed.')
        return atoms

    with open(file_path, 'r', encoding='utf8') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines if line[0] == '?']
        rule_confs = [(lambda x: (x[0], float(x[3])))(
            line.split('\t')) for line in lines]
    
    rules = []
    for rule, confs in rule_confs:
        premises, hypothesis = rule.split('=>')
        premises = atom_parser(premises)
        hypothesis = atom_parser(hypothesis)
        if not len(hypothesis) == 1:
            print('-------------------------')
            print_time_info(rule)
            raise ValueError('Parse rule failed.')
        rules.append((premises, hypothesis))
        # premises, hypothesis
    return rules

def parse_and_dump_rules(read_path, dump_path, mapping):
    rules = rule_parser(read_path)
    with open(dump_path, 'w', encoding='utf8') as f:
        for premises, hypothesis in rules:
            premises = [(head, tail, mapping[relation]) for head, tail, relation in premises]
            hypothesis = [(head, tail, mapping[relation]) for head, tail, relation in hypothesis]
            f.write(json.dumps((premises, hypothesis), ensure_ascii=False) + '\n')