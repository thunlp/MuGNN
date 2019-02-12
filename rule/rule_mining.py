
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
    maxad = 2
    jar_patch_path = executable_dir / 'amie_plus.jar'
    command = 'java -jar %s -maxad %d -minpca %f %s > %s &' % (
        jar_patch_path, maxad, minpca, path2triples, path2rules)
    
    res = subprocess.call(command, shell=True)
    print(res)
    

def mine_rule_for_dbp15k():
    from project_path import bin_dir
    bin_dir = bin_dir / 'dbp15k'
    language_pair_dirs = list(bin_dir.glob('*_en'))
    for directory in language_pair_dirs:
        local_bin_dir = directory / 'AMIE'
        file_paths = local_bin_dir.glob('triples_*.txt')
        for file_path in file_paths:
            file_name = file_path.name
            output_path = local_bin_dir / ('rule_for_' + file_name)
            mine_rule_with_amie(file_path, output_path)


def main():
    from project_path import bin_dir
    # path = bin_dir /
    mine_rule_for_dbp15k()
