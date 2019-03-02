from .kg_path import kg_data_dir
from .format_dbp15k import _dump_mapping, _dump_seeds, _dump_triples

def format_dwy100k():
    def _read_id(directory):
        with open(directory / 'ent_ids_1', 'r', encoding='utf8') as f:
            lines = [line.strip().split('\t') for line in f.readlines()]
            id2ent_sr = dict(lines)
        with open(directory / 'ent_ids_2', 'r', encoding='utf8') as f:
            lines = [line.strip().split('\t') for line in f.readlines()]
            id2ent_tg = dict(lines)
        with open(directory / 'rel_ids_1', 'r', encoding='utf8') as f:
            lines = [line.strip().split('\t') for line in f.readlines()]
            id2rel_sr = dict(lines)
        with open(directory / 'rel_ids_2', 'r', encoding='utf8') as f:
            lines = [line.strip().split('\t') for line in f.readlines()]
            id2rel_tg = dict(lines)
        return id2ent_sr, id2ent_tg, id2rel_sr, id2rel_tg

    def _read_triples(directory, id2ent_sr, id2ent_tg, id2rel_sr, id2rel_tg):
        with open(directory / 'triples_1', 'r', encoding='utf8') as f:
            lines = [line.strip().split('\t') for line in f.readlines()]
            triples_sr = [(id2ent_sr[h], id2ent_sr[t], id2rel_sr[r]) for h, r, t in lines]
        with open(directory / 'triples_2', 'r', encoding='utf8') as f:
            lines = [line.strip().split('\t') for line in f.readlines()]
            triples_tg = [(id2ent_tg[h], id2ent_tg[t], id2rel_tg[r]) for h, r, t in lines]
        return triples_sr, triples_tg

    def _read_align(directory, id2ent_sr, id2ent_tg):
        with open(directory / 'sup_ent_ids', 'r', encoding='utf8') as f:
            lines = [line.strip().split('\t') for line in f.readlines()]
            sup_align = [(id2ent_sr[sr], id2ent_tg[tg]) for sr, tg in lines]
        with open(directory / 'ref_ent_ids', 'r', encoding='utf8') as f:
            lines = [line.strip().split('\t') for line in f.readlines()]
            ref_align = [(id2ent_sr[sr], id2ent_tg[tg]) for sr, tg in lines]
        return sup_align, ref_align

    def _read_relation_align(directory):
        with open(directory / 'rel_aligned', 'r', encoding='utf8') as f:
            rel_align = [line.strip().split('\t') for line in f.readlines()]
            import numpy as np
            print(np.asarray(rel_align).shape, '-------------------------------------------------------')
            return rel_align

    # def _construct_align_for_
    dwy_dir = kg_data_dir / 'DWY100k'
    dwy_dir_oris = dwy_dir.glob('*')
    from project_path import bin_dir
    bin_dir = bin_dir / 'DWY100k'
    if not bin_dir.exists:
        bin_dir.mkdir()

    for dir_ori in dwy_dir_oris:
        sr_name, tg_name = dir_ori.name.split('_')
        local_bin = bin_dir / dir_ori.name
        if not local_bin.exists:
            local_bin.mkdir()
        dir_ori = dir_ori / 'mapping' / '0_3'
        id2ent_sr, id2ent_tg, id2rel_sr, id2rel_tg = _read_id(dir_ori)
        triples_sr, triples_tg = _read_triples(dir_ori, id2ent_sr, id2ent_tg, id2rel_sr, id2rel_tg)
        sup_align, ref_align = _read_align(dir_ori, id2ent_sr, id2ent_tg)
        rel_align = _read_relation_align(dir_ori)
        ent_sr = set(id2ent_sr.values())
        ent_tg = set(id2ent_tg.values())
        rel_sr = set(id2rel_sr.values())
        rel_tg = set(id2rel_tg.values())
        assert len(ent_sr) == len(id2ent_sr)
        assert len(ent_tg) == len(id2ent_tg)
        assert len(rel_sr) == len(id2rel_sr)
        assert len(rel_tg) == len(id2rel_tg)
        ent2id_sr = {ent: i for i, ent in enumerate(ent_sr)}
        ent2id_tg = {ent: i for i, ent in enumerate(ent_tg)}
        rel2id_sr = {rel: i for i, rel in enumerate(rel_sr)}
        rel2id_tg = {rel: i for i, rel in enumerate(rel_tg)}
        _dump_mapping(ent2id_sr, 'entity2id_' + sr_name, local_bin)
        _dump_mapping(ent2id_tg, 'entity2id_' + tg_name, local_bin)
        _dump_mapping(rel2id_sr, 'relation2id_' + sr_name, local_bin)
        _dump_mapping(rel2id_tg, 'relation2id_' + tg_name, local_bin)
        sup_align = [(ent2id_sr[sr], ent2id_tg[tg]) for sr, tg in sup_align]
        ref_align = [(ent2id_sr[sr], ent2id_tg[tg]) for sr, tg in ref_align]
        rel_align = [(rel2id_sr[sr], rel2id_tg[tg]) for sr, tg in rel_align]
        _dump_seeds(sup_align, 'train_entity', local_bin)
        _dump_seeds(ref_align, 'test_entity', local_bin)
        _dump_seeds(rel_align, 'train_relation', local_bin)


        all2id_sr = {}
        for ent in ent2id_sr:
            all2id_sr[ent] = len(all2id_sr)
        for rel in rel2id_sr:
            all2id_sr[rel] = len(all2id_sr)
        all2id_tg = {}
        for ent in ent2id_tg:
            all2id_tg[ent] = len(all2id_tg)
        for rel in rel2id_tg:
            all2id_tg[rel] = len(all2id_tg)
        amie_triples_sr = [(all2id_sr[h], all2id_sr[r], all2id_sr[t]) for h, t, r in triples_sr]
        amie_triples_tg = [(all2id_tg[h], all2id_tg[r], all2id_tg[t]) for h, t, r in triples_tg]
        amie_triples_sr = [["<" + str(item) + ">" for item in triple] for triple in amie_triples_sr]
        amie_triples_tg = [["<" + str(item) + ">" for item in triple] for triple in amie_triples_tg]


        triples_sr = [(ent2id_sr[h], ent2id_sr[t], rel2id_sr[r]) for h, t, r in triples_sr]
        triples_tg = [(ent2id_tg[h], ent2id_tg[t], rel2id_tg[r]) for h, t, r in triples_tg]
        _dump_triples(triples_sr, 'triples_' + sr_name, local_bin)
        _dump_triples(triples_tg, 'triples_' + sr_name, local_bin)


        AMIE_dir = local_bin / 'AMIE'
        AMIE_dir.mkdir()
        _dump_triples(amie_triples_sr, 'triples_' + sr_name, AMIE_dir)
        _dump_triples(amie_triples_tg, 'triples_' + tg_name, AMIE_dir)
        _dump_mapping(all2id_sr, 'all2id_' + sr_name, AMIE_dir)
        _dump_mapping(all2id_tg, 'all2id_' + tg_name, AMIE_dir)