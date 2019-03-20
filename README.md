# Multi-Channel Graph Neural Network for Entity Alignment

This is the code of the *"Multi-Channel Graph Neural Network for Entity Alignment"* in XXX. The details of this model could be found at [paper link](www.google.com).

## Dependencies

* python 3.6
* [PyTorch 1.0](https://pytorch.org/get-started/locally/)
* [rdflib 4.2.2](https://pypi.org/project/rdflib/)
* [tensorboardx 1.6](https://pypi.org/project/tensorboardX/)

*Note: tensorboardX needs a TensorFlow installation to work correctly.*

## Datasets

Folder ./bin contains [DBP15k](https://github.com/nju-websoft/JAPE) and [DWY100k](https://github.com/nju-websoft/BootEA) datasets.

### Directory structure

      DBP15k/
            kg1_kg2/
                  entity2id_kg1.txt
                  entity2id_kg2.txt
                  relation2id_kg1.txt
                  relation2id_kg2.txt
                  triples_kg1.txt
                  triples_kg2.txt
                  relation_seeds.txt
                  entity_seeds.txt
                  AMIE/
                        all2id_kg1.txt
                        all2id_kg2.txt
                        triples_kg1.txt
                        triples_kg2.txt
      DWY100k/
            kg1_kg2/
                  entity2id_kg1.txt
                  entity2id_kg2.txt
                  relation2id_kg1.txt
                  relation2id_kg2.txt
                  triples_kg1.txt
                  triples_kg2.txt
                  relation_seeds.txt
                  train_entity_seeds.txt
                  test_entity_seeds.txt
                  AMIE/
                        all2id_kg1.txt
                        all2id_kg2.txt
                        triples_kg1.txt
                        triples_kg2.txt

### Data format

* entity2id_kgx.txt: all entities from kgx with the corresponding ids. Format: ***entity_name*** + \t + ***id*** + \n;
* relation2id_kgx.txt: all relations from kgx with the corresponding ids. Format: ***relation_name*** + \t + ***id*** + \n;
* triples_kgx.txt: all triples from kgx. Format: ***entity1*** + \t + ***entity2*** + \t + ***relation*** + \n;
* entity_seeds.txt: all entity seed alignments. Format: ***entity1*** (from kg1) + \t + ***entity2*** (from kg2) + \n;
* train_entity_seeds.txt: entity seed alignments for training. Format: ***entity1*** (from kg1) + \t + ***entity2*** (from kg2) + \n;
* test_entity_seeds.txt: entity seed alignments for test. Format: ***entity1*** (from kg1) + \t + ***entity2*** (from kg2) + \n;
* relation_seeds.txt: all relation seed alignments. Format: ***relation1*** (from kg1) + \t + ***relation2*** (from kg2) + \n;
* all2id_kgx.txt: all entities and relations from kgx with the corresponding ids. Format: ***entity/relation*** + \t + id + \n.

### Note

* The difference between arrangements of DBP15k and DWY100k is that DWY100k has split the train and test set of entity alignments but DBP15k has not;
* Folder AMIE contains data arranged in the structure which is designed to be compatible with [AMIE+](https://www.mpi-inf.mpg.de/departments/databases-and-information-systems/research/yago-naga/amie/);
* triples_kgx.txt in folder AMIE is encoded with all2id_kgx.txt.

## Code

### Demo

To run a demo, simply execute the following script:

```bash
>> python example_train.py [GPU_id (if available)]
# example
>> python example_train.py 0
```

### Customized running

To run the code on your own dataset:

1. Format your data as described in [Datasets](#Datasets);
2. Execute rule mining with AMIE+;

      ```bash
      >> python format_data.py [PATH_TO_YOUR_DATASET]
      # example
      >> python format_data.py ./bin/DBP15k/fr_en
      ```

      *Note: AMIE+ runs as an independent JAVA program. So you will need to wait until AMIE+ ended, and then input "amie ended" at the prompt to inform the python program to execute the next step.*

3. Customize your running;

   * Customization with config.py

      ```python
      from config import Config
      config = Config()
      ```

   * Set the hyper-parameters;

      ```python
      config.set_cuda(True) # set train on cpu or gpu
      config.set_dim(128) # set dimension number of embeddings and weight matrices
      config.set_align_gamma(1.0) # set gamma_1 and gamma_2
      config.set_rule_gamma(0.12) # set gamma_r
      config.set_num_layer(2) # set layer number of MuGNN
      config.set_dropout(0.2) # set dropout rate
      config.set_learning_rate(0.001) # set learning rate
      config.set_l2_penalty(1e-2) # set L2 regularization coefficient
      config.set_update_cycle(5) # set negative sampling frequency
      ```

   * Set your dataset path;

      ```python
      config.init(YOUR_DATASET_PATH)
      # example
      config.init('./bin/DBP15k/fr_en')
      ```

   * Set log path;

      ```python
      config.init_log(LOG_FILE_PATH)
      # example
      config.init_log('./log/test')
      ```

   * Train;

      ```python
      config.train()
      ```

If you have any difficulties and questions while running the code, feel free to create an issue or contact us directly at achark@outlook.com.

## Reference

If you use code, please cite our [paper](www.google.com):