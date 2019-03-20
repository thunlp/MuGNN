# Multi-Channel Graph Neural Network for Entity Alignment

This is the code of the Multi-Channel Graph Neural Network for Entity Alignment in XXX. The details of this model can be found from [paper link](www.google.com)

## Environment

* python 3.6
* [PyTorch 1.0](https://pytorch.org/get-started/locally/)
* [rdflib 4.2.2](https://pypi.org/project/rdflib/)
* [tensorboardx 1.6](https://pypi.org/project/tensorboardX/)

Note: tensorboardX needs a TensorFlow to work correctly.

## Run the code

To run a demo, you can execute the following script:

```bash
python example_train.py [GPU_id (if available)]
```

## Config file

## Datasets

我们的project中包含了DBP15K和DWY100K数据集，他们分别被整理为以下形式

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

DBP15k和DWY100k数据整理的区别在于，DWY100k预先分好了实体对齐的测试和训练集，而DBP15k没有

假如你需要在自己的数据集上跑我们的代码，you might want to把自己的数据整理为相似的格式，以下是对各个文件内容的说明：

* entity2id_kgx.txt: all entities from kgx with the corresponding ids. 
  Format: **entity_name** + \t + **entity_id** + \n
* relation2id_kgx.txt: all relations from kgx with the corresponding ids. One relation per line, \t separated.
* triples

## Reference