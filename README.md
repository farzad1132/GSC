# GSC: Generalizable Service Coordination

Official implementation of **[GSC: Generalizable Service Coordination](https://arxiv.org/abs/2311.02657)**. GSC performs coordination of services consisting of inter-dependent components in multi-cloud networks. Service coordination comprises the placement and scalability of components and scheduling incoming traffic requests for services between deployed instances.

## Setup
This code has been tested using **Python 3.8.6**. We recommend installing the required packages (specified by ***requirements.txt*** file) inside a **virtual environment**. 

## Usage
GSC can be configured using these config files:
- agent config: specifies the configuration of the DRL agent, including parameters of the GNN Embedder and DRL training.
- network config: specifies the network topology and features of entities (nodes and links).
- service config: specifies the service's characteristics, such as processing delay of functions.
- simulator config: specifies the traffic generation pattern and its configuration.
- scheduler config: Determines how aspects of the network should change during the training phase.

## Citation
```
@misc{mohammadi2023gsc,
      title={GSC: Generalizable Service Coordination}, 
      author={Farzad Mohammadi and Vahid Shah-Mansouri},
      year={2023},
      eprint={2311.02657},
      archivePrefix={arXiv},
      primaryClass={cs.DC}
}
```

## Acknowledgement
We sincerely thank the developers of the [DeepCoord](https://github.com/RealVNF/DeepCoord) and [coord-sim](https://github.com/RealVNF/coord-sim) projects for making the source code of their efforts available, which immensely helped us in the development of GSC.
