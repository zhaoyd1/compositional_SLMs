# Compositional_SLMs

The repository of ["A Systematic Study of Compositional Syntactic Transformer Language Models"](https://aclanthology.org/2025.acl-long.350/).

Each model is named with four binary code like 0110. The meaning of each 0/1 is presented in order in the table below:

|Aspect|0|1|
|---|---|---|
|Linearization Method|Bottom-up|Top-down|
|Composition Function|External|Internal|
|Tree Binarization|Binary|Non-binary|
|Sub-constituent Masking|No masking|Masking|


The scripts for training and evaluation of different variants are in the ``scripts/`` folder. 

You should follow [GPST](https://github.com/ant-research/StructuredLM_RTDT) to proprocess the data, set up the environment, and then run the training and evaluation. 
There is also an example script of ``preprocess_corpus.sh``. You should first carefully relate the python code (``utils/dataset_builder.py``) to example data file in ``corpus/bllip/`` folder before using the script, which help you set the correct parameters. 
``Note that due to the unalignment in the training, for models xx0x, tokenization is conducted with No prefix space before each sentence. But for models xx1x, there is a prefix space before each sentence. You can refer the example data for better understanding of the difference.``

Due to the large volume and complexity of code in this project, there may be some missing parts in the repository. If you have any questions, please feel free to contact me.




