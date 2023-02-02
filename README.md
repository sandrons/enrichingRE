# enrichingRE


This repository contains the code for the task of Relation Extraction (RE) augmented with Open Information Extraction (OpenIE). We examine the impact of recent OpenIE techniques on the task of relation extraction (RE) by converting information about sentence components like subjects, objects, verbs, and adverbials into vector representations. Our hypothesis is that breaking down complex sentences into smaller parts through OpenIE enhances the performance of context-aware language models like BERT in RE. Results from testing on two annotated datasets, KnowledgeNet and FewRel, show improved accuracy with our enriched models compared to current RE methods. Our best scores were 92% F1 on KnowledgeNet and 71% F1 on FewRel, demonstrating the effectiveness of our approach on these benchmark datasets. Both datasets are publicly available, KnowledgeNet at https://raw.githubusercontent.com/diffbot/knowledge-net/master/dataset/train.json and FewRel at https://github.com/fractalego/fewrel_zero_shot/tree/master/data
At the moment, a snippet of the code is available for reproducibility purposes. The train and test sets (i.e. the preprocessed datasets broken down into clauses for fine-tuning) are released under the folder data. Those are small subsets of the original data. In terms of reproducibility, we used OpenIE5 https://github.com/dair-iitd/OpenIE-standalone to break down the sentences into clauses for KnowledgeNet and we used OpenIE6 https://github.com/dair-iitd/openie6 for FewRel.   


# Run the code

When running the code on your local machine, be sure to change the path to the data folder accordingly. The list of requirements is located inside the folder src. To run the code is sufficient the script run.sh 
