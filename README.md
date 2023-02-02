# enrichingRE


This repository contains the code for the task of Relation Extraction (RE) augmented with Open Information Extraction (OpenIE). We examine the impact of recent OpenIE techniques on the task of relation extraction (RE) by converting information about sentence components like subjects, objects, verbs, and adverbials into vector representations. Our hypothesis is that breaking down complex sentences into smaller parts through OpenIE enhances the performance of context-aware language models like BERT in RE. Results from testing on two annotated datasets, KnowledgeNet and FewRel, show improved accuracy with our enriched models compared to current RE methods. Our best scores were 92% F1 on KnowledgeNet and 71% F1 on FewRel, demonstrating the effectiveness of our approach on these benchmark datasets.
At the moment, a snippet of the code is available for reproducibility purposes. Our data (i.e. the preprocessed datasets broken down into clauses) will be available soon. 
