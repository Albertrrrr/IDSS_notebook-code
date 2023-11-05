graph LR

Evidence_1 --> BayesRule_1;
BayesRule_1((Bayes' Rule)) --> Posterior_1;
Prior_1-->BayesRule_1;
Posterior_1 --> BayesRule_2;

BayesRule_2((Bayes' Rule)) --> Posterior_2;
Evidence_2 --> BayesRule_2;


BayesRule_3((Bayes' Rule)) --> Posterior_3;
Posterior_2 --> BayesRule_3;
Evidence_3 --> Posterior_3;

Posterior_3 -.-> r[...];

 classDef br fill:#fe8,stroke:#333,stroke-width:4px;

class BayesRule_1,BayesRule_2,BayesRule_3 br;


