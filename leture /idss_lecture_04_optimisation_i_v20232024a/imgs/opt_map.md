graph LR;


Optimization --> Discrete;
Discrete --> ip[Integer programming];
Discrete --> comb[Combinatorial optimization];
comb --> cons[Constraint programming];
Optimization --> Continuous;

Continuous --> Convex[Convex: mathematical];
Convex --> lp[Linear programming];
Convex --> sqp[Semi-Quadratic programming];
Convex --> qp[Quadratic programming];
Convex --> lq[Least squares];

Continuous --> Nonconvex[Non-convex: iterative];
Nonconvex --> nc0[Zeroth order];
nc0 --> rs[Random search];
nc0 --> nmead[Nelder-Mead];
rs--> mrs[Random search w/memory];
mrs --> hc[Hill climbing];
mrs--> prs[w/population];
prs --> ga[Genetic algorithms];

mrs --> sa[Simulated annealing];
prs --> aco[Ant colony optimisation]
mrs --> tabu[Tabu search];
prs --> swarm[Swarm search];

Nonconvex --> nc1[First order];
nc1 --> gd[Gradient descent];
nc1 --> co[Coordinate descent];
gd --> cgd[Conjugate gradient descent];
gd --> sgd[Stochastic gradient descent]
Nonconvex --> nc2[Second order];
nc2 --> quasi-Newton;
nc2 --> L-BFGS;

classDef br fill:#eee,stroke:#333,stroke-width:1px;

class Convex,lp,qp,sqp,lq,Discrete,ip,comb,cons,tabu,aco,swarm,co,L-BFGS,cgd br;

