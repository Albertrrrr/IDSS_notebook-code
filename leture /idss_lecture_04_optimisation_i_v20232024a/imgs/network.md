graph LR;

X("X: input") --> W1("W[0]x");

W1 --> G1("G(x)");
G1 --> W2("W[1]x");

W2 --> G2("G(x)");

G2 --> W3("W[2]x");
W3 --> G3("G(x)");

G3 --> Y'("Y': Prediction");

Y("Y: True")  --> Loss;
Y' --> Loss("L(Y, Y')");

classDef nonlinear fill:#ff3,stroke:#333,stroke-width:4px;
classDef linear fill:#ddd,stroke:#333,stroke-width:1px;
classDef loss fill:#fdd,stroke:#333,stroke-width:1px;

class G1,G2,G3 nonlinear;
class W1,W2,W3 linear;
class Loss loss;

