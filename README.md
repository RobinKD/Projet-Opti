# Research paper associated with the project

https://www.tik.ee.ethz.ch/file/e8e3a4750b11117c05b4e00ac846c6ef/ZK2004a.pdf

# Bit of example code for function evaluation in biobj case

It should be as simple as calling the example experiment with 'bbob-biobj' as suite name. The call to 'fun' should return a vector with two entries in this case (the two objective vectors). Here is the most basic example to evaluate the biobjective bbob function 17 in dimension 5 (instance 11):

    In [1]:  import cocoex
             suite = cocoex.Suite('bbob-biobj', '', '')
             fun = suite.get_problem_by_function_dimension_instance(17, 5, 11)
            fun([0, 1, 0.5, 0.7, 0.3])
    Out [1]: array([ 1.63665638e+06, -1.58054979e+02])


I hope this helps to clarify that the biobjective BBOB problems are indeed returning vectors with the two objective function values. BTW, using a question-mark ("?") behind a command as well as tab-completion will typically help to understand the classes of the COCO code. For example on the above fun, I get:

    In [2]:  type(fun)
    Out [2]: cocoex.interface.Problem 

    In [3]: fun?
    Type:           Problem
    String form:    bbob-biobj_f17_i11_d05: a 5-dimensional bi-objective problem (problem 1900 of suite "bbob-biobj" with name "bbob_f002_i23_d05__bbob_f017_i24_d05")
    File:           c:\users\dimo\appdata\local\python-eggs\python-eggs\cache\cocoex-0.0.0-py2.7-win32.egg-tmp\cocoex\interface.pyd
    Docstring:     
    `Problem` instances are usually generated using `Suite`.

The main feature of a problem instance is that it is callable, returning the
objective function value when called with a candidate solution as input.
Call docstring: return objective function value of input `x`

# Project instructions

http://www.cmap.polytechnique.fr/~dimo.brockhoff/optimizationSaclay/2017/groupproject.php

Content of report (based on latex paper template for biobj /!\):
Recommended content of the paper:
- Abstract
- Introduction (giving the background and motivation for the work)
- Description of the algorithm, including pseudocode
- Description of the implementation/used parameters/stopping criteria
- Information about the timing experiment
- Discussion of the results
- Conclusions (including a discussion about limitations and potential improvements of the algorithm, anomalies observed, exceptional performance (bad or good), ...)

# Proposed LaTeX package for pseudo code

algorithmicx, see https://tex.stackexchange.com/questions/163768/write-pseudo-code-in-latex
