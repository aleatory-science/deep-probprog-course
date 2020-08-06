## Syllabus

### TODOs
* Develop and Solve new Exercises for Bayesian Regression
* Find Dataset and Solve Exercises for Hidden Markov Models
* Solve Exercises for Deep Generative Models
* Summarize the BDA exercises in a worksheet, with correct attribution.
* Summarize the NUTS paper in 3-pages to make it easier to read
* Update industrial outreach on Aleatory Science website to have company logos instead of table and directly point to this page instead.

### Part I - Introduction to Deep Probabilistic Programming

| Week | Topic | Exercise | Links |
|:----:|:-----:|:--------:|:-----:|
| 1    | Introduction to Bayesian Inference | Read Pattern Recognition and Machine Learning (PRML), Chapters 1 & 2, Intro to Bayesian updating paper, and Pyro paper. <br /> <br /> Form up groups and ask a question for each chapter/paper you have read. | [Pattern Recognition and Machine Learning](https://www.microsoft.com/en-us/research/publication/pattern-recognition-machine-learning/)  <br /> <br /> [Bayesian Updating Paper](https://arxiv.org/pdf/1411.5018.pdf) <br /> <br />[Pyro Paper](http://jmlr.org/papers/volume20/18-403/18-403.pdf) |
| 2  | Variational Inference | Read the Variational Inference paper, and ask three questions about it. <br /> <br /> Use Pyro’s Variational Inference support to implement the Bayesian Data Analysis kidney cancer model (Fig. 2.8), and the BDA hierarchical 2008 election poll model (Exc. 2.21) <br /> <br /> **Hint:** Instead of having individual parameters for the Gamma distribution in the guide, it is possible to make the arguments functions based on the observed death count `yj` and population `nj` like so: <br /> ![Equation](images/Tex2Img_1593179365.png) <br /> Here `k`, `w`, `c` and `v` are global parameters. This makes inference orders of magnitude more effective! Remember, to initialize the parameters to arbitrary values, to see training happening. | [Variational Inference Paper](https://arxiv.org/pdf/1601.00670.pdf) <br /> <br /> [Bayesian Data Analysis](http://www.stat.columbia.edu/~gelman/book/) <br /> <br /> [Pyro Website](https://pyro.ai) |
| 3 | Hamiltonian Monte Carlo | Read papers on Hamiltonian Monte Carlo and No U-Turn Sampler. Look at the source code for Mini-MC. <br /> <br /> Read on how enumerated inference can be used to marginalize latent variables using Pyro mixture model tutorial. <br /> <br /> Ask a question each for HMC, the Mini-MC implementation and discrete variable marginalization. <br /> <br /> Implement Bayesian Change-point model in Pyro using NUTS. | [Hamiltonian Monte Carlo Paper](https://arxiv.org/pdf/1701.02434.pdf) <br /> <br /> [No U-Turn Sampler](http://jmlr.csail.mit.edu/papers/volume15/hoffman14a/hoffman14a.pdf) <br /> <br /> [Mini-MC implementation](https://github.com/ColCarroll/minimc)<br /> <br /> [Change-point model](https://cscherrer.github.io/post/bayesian-changepoint/) |
| 4 | Hidden Markov Models and Discrete Variables. | Read Paper on Hidden Markov Models and ask three questions about it. <br /> <br /> Read Pyro tutorials on Discrete Variables and Gaussian Mixture Models. <br /> <br /> Read Pyro Hidden Markov Model Code Example. Adapt the Hidden Markov Model to work on custom dataset. <br /> <br /> **TODO** Find a dataset. | [Hidden Markov Models](http://ai.stanford.edu/~pabbeel/depth_qual/Rabiner_Juang_hmms.pdf) <br /> <br /> [Pyro Discrete Variables Tutorial](https://pyro.ai/examples/enumeration.html) <br /> <br />  [Pyro Gaussian Mixture Model Tutorial](https://pyro.ai/examples/gmm.html)   <br /> <br />  |
| 5 |  Bayesian Regression Models | Read PRML Chapter 3 on Linear Models. <br /> <br /> Ask 3 questions about the chapter. <br /> <br /> Read the Pyro tutorials on Bayesian Regression. <br /> <br /> **TODO** Exercises | Pyro Bayesian Regression: [Part I](https://pyro.ai/examples/bayesian_regression.html), [Part II](https://pyro.ai/examples/bayesian_regression_ii.html) |
| 6 | Variational Auto-Encoders | Read Variational Auto Encoders (VAE) foundations Chapters 1 & 2, and Pyro tutorial on VAE. Ask three questions about the paper and tutorial. <br /> <br /> Implement Frey Faces model from VAE paper in Pyro. Rely on the existing VAE implementation (see tutorial link). | [Variational Auto Encoders Foundations](https://arxiv.org/abs/1906.02691) <br /> <br /> [Pyro Tutorial on VAE](https://pyro.ai/examples/vae.html) |
| 7 | Deep Generative Models | Read Interpretable Representation VAE, Causal Effect VAE, and DRAW papers. Ask three questions about the papers. <br/> <br/> Implement Interpretable Representation VAE in Pyro. <br /> <br /> Try out the Causal Effect VAE Implementation in Pyro on a suitable dataset. | [Interpretable Representation VAE](https://arxiv.org/abs/1611.07492) <br /> <br /> [Causal Effect VAE](https://arxiv.org/abs/1705.08821) <br /> <br /> [Causal Effect VAE Pyro Implementation](http://docs.pyro.ai/en/stable/contrib.cevae.html) <br /> <br /> [DRAW](https://arxiv.org/abs/1502.04623) |


### Part II - Deep Probabilistic Programming Project
The second part of the course concerns applying the techniques learned in the first part, as a project solving a practical problem. We have several types of projects depending on the interests of the student.

For those interested in boosting their CV and potentially getting a student job, we warmly recommend working with one of our industrial partners on a suitable probabilistic programming project. For those interested in working with a more academic-oriented project, we have different interesting problems in Computer Science and Biology.

#### Industrial Projects

| Company | Area   | Ideas |
|:-------:|:------:|:-----:|
| [Relion](https://relion.dk) | Shift-planning AI | Shift planning based on worker availability, historical sales data, weather and holidays.  <br /> <br /> Employee satisfaction quantification based on previously assigned shifts. <br /> <br /> Employee shift assignment based on wishes and need |
| [Paperflow](https://paperflow.com) | Invoice Recognition AI | Talk to us |
| [Hypefactors](https://hypefactors.com) | Media and Reputation Tracking AI | Talk to us |

#### Academic Projects

| Type | Description | Notes/Links |
|:----:|:-----------:|:-----:|
| Computer Science | Implement a predictive scoring model for your favourite sports game, inspired by FiveThirtyEight. | [FiveThirtyEight Methodology and Models](https://fivethirtyeight.com/tag/methodology/) |
| Computer Science | Implement a ranking system for your favourite video or board game, inspired by Microsoft TrueSkill. | [Microsoft TrueSkill Model](https://www.microsoft.com/en-us/research/publication/trueskill-2-improved-bayesian-skill-rating-system/) <br /> <br /> Can be implemented in [Infer.NET](https://dotnet.github.io/infer/) using [Expectation Propagation](https://arxiv.org/pdf/1412.4869.pdf) |
| Computer Science | Use Inference Compilation in PyProb to implement a CAPTCHA breaker or a Spaceship Generator | [Inference Compilation](https://arxiv.org/pdf/1610.09900.pdf) and [PyProb](https://pyprob.readthedocs.io/en/latest/#). You can use the experimental [PyProb bindings for Java](https://github.com/ahmadsalim/pyprob_java). <br /> <br /> [CAPTCHA breaking](https://arxiv.org/pdf/1610.09900.pdf) with [Oxford CAPTCHA Generator](https://github.com/gbaydin/OxCaptcha). <br /> <br />  [Spaceship Generator](https://dritchie.github.io/pdf/sosmc.pdf) |
| Computer Science | Implement asterisk corrector suggested by XKCD | [XKCD #2337: Asterisk Correction](https://xkcd.com/2337/) |
| Biology | Auto-Encoders or Deep Markov Models for Protein Folding | [Deep Markov Model](https://arxiv.org/pdf/1609.09869.pdf) <br /> <br /> [Pyro Deep Markov Model](https://pyro.ai/examples/dmm.html) |
| Biology | Inference Compilation for Ancestral Reconstruction | [Inference Compilation](https://arxiv.org/pdf/1610.09900.pdf) and [PyProb](https://pyprob.readthedocs.io/en/latest/#). Talk to us for details.  | 
| Biology | Recurrent Causal Effect VAE for modelling mutations in proteins | Talk to us for details. |

## Recommendations
* Sometimes sampling can be slow on the CPU for complex models, so try using Google Colab and GPUs and see if that provides an improvement.


### Acknowledgements
This course has been developed by [Thomas Hamelryck](https://github.com/thamelry) and [Ahmad Salim Al-Sibahi](https://github.com/ahmadsalim). Thanks to [Ola Rønning](https://github.com/olaronning) for suggesting the [Variational Auto Encoders Foundations](https://arxiv.org/abs/1906.02691) paper instead of [Auto-Encoding Variational Bayes](https://arxiv.org/pdf/1312.6114.pdf) which we originally proposed to read on week 3. Thanks to Richard Michael for testing out the course and provide us with valuable feedback on the content!
