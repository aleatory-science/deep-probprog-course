## Syllabus

### Part I - Introduction to Deep Probabilistic Programming

| Week | Topic | Exercise | Links |
|:----:|:-----:|:--------:|:-----:|
| 1    | Introduction to Bayesian Inference | Read Pattern Recognition and Machine Learning (PRML), Sections 1.1-1.3, 1.5-1.6 & 2-2.3.4 (inclusive ranges), Intro to Bayesian updating paper, and Pyro paper. <br /> <br /> Form up groups and ask a question for each chapter/paper you have read. | [Pattern Recognition and Machine Learning](https://www.microsoft.com/en-us/research/publication/pattern-recognition-machine-learning/)  <br /> <br /> [Bayesian Updating Paper](https://arxiv.org/pdf/1411.5018.pdf) <br /> <br />[Pyro Paper](http://jmlr.org/papers/volume20/18-403/18-403.pdf) |
| 2  | Variational Inference | Read the Variational Inference paper and Pyro tutorials on Stochastic Variational Inference (SVI). Ask three questions about them. <br /> <br /> Use Pyro’s Variational Inference support to implement the kidney cancer model. See worksheet and Bayesian Data Analysis 3rd Edition (BDA3) Section 2.7. | [Variational Inference Paper](https://arxiv.org/pdf/1601.00670.pdf) <br /> <br /> [Worksheet](week2/DPP_course___Week_2_worksheet.pdf) <br /> <br /> [Bayesian Data Analysis](http://www.stat.columbia.edu/~gelman/book/) <br /> <br /> Pyro SVI tutorial: [Part I](https://pyro.ai/examples/svi_part_i.html) and [Part II](https://pyro.ai/examples/svi_part_ii.html) <br /> <br /> [Pyro Website](https://pyro.ai) |
| 3 | Hamiltonian Monte Carlo | Read paper on Hamiltonian Monte Carlo and blog post on gradient-based Markov Chain Monte Carlo (MCMC). Look at the source code for Mini-MC. <br /> <br /> Ask a question each for HMC, the Mini-MC implementation and discrete variable marginalization. <br /> <br /> Implement Bayesian Change-point model in Pyro using NUTS. | [Hamiltonian Monte Carlo Paper](https://arxiv.org/pdf/1701.02434.pdf) <br /> <br /> [Gradient-based MCMC](https://elevanth.org/blog/2017/11/28/build-a-better-markov-chain/) <br /> <br /> [Mini-MC implementation](https://github.com/ColCarroll/minimc)<br /> <br /> [Change-point model](https://cscherrer.github.io/post/bayesian-changepoint/) <br /> <br /> [Pyro NUTS Example](http://pyro.ai/examples/mcmc.html) |
| 4 | Hidden Markov Models and Discrete Variables. | Read Paper on Hidden Markov Models and ask three questions about it. <br /> <br /> Read Pyro tutorials on Discrete Variables and Gaussian Mixture Models. <br /> <br /> Read Pyro Hidden Markov Model code example and describe in your own words what the different models do. <br /> <br /> Add amino acid prediction output to the TorusDBN HMM and show that the posterior predictive distribution of the amino acids matches the one found in data. | [Hidden Markov Models](http://ai.stanford.edu/~pabbeel/depth_qual/Rabiner_Juang_hmms.pdf) <br /> <br /> [Pyro Discrete Variables Tutorial](https://pyro.ai/examples/enumeration.html) <br /> <br />  [Pyro Gaussian Mixture Model Tutorial](https://pyro.ai/examples/gmm.html)   <br /> <br /> [Pyro Hidden Markov Model Example](https://pyro.ai/examples/hmm.html) <br /> <br /> [TorusDBN](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2440424/) <br /> <br /> **Optional:** [Epidemological Inference via HMC](http://pyro.ai/examples/sir_hmc.html) |
| 5 |  Bayesian Regression Models | Read PRML Chapter 3 on Linear Models. <br /> <br /> Ask 3 questions about the chapter. <br /> <br /> Read the Pyro tutorials on Bayesian Regression. <br /> <br /> Solve the weather check exercise in the worksheet. | Pyro Bayesian Regression: [Part I](https://pyro.ai/examples/bayesian_regression.html), [Part II](https://pyro.ai/examples/bayesian_regression_ii.html) <br /> <br />  [Worksheet](week5/DPP_course___Week_5_worksheet.pdf) |
| 6 | Variational Auto-Encoders | Read Variational Auto Encoders (VAE) foundations Chapters 1 & 2, and Pyro tutorial on VAE. Ask three questions about the paper and tutorial. <br /> <br /> Implement Frey Faces model from VAE paper in Pyro. Rely on the existing VAE implementation (see tutorial link). | [Variational Auto Encoders Foundations](https://arxiv.org/abs/1906.02691) <br /> <br /> [Pyro Tutorial on VAE](https://pyro.ai/examples/vae.html) |
| 7 | Deep Generative Models | Read one of these papers: Interpretable Representation VAE, Causal Effect VAE, Deep Markov Model or DRAW (one paper per group). <br /> <br /> Try out the relevant Pyro or PyTorch implementation on your choice of relevant dataset which was not used in the paper. <br /> <br /> Make a small (10 minute) presentation about the paper and your experiences with the implementation. | [Deep Markov Model](https://arxiv.org/pdf/1609.09869.pdf) <br /> <br /> [Interpretable Representation VAE](https://arxiv.org/abs/1611.07492) <br /> <br /> [Causal Effect VAE](https://arxiv.org/abs/1705.08821) <br /> <br /> [DRAW](https://arxiv.org/abs/1502.04623) |


### Part II - Deep Probabilistic Programming Project
The second part of the course concerns applying the techniques learned in the first part, as a project solving a practical problem. We have several types of projects depending on the interests of the student.

For those interested in boosting their CV and potentially getting a student job, we warmly recommend working with one of our industrial partners on a suitable probabilistic programming project. For those interested in working with a more academic-oriented project, we have different interesting problems in Computer Science and Biology.

#### Industrial Projects

| Company | Area   | Ideas |
|:-------:|:------:|:-----:|
| [Relion](https://relion.dk) | Shift-planning AI | Shift planning based on worker availability, historical sales data, weather and holidays.  <br /> <br /> Employee satisfaction quantification based on previously assigned shifts. <br /> <br /> Employee shift assignment based on wishes and need |
| [Paperflow](https://paperflow.com) | Invoice Recognition AI | Talk to us |
| [Hypefactors](https://hypefactors.com) | Media and Reputation Tracking AI | Talk to us |
| ‹Your Company› | ‹Your Area› | Interested in collaboration with our group? contact Ahmad Salim to hear more! |

#### Academic Projects

| Type | Description | Notes/Links |
|:----:|:-----------:|:-----:|
| Computer Science | Implement a predictive scoring model for your favourite sports game, inspired by FiveThirtyEight. | [FiveThirtyEight Methodology and Models](https://fivethirtyeight.com/tag/methodology/) |
| Computer Science | Implement a ranking system for your favourite video or board game, inspired by Microsoft TrueSkill. | [Microsoft TrueSkill Model](https://www.microsoft.com/en-us/research/publication/trueskill-2-improved-bayesian-skill-rating-system/) <br /> <br /> Can be implemented in [Infer.NET](https://dotnet.github.io/infer/) using [Expectation Propagation](https://arxiv.org/pdf/1412.4869.pdf) |
| Computer Science | Use Inference Compilation in PyProb to implement a CAPTCHA breaker or a Spaceship Generator | [Inference Compilation](https://arxiv.org/pdf/1610.09900.pdf) and [PyProb](https://pyprob.readthedocs.io/en/latest/#). You can use the experimental [PyProb bindings for Java](https://github.com/ahmadsalim/pyprob_java). <br /> <br /> [CAPTCHA breaking](https://arxiv.org/pdf/1610.09900.pdf) with [Oxford CAPTCHA Generator](https://github.com/gbaydin/OxCaptcha). <br /> <br />  [Spaceship Generator](https://dritchie.github.io/pdf/sosmc.pdf) |
| Computer Science | Implement asterisk corrector suggested by XKCD | [XKCD #2337: Asterisk Correction](https://xkcd.com/2337/) |
| Computer Science | Implement an inference compilation based program-testing tool like QuickCheck or SmallCheck | [Inference Compilation](https://arxiv.org/pdf/1610.09900.pdf)  <br /> <br /> [QuickCheck](https://www.cs.tufts.edu/~nr/cs257/archive/john-hughes/quick.pdf) <br /> <br /> [SmallCheck](https://www.cs.york.ac.uk/fp/smallcheck/smallcheck.pdf) |
| Computer Science | Magic: The Gathering, Automated Deck Construction. Design a model that finds a good deck automatically based on correlations in existing deck design. Ideas like card substitution models could be integrated too. | [Magic: The Gathering - Meta Site](https://mtgmeta.io) |
| Computer Science | Use probabilistic programming to explore ideas for solving Eternity II (No $2 million prize anymore, but still interesting from a math point of view) | [Eternity II](https://web.archive.org/web/20070624073324/http://uk.eternityii.com/) |
| Biology | Auto-Encoders or Deep Markov Models for Protein Folding | [Deep Markov Model](https://arxiv.org/pdf/1609.09869.pdf) <br /> <br /> [Pyro Deep Markov Model](https://pyro.ai/examples/dmm.html) |
| Biology | Inference Compilation for Ancestral Reconstruction | [Inference Compilation](https://arxiv.org/pdf/1610.09900.pdf) and [PyProb](https://pyprob.readthedocs.io/en/latest/#). Talk to us for details.  | 
| Biology | Recurrent Causal Effect VAE for modelling mutations in proteins | Talk to us for details. |

## Recommendations
* Sometimes sampling can be slow on the CPU for complex models, so try using Google Colab and GPUs and see if that provides an improvement.


### Acknowledgements
This course has been developed by [Thomas Hamelryck](https://github.com/thamelry) and [Ahmad Salim Al-Sibahi](https://github.com/ahmadsalim). Thanks to [Ola Rønning](https://github.com/olaronning) for suggesting the [Variational Auto Encoders Foundations](https://arxiv.org/abs/1906.02691) paper instead of [Auto-Encoding Variational Bayes](https://arxiv.org/pdf/1312.6114.pdf) which we originally proposed to read on week 3. Thanks to Richard Michael for testing out the course and provide us with valuable feedback on the content!
