# Preface

This book is intended to cover some advanced modelling techniques applied to equity **investment strategies** that are built on **firm characteristics**. The content is threefold. First, we try to simply explain the ideas behind most mainstream machine learning algorithms that are used in equity asset allocation. Second, we mention a wide range of academic references for the readers who wish to push a little further. Finally, we provide hands-on **Python** code samples that show how to apply the concepts and tools on a realistic dataset which we share to encourage **reproducibility**.


This book is intended to cover some advanced modelling techniques applied to equity **investment strategies** that are built on **firm characteristics**. The content is threefold. First, we try to simply explain the ideas behind most mainstream machine learning algorithms that are used in equity asset allocation. Second, we mention a wide range of academic references for the readers who wish to push a little further. Finally, we provide hands-on **Python** code samples that show how to apply the concepts and tools on a realistic dataset which we share to encourage **reproducibility**.

## What this book is not about

This book deals with machine learning (ML) tools and their applications in factor investing. Factor investing is a subfield of a large discipline that encompasses asset allocation, quantitative trading and wealth management. Its premise is that differences in the returns of firms can be explained by the characteristics of these firms. Thus, it departs from traditional analyses which rely on price and volume data only, like classical portfolio theory à la {cite:p}`markowitz1952portfolio`, or high frequency trading. For a general and broad treatment of Machine Learning in Finance, we refer to {cite:p}`dixon2020machine`.


The topics we discuss are related to other themes that will not be covered in the monograph. These themes include:

- Applications of ML in **other financial fields**, such as **fraud detection** or **credit scoring**. We refer to {cite:p}`ngai2011application` and {cite:p}`baesens2015fraud` for general purpose fraud detection,
to {cite:p}`bhattacharyya2011data` for a focus on credit cards and to {cite:p}`ravisankar2011detection`
and {cite:p}`abbasi2012metafraud` for studies on fraudulent financial reporting. On the topic of credit scoring,
{cite:p}`wang2011comparative` and {cite:p}`brown2012experimental` provide overviews of methods and some empirical results.
Also, we do not cover ML algorithms for data sampled at higher (daily or intraday) frequencies (microstructure models, limit order book).
The chapter from {cite:p}`kearns2013machine` and the recent paper by {cite:p}`sirignano2019universal` are good introductions on this topic.
- Use cases of alternative datasets that show how to leverage textual data from social media, satellite imagery, or credit card logs to predict sales, earning reports, and, ultimately, future returns.
The literature on this topic is still emerging (see, e.g., {cite:p}`blank2019using`,
{cite:p}`jha2019implementing` and {cite:p}`ke2019predicting`) but will likely blossom in the near future.
- Technical details of machine learning tools. While we do provide some insights on specificities of some approaches
(those we believe are important), the purpose of the book is not to serve as reference manual on statistical learning.
We refer to {cite:p}`hastie2009elements`, {cite:p}`cornuejols2018apprentissage` (written in French),
{cite:p}`james2013introduction` (coded in R!) and {cite:p}`mohri2018foundations` for a general treatment on the subject.
Moreover, {cite:p}`du2013neural` and {cite:p}`goodfellow2016deep` are solid monographs on neural networks particularly
and {cite:p}`sutton2018reinforcement` provide a self-contained and comprehensive tour in reinforcement learning.
Finally, the book does not cover methods of natural language processing (NLP) that can be used to evaluate sentiment
which can in turn be translated into investment decisions. This topic has nonetheless been trending lately
and we refer to {cite:p}`loughran2016textual`, {cite:p}`cong2019analyzing`, {cite:p}`cong2019textual` and {cite:p}`gentzkow2019text` for recent advances on the matter.



## The targeted audience

Who should read this book? This book is intended for two types of audiences. First, **postgraduate students** who wish to pursue their studies in quantitative finance with a view towards investment and asset management. The second target groups are **professionals from the money management industry** who either seek to pivot towards allocation methods that are based on machine learning or are simply interested in these new tools and want to upgrade their set of competences. To a lesser extent, the book can serve **scholars or researchers** who need a manual with a broad spectrum of references both on recent asset pricing issues and on machine learning algorithms applied to money management. While the book covers mostly common methods, it also shows how to implement more exotic models, like causal graphs (Chapter 14), Bayesian additive trees (Chapter 9), and hybrid autoencoders (Chapter 7).

The book assumes basic knowledge in **algebra** (matrix manipulation), **analysis** (function differentiation, gradients), **optimization** (first and second order conditions, dual forms), and **statistics** (distributions, moments, tests, simple estimation method like maximum likelihood). A minimal **financial culture** is also required: simple notions like stocks, accounting quantities (e.g., book value) will not be defined in this book. Lastly, all examples and illustrations are coded in Python. A minimal culture of the language is sufficient to understand the code snippets which rely heavily on the most common functions and libraries.


## How this book is structured

The book is divided into four parts.


Part I gathers preparatory material and starts with notations and data presentation (Chapter 1), followed by introductory remarks (Chapter 2). Chapter 3 outlines the economic foundations (theoretical and empirical) of factor investing and briefly sums up the dedicated recent literature. Chapter 4 deals with data preparation. It rapidly recalls the basic tips and warns about some major issues.

Part II of the book is dedicated to predictive algorithms in supervised learning. Those are the most common tools that are used to forecast financial quantities (returns, volatilities, Sharpe ratios, etc.). They range from penalized regressions (Chapter 5), to tree methods (Chapter 6), encompassing neural networks (Chapter 7), support vector machines (Chapter 8) and Bayesian approaches (Chapter 9).

The next portion of the book bridges the gap between these tools and their applications in finance. Chapter 10 details how to assess and improve the ML engines defined beforehand. Chapter 11 explains how models can be combined and often why that may not be a good idea. Finally, one of the most important chapters (Chapter 12) reviews the critical steps of portfolio backtesting and mentions the frequent mistakes that are often encountered at this stage.

The end of the book covers a range of advanced topics connected to machine learning more specifically. The first one is **interpretability**. ML models are often considered to be black boxes and this raises trust issues: how and why should one trust ML-based predictions? Chapter 13 is intended to present methods that help understand what is happening under the hood. Chapter 14 is focused on **causality**, which is both a much more powerful concept than correlation and also at the heart of many recent discussions in Artificial Intelligence (AI). Most ML tools rely on correlation-like patterns and it is important to underline the benefits of techniques related to causality. Finally, Chapters 15 and 16 are dedicated to non-supervised methods. The latter can be useful, but their financial applications should be wisely and cautiously motivated.

## Coding instructions

One of the purposes of the book is to propose a large-scale tutorial of ML applications in financial predictions and portfolio selection. Thus, one keyword is **REPRODUCIBILITY**! In order to duplicate our results (up to possible randomness in some learning algorithms), you will need running versions of Python and Anaconda on your computer.

[texte du lien](https://)A list of the packages we use can be found in the Table below.

|Package|Purpose|Chapter(s)|short|
|---|---|---|---|
|pandas|Multiple usage|almost all|pd|
|urllib.request|Data from url|3||
|statsmodels|Statistical regression|3, 4, 14,15, 16|sm|
|numpy|Multiple usage|almost all|np|
|matplotlib|Plotting|almost all|plt|
|seaborn|Plotting|4,6,15|sns|
|IPython.display|Table display|4||
|sklearn|Machine learning|5,6,7,8,9,10,11,15||
|xgboost|Machine learning|6,10,12|xgb|
|tensorflow|Machine learning|7,11|tf|
|plot_keras_history|Plotting|7,11||
|xbart|Bayesian trees|9||
|skopt|Bayesian optimisation|10||
|cvxopt|Optimisation|11|cvx|
|datetime|date functions|12||
|itertools|Iterate utils|12||
|scipy|Optimisation|12||
|random|Statistics|13||
|collections|Utils|13||
|lime|Interpretability|13||
|shap|Interpretability|13||
|dalex|Interpretability|13||
|causalimpact|Causality|14||
|cdt|Causality|14||
|networks|Graph / Causality|14|nx|
|icpy|Causality|14|icpy|
|pca|Plotting pca|15||



As much as we could, we created short code chunks and commented each line whenever we felt it was useful. Comments are displayed at the end of a row and preceded with a single hastag #.

The book is constructed as a very big notebook, thus results are often presented below code chunks. They can be graphs or tables. Sometimes, they are simple numbers and are preceded with two hashtags ##. The example below illustrates this formatting.


The book can be viewed as a very big tutorial. Therefore, most of the chunks depend on previously defined variables. When replicating parts of the code (via online code), please make sure that the environment includes all relevant variables. One best practice is to always start by running all code chunks from Chapter 1. For the exercises, we often resort to variables created in the corresponding chapters.



## Acknowledgments

The core of the book was originally prepared for a series of lectures given by one of the authors to students of master’s degrees in finance at EMLYON Business School and at the Imperial College Business School in the Spring of 2019. We are grateful to those students who asked fruitful questions and thereby contributed to improve the content of the book. 

For the first edition, we were grateful to Bertrand Tavin and Gautier Marti for their thorough screening of the book. We also thanked Eric André, Aurélie Brossard, Alban Cousin, Frédérique Girod, Philippe Huber, Jean-Michel Maeso, Javier Nogales and for friendly reviews; Christophe Dervieux for his help with bookdown; Mislav Sagovac and Vu Tran for their early feedback; Lara Spieker, John Kimmel for making it happen and Jonathan Regenstein for his availability, no matter the topic. Lastly, we were grateful for the anonymous reviews collected by John, our original editor.

The second version has benefitted from a large number of conversations with academics and practitioners, including Arnaud Battistella, Jean-Charles Bertrand, Guillaume Chevallier, Jean-Michel Maeso (again!), Nicholas McLoughlin, Thomas Raffinot.

Finally, we must acknowledge great help by several LLMs for the processing of references while recycling old files, especially Claude. Life is not the same without them, yet we must use them with caution.

