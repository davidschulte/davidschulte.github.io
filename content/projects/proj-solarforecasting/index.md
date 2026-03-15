---
title: "Solar Forecasting"
date: 2022-03-01
draft: false
description: "Analyzing and forecasting daily solar power generation in Germany using time series models."
---

# German Solar Power Forecasting

Analyzing and forecasting daily solar power generation in Germany using time series models.

<!-- - <a href="https://github.com/davidschulte/solar-forecasting" -->
<!--   class="icon brands alt fa-github"><span class="label">GitHub</span>    -->
<!--   Code</a> -->
<!-- - <a href="Forecasting%20of%20Solar%20Power%20Generation.pdf" -->
<!--   class="icon alt fa-file-pdf"><span class="label">Document</span>  Course -->
<!--   Paper</a> -->


## Motivation

Solar energy is one of the fundaments of our transition toward green
energy. With all its advantages over other power sources, solar energy
has one drawback that complicates its use: It only gets generated when
it's sunny. To utilize an efficient mix of energy sources, it is
important to forecast their availability in the future.

## Data

The data used is the daily solar power generation in Germany by its four
network operators (50Hertz, Amprion, Tennet and Transnet BW). The data
is provided by Fraunhofer Institute for Solar Energy Systems ISE and is
publicly available on [Energy-Charts](https://energy-charts.info/).

## Approach

To model the process, the following steps were undertaken:

1.  Transformation of the distribution of the data
2.  Modeling of the seasonality
3.  Modeling of the short-term effects using an ARIMA model


{{< include-notebook file="content/projects/proj-solarforecasting/notebook.md" >}}
<!-- {{< include-notebook file="content/projects/proj-solarforecasting/notebook.md" >}} -->
## Learnings and outlook

Even with limited knowledge about the underlying process, it is possible
to model seasonality fairly easily. We do not need deep learning but can
resort to a statistical time series model.  
The scope of the project was fairly narrow and given more time, the
following steps should be performed:

- Retransformation of the data to its original distribution
- Out-of-sample predictions
- Thorough evaluation of the performance and generality of the model

Additionally, it would be interesting to also forecast power generation
in the short term. To do this, one could incorporate different sources,
like weather forecasts.
