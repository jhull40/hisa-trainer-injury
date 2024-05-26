# HISA Trainer Injury Risk 

**Objective**: The goal of this project is to create a model to assign an injury risk score associated with each trainer. This model utilizes past racing, workout, medication, and injury data to predict the future injury risk of the trainer.

This code was created by [James Hull](mailto:hulljames96@gmail.com), in associated with NYTHA and HISA. 

Last updated: 2024-05-25

## Outline <br>
1. Baseline Model
2. Historical Estimates
3. Risk Model
4. Implementation
5. Notes and Concerns

## Baseline Model

Initially, a baseline model was created to estimate the probability of a horse becoming injured (DNF) in a race, only accounting for very basic features. This model serves two main goals. It creates a strong baseline for which to compare more sophisticated features and model types against. It also acts as an "expected" injury rate, allowing an analyst to identify trainers who consistently over(under) perform relative to expectations.

## Historical Estimates
While not factoring in severity of injury, a process was created to collect historical injury rates. This data was used in the model building and also could be useful in other analyses. To control for trainers with vastly different numbers of horses, a Bayesian smoothing method was used, which also allowed for estimates of confidence intervals.



 
