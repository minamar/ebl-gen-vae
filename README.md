# Generating Robotic Emotional Body Language Using VAEs 


## Overview

We train a Conditional Variational Autoencoder 
for the generation of novel Robotic Emotional Body Language (REBL) for a Pepper
robot. The training set has been compiled with a selection of 36 hand-designed 
animations from a broader animation library created by 
[SoftBank Robotics](https://www.softbankrobotics.com).  

The selected animations were chosen to convey emotion through body motion, 
eye LEDs colour and patterns, and non-linguistic sounds. A user study (N=20) 
was conducted to derive reliable core affect labels for each animation. Core 
affect is defined by two dimensions: valence (displeased to pleased), 
and arousal (deactivated to activated). Valence and arousal ratings
were collected as continuous values in the interval [0,1] and they were aggregated 
across participants, for each animation [[1]](#1). 

Subsequently, the audio data were excluded and the rest of the modalities 
(motion and eye LEDs) were recorded with a sampling rate of 25 fps, augmented, 
and used to train an emotion-agnostic Variational Autoencoder [[2]](#2), 
and later a Conditional Variational Autoencoder for the generation of targeted 
emotion animations [[3]](#3). A second user study was conducted to evaluate a set
of generated animations in terms of valence, arousal, and dominance [[3]](#3). 


## References

<a id="1">[1]</a> 
Marmpena M., Lim, A., and Dahl, T. S. (2018). How does the robot feel? Perception of valence and
arousal in emotional body language. Paladyn, Journal of Behavioral Robotics, 9(1), 168-182.
[DOI](https://doi.org/10.1515/pjbr-2018-0012)

<a id="2">[2]</a> 
Marmpena M., Lim, A., Dahl, T. S., and Hemion, N. (2019). Generating robotic emotional body
language with Variational Autoencoders. In Proceedings of the 8th International Conference
on Affective Computing and Intelligent Interaction (ACII), pages 545–551.
[DOI](https://doi.org/10.1109/ACII.2019.8925459)

<a id="3">[3]</a> 
Marmpena M., Garcia, F., and Lim, A. (2020). Generating robotic emotional body language of
targeted valence and arousal with Conditional Variational Autoencoders. In Companion of
the 2020 ACM/IEEE International Conference on Human-Robot Interaction, HRI ’20, page
357–359. [DOI](https://doi.org/10.1145/3371382.3378360)










