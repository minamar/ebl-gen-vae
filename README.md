# ebl-gen-vae

The goal of this project is to create emotional body language animations for Pepper, by utilizing algorithms of deep learning. 
As a training set we use a small set of animations that have been designed by professional robot animators with a creative process approach. 
We want to explore if we can use generative models such as Variational Autoencoder, 
or LSTMs to learn underlying properties of these examples and generate new animations.

### Training Set 
The initial training set contains 36 animations of different duration each
(in seconds min=2.46, max=25.18, and mean=5.88). The animators select different points of a timeline
to position keyframes that represent robot postures (configurations of 17 joints values).
These keyframes can be thought as the most salient postures of a motion. 
Subsequently the intermediate configurations are derived with Bezier interpolation through the keyframes.

### Labels
The animators have also assigned a descriptive tag to each of these animations, such as "happy" or "sad". 
However these tags were assigned in a subjective way, following the discrete emotions theory. We conducted an experiment where we asked participants
to evaluate each animation in terms of valence (how positive or negative the emotion 
depicted is) and arousal (the intensity of the expression), following the dimensional theory
 of emotion. These scores range from 0 to 1. Besides the animation tag, and the valence/arousal continuous values, 
we also created another categorical label. We discretized the valence/arousal space in three
 levels for valence (positive, neutral, negative) and three levels for arousal (excited, calm, tired),
 and we derived 3x3 combinations of the two metrics, that is 9 categories such as Neutral/Calm, Positive/Excited, Negative/Tired etc.
Therefore we have three different options to label the training set.

### Data preprocessing 
The training data have been corrected for joint values exceeding the limits. On the robot, these 
values are corrected automatically, but replacing them with the limits beforehand was necessary for 
the training because in the next step we normalized the values in (0,1), since different joints have different 
ranges. Finally, we augmented the training set with the mirrored version of each animation 
with respect to left and right orientation. Without this augmentation of the training set, 
the generated animations where biased towards the one side of the body.
   
### Variational Autoencoder with all time-frames (previously branch EBLgen_VAE)
In this branch, the goal is to experiment with a variational autoencoder 
for the generation of new animations. The VAE cannot capture the
temporal dynamics of the animations as the LSTM does, but it can learn a compact latent 
representation from the training set which we can sample with interpolation 
to generate new animations. For this experiment we use all the time-frames, not just the keyframes. For the implementation of the VAE network we use 
the library nhemion/tfmodellib/vae.py with tensorflow which constructs the encoder and decoder with MLPs.
Different sampling strategies and attribute vectors are tested.

### Future work
 - Variational autoencoder
   - Mix latent vectors between functional and emotional animations
   - Interface for latent space sampling
   - Pytorch implementation of the VAE
 - VAE + LSTM 
  

    
