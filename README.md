# Sample-specific Bayesian Networks
Framework for estimating the structures and parameters of Bayesian networks (DAGs) at per-sample or per-patient resolution, formally dubbed NOTMAD (NO-TEARS Mixtures of Archetypal DAGs)

Developed by Dr. Ben Lengerich (MIT) and Caleb Ellington (CMU)

## Install and Use NOTMAD
```
pip install git+https://github.com/cnellington/SampleSpecificDAGs.git
```
Then in your code use
```
from notmad.notmad import NOTMAD
```
Load your Context data `C` and Target data `X`, specify hyperparameters, and train the model
```
C, X = your_data()
sample_specific_loss_params = {'l1': 0., 'alpha': 2e1, 'rho': 1e0}
archetype_loss_params = {'l1': 0., 'alpha': 1e-1, 'rho': 1e-2}

model = NOTMAD(C.shape, X.shape, k_archetypes, 
                sample_specific_loss_params, archetype_loss_params)
model.fit(C, X, batch_size=1, epochs=50)
```
Then use it to estmate samples-specific networks! Simple as that.
```
ss_networks = model.predict_w(C_unseen)
```

## Reproduce Experiments
Run one round of simulations with `run_experiments.sh` to compare population-based, cluster-based, and sample-specific network inference.
