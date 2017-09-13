# GAN

Playground to use GAN for data/MC morphing

## Dependencies

* Standard python ML echosystem (numpy, pandas, matplotlib, sckit-learn,
  keras, theano, thensorflow)
* Keras adversarial
  https://github.com/bstriner/keras-adversarial/tree/master/keras_adversarial


## Examples
* Toys
  * `toy_1D.ipynb`
  * `toy_1D_conditional.ipynb`
  * `toy_2D.ipynb`

* CMS Zee
  * `cms_zee_conditional.ipynb` simple example using naive freeze/unfreeze parameters in fit  
  * `cms_zee_conditional_keras_advesarial.ipynb` same as above using keras adversarial for fitting
  * `cms_zee_conditional.ipynb` parametrized notebook, that can be run from command line using
    `bin/nb_batch <input_nb> [output_nb] --Parameters.name=VALUE`
    
