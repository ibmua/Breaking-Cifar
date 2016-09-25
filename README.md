# Breaking-Cifar
Tests and tries. Experiments I've ran with a modified code from here https://github.com/szagoruyko/wide-residual-networks. Best result so far ~18.3% error on Cifar-100. Which might be just about the best result so far overall. ᕙ(^▿^-ᕙ) That model uses 80m parameters. Uses plain Relu, so it's likely it can easily be improved by using some smarter activation function (and, possibly, other appropriate initialization). View `notebooks/visualize.ipynb` to find how the trainings went.

If you'll notice any bugs, please, let know.

One "bug" known by now is use of the raw dataset without adjusting mean and deviations. Should get even better results if those are adjusted.
