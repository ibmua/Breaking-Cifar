# Update
This is how I tried to discover what was later published by Facebook FAIR team of Kaiming He and co. in an update to their state-of-the-art ResNet architecture https://arxiv.org/pdf/1611.05431.pdf . I failed for a couple of reasons, like lack of computational resources (posessing only 2 GPUs) and overestimation of importance of Sergey's finding in https://arxiv.org/pdf/1605.07146.pdf of impact of block types - which I'm not sure right now, would even be the same had he not, unfortunately to himself and plenty of the readers, especially the ones who discovered networks inferior to resNet, but, because of that mistake, had an impression that their networks were superior, and so to readers of the papers published by that crowd of Sergeys paper's readers, preprocessed his data with whitening. So, "a message to a younger generation", be very very attentive to your data. And try things if you think they might work even if the stuff you read suggests it probably won't. Turns out that 1-3-1 blocks were exactly the way to go.

# Breaking-Cifar
Tests and tries. Experiments I've ran with a modified code from here https://github.com/szagoruyko/wide-residual-networks. Best result so far ~18.3% error on Cifar-100. Which might be just about the best result so far overall. ᕙ(^▿^-ᕙ) That model uses 80m parameters. Uses plain Relu, so it's likely it can easily be improved by using some smarter activation function (and, possibly, other appropriate initialization). View `notebooks/visualize.ipynb` to find how the trainings went.

If you'll notice any bugs, please, let know.

One "bug" known by now is use of the raw dataset without adjusting mean and deviations. Should get even better results if those are adjusted with something like


```
cifar = torch.load("./datasets/cifar.t7")

function prepare( d , result )
    m = torch.mean( torch.reshape(d , (#d)[1], 3,32*32), 3)
    s = torch.std ( torch.reshape(d , (#d)[1], 3,32*32), 3)

    m = torch.reshape( torch.mean( m, 1) , 3 )
    s = torch.reshape( torch.mean( s, 1) , 3 )
    print(#m)
    print(m)
    print(#s)
    print(s)

    for i=1, (#result)[1] do
        result[i][1]:csub( m[1] )
        result[i][2]:csub( m[2] )
        result[i][3]:csub( m[3] )

        result[i][1]:div ( s[1] )
        result[i][2]:div ( s[2] )
        result[i][3]:div ( s[3] )
        end
    end

prepare( cifar.trainData.data , cifar.testData.data  )
prepare( cifar.trainData.data , cifar.trainData.data )

torch.save("./datasets/cifar_mean_std.t7", cifar)
```
