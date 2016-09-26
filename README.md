# Breaking-Cifar
Tests and tries. Experiments I've ran with a modified code from here https://github.com/szagoruyko/wide-residual-networks. Best result so far ~18.3% error on Cifar-100. Which might be just about the best result so far overall. ᕙ(^▿^-ᕙ) That model uses 80m parameters. Uses plain Relu, so it's likely it can easily be improved by using some smarter activation function (and, possibly, other appropriate initialization). View `notebooks/visualize.ipynb` to find how the trainings went.

If you'll notice any bugs, please, let know.

One "bug" known by now is use of the raw dataset without adjusting mean and deviations. Should get even better results if those are adjusted with something like


```
cifar10 = torch.load("./datasets/cifar10.t7")

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

prepare( cifar10.trainData.data , cifar10.testData.data  )
prepare( cifar10.trainData.data , cifar10.trainData.data )

torch.save("./datasets/cifar10_mean_std.t7", cifar10)
```
