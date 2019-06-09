import torch.nn as nn

# convolutional layer helper function. From the DCGAN notebook
def conv(in_channels, out_channels, kernel_size=4, stride=2, padding=1, batch_norm=True):
    ''' Create a convolutional layer with optional batch normalization.
        Used in the Discriminator class.
    '''
    layers = []
    # append a convolutional layer:
    conv_layer = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
    layers.append(conv_layer)

    if batch_norm:
        # append a batch norm layer:
        layers.append(nn.BatchNorm2d(out_channels))

    return nn.Sequential(*layers)

# transposed-convolutional layer helper function. From the DCGAN notebook
def deconv(in_channels, out_channels, kernel_size=4, stride=2, padding=1, batch_norm=True):
    ''' Create a transposed-convolutional layer, with optional batch-normalization
        Used in the Generator class.
    '''
    layers = []
    # append a transposed-convolutional layer:
    trans_conv_layer = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=True)
    layers.append(trans_conv_layer)

    if batch_norm:
        # append a batch-normalization layer
        layers.append(nn.BatchNorm2d(out_channels))

    return nn.Sequential(*layers)
