input_dim = 14
output_dim = 10
stride = 1
padding = 0
dilation = 1
kernel = (output_dim * stride )-1
kernel = kernel - input_dim -2*padding
kernel += 1
kernel = kernel / -dilation +1
print(kernel)