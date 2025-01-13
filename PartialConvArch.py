import torch
import torch.nn as nn
from torchvision import transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO
import base64

class Partial_Conv(nn.Module):
    def __init__(self, input_filters, output_filters, kernel_size=(3,3),stride=1, bias=True, groups=1,dilation=1, bn=True):
        super().__init__()
        self.bn = bn
        padding = (kernel_size[0]//2, kernel_size[1]//2)
        self.input_conv = nn.Conv2d(input_filters,output_filters,kernel_size,stride,padding,dilation,groups,bias)
        self.mask_conv = nn.Conv2d(input_filters,output_filters,kernel_size,stride,padding,dilation,groups, False)

        self.window_size = kernel_size[0]*kernel_size[1]
        nn.init.constant_(self.mask_conv.weight, 1.0)
        nn.init.kaiming_normal_(self.input_conv.weight, a = 0, mode='fan_in')

        if self.bn:
            self.bath_normalization = nn.BatchNorm2d(output_filters)
        for param in self.mask_conv.parameters():
            param.requires_grad = False
    
    def forward(self, input, mask):
        output = self.input_conv(input * mask)

        with torch.no_grad():
            output_mask = self.mask_conv(mask)

        mask_ratio = self.window_size / (output_mask + 1e-8)
        output_mask = torch.clamp(output_mask, 0, 1)
        mask_ratio = mask_ratio * output_mask
        output = output * mask_ratio


        return output, output_mask

class EncodingLayer(nn.Module):
    def __init__(self, in_filters, out_filters, kernel=3, stride=1, bias=False, bn=True):
        super().__init__()
        self.in_filters = in_filters
        self.out_filters = out_filters
        self.activation = nn.ReLU()
        self.conv = Partial_Conv(self.in_filters, self.out_filters,(kernel,kernel),stride, bias = bias, bn=bn)

    def forward(self, input, input_mask):
        output, output_mask = self.conv(input, input_mask)
        output = self.activation(output)
        return output, output_mask

class DecodingLayer(nn.Module):
    def __init__(self, in_filters, out_filters, bn=True, activation=True, bias=False):
        super().__init__()
        self.in_filters = in_filters
        self.out_filters = out_filters
        self.up_image = nn.Upsample(scale_factor=(2,2))
        self.up_mask = nn.Upsample(scale_factor=(2,2))
        self.activate = activation
        self.conv = Partial_Conv(self.in_filters, self.out_filters, bias=bias,bn=bn)
        self.act = nn.LeakyReLU(negative_slope=0.2)
    
    def forward(self, input_image, input_mask, shared_image, shared_mask):
        upscaled_image = F.interpolate(input_image, scale_factor=2)
        upscaled_mask = F.interpolate(input_mask, scale_factor=2)
        combined_image = torch.cat([upscaled_image, shared_image], dim=1)
        combined_mask = torch.cat([upscaled_mask, shared_mask], dim=1)
        output_image, output_mask = self.conv(combined_image, combined_mask)
        if self.activate:
            output_image = self.act(output_image)
        return output_image, output_mask

class PartialConvUNet(nn.Module):
    def __init__(self, input_size = 256, layers = 7):
        if 2**(layers + 1) != input_size:
            raise AssertionError
        super().__init__()
        self.layers = layers 
        self.encoder_1 = EncodingLayer(3, 64, 7, 2, bn=False)
        self.encoder_2 = EncodingLayer(64, 128, 5, 2)
        self.encoder_3 = EncodingLayer(128, 256, 3, 2)
        self.encoder_4 = EncodingLayer(256, 512, 3, 2)

        for i in range(5, layers + 1):
            name = f"encoder_{i}"
            setattr(self, name, EncodingLayer(512, 512, 3, 2))

        for i in range(5, layers + 1):
            name = f"decoder_{i}"
            setattr(self, name, DecodingLayer(512+512, 512))

        self.decoder_4 = DecodingLayer(512 + 256, 256)
        self.decoder_3 = DecodingLayer(256 + 128, 128)
        self.decoder_2 = DecodingLayer(128 + 64, 64)
        self.decoder_1 = DecodingLayer(64 + 3, 3, bn=False, activation=False, bias=True)

    def forward(self, input_x, mask):
        encoder_dict = {}
        mask_dict = {}
        key_prev = "h_0"
        encoder_dict[key_prev], mask_dict[key_prev] = input_x, mask

        for i in range(1, self.layers + 1):
            encoder_key = f'encoder_{i}'
            key = f"h_{i}"
            encoder_dict[key], mask_dict[key] = getattr(self, encoder_key)(encoder_dict[key_prev], mask_dict[key_prev])
            key_prev = key
        
        out_key = f"h_{self.layers}"
        out_data, out_mask = encoder_dict[out_key], mask_dict[out_key]

        for i in range(self.layers, 0, -1):
            encoder_key = f"h_{i-1}"
            decoder_key = f"decoder_{i}"
            out_data, out_mask = getattr(self, decoder_key)(out_data, out_mask, encoder_dict[encoder_key], mask_dict[encoder_key])
        return out_data

def create_mask(mask_data):
    print(mask_data)
    header, encoded = mask_data.split(",", 1)

    # Decode the Base64 string
    image_data = base64.b64decode(encoded)

    # Convert the binary data to a PIL image
    image = Image.open(BytesIO(image_data))

    # Define a transformation to convert the PIL image to a PyTorch tensor
    transform = transforms.ToTensor()
    image_tensor = transform(image)
    # image_tensor = image_tensor.permute(1, 2, 0)  # Change to (H, W, C)
    # Normalize to [0, 1] range if not already
    image_tensor = image_tensor.clamp(0, 1)
    # Display using matplotlib
    display = image_tensor.permute(1,2,0)
    plt.imshow(display.numpy())  # Convert tensor to NumPy array for Matplotlib
    plt.axis('off')  # Turn off axis
    plt.show()
    # Convert the image to a PyTorch tensor

    print(image_tensor.shape)
    return image_tensor[:3, :, :]

def getinput(image, mask_data):
    mask = create_mask(mask_data).to(dtype=torch.uint8)
    to_tensor = transforms.ToTensor()
    image = to_tensor(image)
    masked_image = torch.bitwise_and(image, mask)
    return masked_image, mask